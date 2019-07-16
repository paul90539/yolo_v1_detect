import os
import argparse
import datetime
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc

slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data, td):
        self.net = net
        self.data = data
        self.test_data = td
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()

        self.variable_to_restore = {}
        # count=0
        # for v in slim.get_variables_to_restore(exclude=['yolo/fc_36']):
        #     self.variable_to_restore[v.op.name] = v
        #     count+=1
        # print("count: " + str(count))
        # self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.variable_to_restore = slim.get_variables_to_restore(exclude=['yolo/fc_36'])
        self.saver = tf.train.Saver(self.variable_to_restore)

        self.saver_for_all_variables = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        # self.variable_to_restore = tf.global_variables()
        # self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_dir, 'yolo')
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)
        self.train_op = slim.learning.create_train_op(
            self.net.total_loss, self.optimizer, global_step=self.global_step)

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        print(self.weights_file)
        if self.weights_file is not None:
            print('Restoring weights from: ' + self.weights_file)
            self.saver.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)

    def train(self):

        train_timer = Timer()
        load_timer = Timer()
        num = 5

        for step in range(1, self.max_iter + 1):

            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}
            # print("image shape: {}, label shape: {}.").format(shape(imagess), shape(labels))
            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:

                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()
                    sum_loss = 0

                    for i in range(num):
                        images_t, labels_t = self.test_data.get()
                        feed_dict_t = {self.net.images: images_t, self.net.labels: labels_t}
                        loss_t = self.sess.run(self.net.total_loss, feed_dict=feed_dict_t)
                        sum_loss += loss_t

                    log_str = ('''{} Epoch: {}, Step: {}, Learning rate: {},'''
                    ''' train_Loss: {:5.3f}, test_Loss: {:.4f}\nSpeed: {:.3f}s/iter,'''
                    ''' Load {:.3f}s/iter, Remain: {}''').format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        self.data.epoch,
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss,
                        sum_loss/num,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter))
                    # log_str = '''{} Epoch: {}, Step: {}, Learning rate: {},'''
                    # ''' Loss: {:5.3f}\nSpeed: {:.3f}s/iter,'''
                    # '''' Load: {:.3f}s/iter, Remain: {}'''.format(
                    #     datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    #     self.data.epoch,
                    #     int(step),
                    #     round(self.learning_rate.eval(session=self.sess), 6),
                    #     loss,
                    #     train_timer.average_time,
                    #     load_timer.average_time,
                    #     train_timer.remain(step, self.max_iter))
                    print(log_str)

                else:
                    #print("11111")
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                self.writer.add_summary(summary_str, step)

            else:
                #print("22222")
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver_for_all_variables.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)

    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def update_config_paths(data_dir, weights_file):
    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')

    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    if args.weights is not None:
        update_config_paths(args.data_dir, args.weights)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet()
    pascal = pascal_voc('train', True)
    pascal_test = pascal_voc('test', True)

    solver = Solver(yolo, pascal, pascal_test)

    print('Start training ...')
    solver.train()
    print('Done training.')


if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
