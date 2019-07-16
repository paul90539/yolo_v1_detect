import os

#
# path and dataset parameter
#

DATA_PATH = 'data'

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')

OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')

WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights_2')

WEIGHTS_FILE = None
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

CLASSES = ['motor', 'car']

#CLASSES = ['motor', 'car', 'person', 'truck', 'bike', 'bick', 'bus', 'van']

# CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#            'train', 'tvmonitor']

FLIPPED = True


#
# model parameter
#

IMAGE_SIZE = 448

CELL_SIZE = 7

BOXES_PER_CELL = 2

ALPHA = 0.1

DISP_CONSOLE = False

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# solver parameter
#

GPU = ''

LEARNING_RATE = 0.01

DECAY_STEPS = 300000

DECAY_RATE = 1

STAIRCASE = True

BATCH_SIZE = 30

MAX_ITER = 50000

SUMMARY_ITER = 10

SAVE_ITER = 1000


#
# test parameter
#

THRESHOLD = 0.08

IOU_THRESHOLD = 0.08
