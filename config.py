import os
import torchvision.transforms as T


DATA_PATH = 'data'
CLASSES_PATH = os.path.join(DATA_PATH, 'classes.json')

# BATCH_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 135
WARMUP_EPOCHS = 0
LEARNING_RATE = 8E-5
ROTATION_ANGLE_RANGE = 90

EPSILON = 1E-6
IMAGE_SIZE = (448, 448)

GRID_SIZE_X = IMAGE_SIZE[1] // 7
GRID_SIZE_Y = IMAGE_SIZE[0] // 7

S = 7       # Divide each image into a SxS grid
B = 2       # Number of bounding boxes to predict
C = 20      # Number of classes in the dataset
