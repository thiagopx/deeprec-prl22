import os

# general
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
SAMPLES_DIR = "/dados/thiagopx/samples/deeprec-prl22"  # replace this in case of generate new samples for training

# DML
SAMPLES_SIZE_DML = [32, 64]
FEAT_BLOCK = "fire3"
FEAT_DIM = 128
POOL_SIZE = 1
MARGIN = 1.0
LEARNING_RATE_DML = 0.1
NUM_EPOCHS_DML = 100
VSHIFT = 10
INPUT_SIZE_H = 3000

# CL
SAMPLES_SIZE_CL = [32, 32]

# DML/CL shared params
BATCH_SIZE = 256
REPRESENTATION = "rgb"
MAX_SAMPLES_PER_CLASS = 1000
RADIUS_NOISE = 2
NUM_STRIPS = 30
VAL_PERC = 0.1
RHO_BLACK = 0.2
STRIDE = 2

# Experiments params
SEEDS = [
    209652396,
    398764591,
    924231285,
    1478610112,
    441365315,
]
SEEDS_RANDOM_EXP = [1537364731, 192771779, 1491434855, 1819583497, 530702035]
WORKLOAD_PERC = [0.1, 0.15, 0.2, 0.25]
DATASETS_TRAIN = ["isri-ocr", "cdip"]
DATASETS_TEST = ["S-MARQUES", "S-ISRI-OCR", "S-CDIP"]
MAP_DATASET_TRAIN_TEST = {
    "cdip": ["S-MARQUES", "S-ISRI-OCR"],
    "isri-ocr": ["S-CDIP"],
}
MAP_DATASET_TEST_TRAIN = dict()
for dataset_train, datasets_test in MAP_DATASET_TRAIN_TEST.items():
    for dataset_test in datasets_test:
        MAP_DATASET_TEST_TRAIN[dataset_test] = dataset_train
NUM_STEPS_PRINT = 100

# number of processors for parallel computing
NUM_PROC = 8
