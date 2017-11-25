import random
import os

from datetime import datetime
import numpy as np

# Net
NUM_TARGETS = 6
IMAGE_SHAPE = (227, 227, 3)

# Normalization
SPIN_THRESHOLD = 1.0
SPEED_NORMALIZATION_FACTOR = 2000.
SPIN_NORMALIZATION_FACTOR = 10.
MEAN_PIXEL = np.array([104., 117., 123.], np.float32)

# HDF5
FRAMES_PER_HDF5_FILE = 1000
MAX_RECORDED_OBSERVATIONS = FRAMES_PER_HDF5_FILE * 250
NUM_TRAIN_FILES_TO_QUEUE = 2000 // FRAMES_PER_HDF5_FILE

# Data directories
DIR_DATE_FORMAT = '%Y-%m-%d__%I-%M-%S%p'
DATE_STR = datetime.now().strftime(DIR_DATE_FORMAT)
DEEPDRIVE_DIR = os.environ.get('DEEPDRIVE_DIR') or os.path.join(os.path.expanduser('~'), 'DeepDrive')
RECORDINGS_DIR = os.path.join(DEEPDRIVE_DIR, 'recordings')
GYM_DIR = os.path.join(DEEPDRIVE_DIR, 'gym')
LOG_DIR = os.path.join(DEEPDRIVE_DIR, 'log')
BENCHMARK_DIR = os.path.join(DEEPDRIVE_DIR, 'benchmark')
TENSORFLOW_OUT_DIR = os.path.join(DEEPDRIVE_DIR, 'tensorflow')

# Seeded random number generator for reproducibility
RNG = random.Random(42.77)

