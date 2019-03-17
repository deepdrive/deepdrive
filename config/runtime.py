from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

# TODO: Bootstrap future module to enable Python 2 support of install which depends on this file to do below
# from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
#                              int, map, next, oct, open, pow, range, round,
#                              str, super, zip)

import random
import os
import sys
from glob import glob

from config.directories import *

import numpy as np

try:
    from gym.utils import seeding
    # Seeded random number generator for reproducibility
    RNG_SEED = 0
    rng = seeding.np_random(RNG_SEED)[0]
except Exception as e:
    import __main__
    if getattr(__main__, '__file__', None) != 'install.py':
        raise e
    else:
        print('Skipping rng seed - not needed for install')


import config.version

# General
CONTROL_NAMES = ['spin', 'direction', 'speed', 'speed_change', 'steering', 'throttle']

# Net
NUM_TARGETS = len(CONTROL_NAMES)

# Normalization
SPIN_THRESHOLD = 1.0
SPEED_NORMALIZATION_FACTOR = 2000.
SPIN_NORMALIZATION_FACTOR = 10.
MEAN_PIXEL = np.array([104., 117., 123.], np.float32)

# HDF5
FRAMES_PER_HDF5_FILE = int(os.environ.get('FRAMES_PER_HDF5_FILE', 1000))
MAX_RECORDED_OBSERVATIONS = FRAMES_PER_HDF5_FILE * 150
NUM_TRAIN_FRAMES_TO_QUEUE = 6000
NUM_TRAIN_FILES_TO_QUEUE = NUM_TRAIN_FRAMES_TO_QUEUE // FRAMES_PER_HDF5_FILE

# OS 
IS_LINUX = sys.platform == 'linux' or sys.platform == 'linux2'
IS_MAC = sys.platform == 'darwin'
IS_UNIX = IS_LINUX or IS_MAC or 'bsd' in sys.platform.lower()
IS_WINDOWS = sys.platform == 'win32'
if IS_WINDOWS:
    OS_NAME = 'windows'
elif IS_LINUX:
    OS_NAME = 'linux'
else:
    raise RuntimeError('Unexpected OS')

# AGENTS
DAGGER = 'dagger'
DAGGER_MNET2 = 'dagger_mobilenet_v2'
BOOTSTRAPPED_PPO2 = 'bootstrapped_ppo2'


# Weights
ALEXNET_BASELINE_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'baseline_agent_weights')
ALEXNET_BASELINE_WEIGHTS_VERSION = 'model.ckpt-143361'
ALEXNET_PRETRAINED_NAME = 'bvlc_alexnet.ckpt'
ALEXNET_PRETRAINED_PATH = os.path.join(WEIGHTS_DIR, ALEXNET_PRETRAINED_NAME)

MNET2_BASELINE_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'mnet2_baseline_weights')
MNET2_BASELINE_WEIGHTS_VERSION = 'model.ckpt-49147'
MNET2_PRETRAINED_NAME = 'mobilenet_v2_1.0_224_checkpoint'
MNET2_PRETRAINED_PATH = os.path.join(WEIGHTS_DIR, MNET2_PRETRAINED_NAME, 'mobilenet_v2_1.0_224.ckpt')

PPO_BASELINE_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'ppo_baseline_agent_weights')
PPO_BASELINE_WEIGHTS_VERSION = '03125'

# Urls
BUCKET_URL = 'https://s3-us-west-1.amazonaws.com/deepdrive'
BASE_WEIGHTS_URL = BUCKET_URL + '/weights'
ALEXNET_BASELINE_WEIGHTS_URL = BASE_WEIGHTS_URL + '/baseline_agent_weights.zip'
ALEXNET_PRETRAINED_URL = '%s/%s.zip' % (BASE_WEIGHTS_URL, ALEXNET_PRETRAINED_NAME)
MNET2_PRETRAINED_URL = '%s/%s.zip' % (BASE_WEIGHTS_URL, MNET2_PRETRAINED_NAME)
MNET2_BASELINE_WEIGHTS_URL = BASE_WEIGHTS_URL + '/mnet2_baseline_weights.zip'
PPO_BASELINE_WEIGHTS_URL = BASE_WEIGHTS_URL + '/ppo_baseline_agent_weights.zip'
SIM_PREFIX = 'deepdrive-sim-' + OS_NAME


# Sim
if 'DEEPDRIVE_SIM_START_COMMAND' in os.environ:
    # Can do something like `<your-unreal-path>\Engine\Binaries\Win32\UE4Editor.exe <your-deepdrive-sim-path>\DeepDrive.uproject -game ResX=640 ResY=480`
    SIM_START_COMMAND = os.environ['DEEPDRIVE_SIM_START_COMMAND']
else:
    SIM_START_COMMAND = None


def get_sim_path():
    orig_path = os.path.join(DEEPDRIVE_DIR, 'sim')
    version_paths = glob(os.path.join(DEEPDRIVE_DIR, 'deepdrive-sim-*-%s.*' % config.version.MAJOR_MINOR_VERSION_STR))
    version_paths = [vp for vp in version_paths if not vp.endswith('.zip')]
    if version_paths:
        return list(sorted(version_paths))[-1]
    else:
        return orig_path

REUSE_OPEN_SIM = 'DEEPDRIVE_REUSE_OPEN_SIM' in os.environ
SIM_PATH = get_sim_path()

DEFAULT_CAM = dict(name='forward cam 227x227 60 FOV', field_of_view=60, capture_width=227, capture_height=227,
                   relative_position=[150, 1.0, 200],
                   relative_rotation=[0.0, 0.0, 0.0])

DEFAULT_FPS = 8

try:
    import tensorflow
except ImportError:
    TENSORFLOW_AVAILABLE = False
else:
    TENSORFLOW_AVAILABLE = True


# Not passing through main.py args yet, but better for reproducing to put here than in os.environ
SIMPLE_PPO = False
# PPO_RESUME_PATH = '/home/a/baselines_results/openai-2018-06-17-17-48-24-795338/checkpoints/03125'
# PPO_RESUME_PATH = '/home/a/baselines_results/openai-2018-06-22-00-00-21-866205/checkpoints/03125'
PPO_RESUME_PATH = None
# TEST_PPO = False


# API
API_PORT = 5557
API_TIMEOUT_MS = 5000
IS_EVAL = False

# Stream
STREAM_PORT = 5558

# Set via main
PY_ARGS = None
