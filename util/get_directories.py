from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import os
import sys
import uuid
from datetime import datetime

p = os.path


def makedirs(name, exist_ok=False):
    if not p.exists(name) or not exist_ok:
        os.makedirs(name)


def _get_deepdrive_dir():
    dir_config_file = p.join(DEEPDRIVE_CONFIG_DIR, 'deepdrive_dir')
    if p.exists(dir_config_file):
        with open(dir_config_file) as dcf:
            ret = dcf.read()
    else:
        default_dir = p.join(p.expanduser('~'), 'Deepdrive')
        if 'DEEPDRIVE_DOCKER_HOST' in os.environ:
            ret = default_dir
        else:
            ret = input('Where would you like to store Deepdrive files '
                        '(i.e. sim binaries (1GB), checkpoints (200MB), '
                        'recordings, and logs)? [Press Enter for %s] '
                        % default_dir)
            deepdrive_dir_set = False
            while not deepdrive_dir_set:
                ret = ret or default_dir
                if 'deepdrive' not in ret.lower():
                    ret = p.join(ret, 'Deepdrive')
                if not p.isabs(ret):
                    ret = input('Path: %s is not absolute, please specify a '
                                'different path [Press Enter for %s] ' %
                                (ret, default_dir))
                if p.isfile(ret):
                    ret = input('Path: %s is already a file, please specify a '
                                'different path [Press Enter for %s] ' %
                                (ret, default_dir))
                else:
                    deepdrive_dir_set = True
        with open(dir_config_file, 'w') as dcf:
            dcf.write(ret)
            print('%s written to %s' % (ret, dir_config_file))
    ret = ret.replace('\r', '').replace('\n', '')
    if not p.exists(ret):
        print('Creating Deepdrive directory at %s' % ret)
        os.makedirs(ret)
    return ret


def _ensure_python_bin_config():
    if 'DEEPDRIVE_DOCKER_HOST' in os.environ:
        return
    else:
        py_bin = p.join(DEEPDRIVE_CONFIG_DIR, 'python_bin')
        with open(py_bin, 'w') as _dpbf:
            _dpbf.write(sys.executable)


# Directories
ROOT_DIR = p.dirname(p.dirname(p.realpath(__file__)))
DEEPDRIVE_DIR = os.environ.get('DEEPDRIVE_DIR')
DEEPDRIVE_CONFIG_DIR = p.expanduser('~') + '/.deepdrive'
makedirs(DEEPDRIVE_CONFIG_DIR, exist_ok=True)
if DEEPDRIVE_DIR is None:
    DEEPDRIVE_DIR = _get_deepdrive_dir()
_ensure_python_bin_config()

# Data and log directories
RUN_ID = uuid.uuid4().hex[:4]
DIR_DATE_FORMAT = '%Y-%m-%d__%I-%M-%S%p'
DATE_STR = datetime.now().strftime(DIR_DATE_FORMAT)
RECORDING_DIR = os.environ.get('DEEPDRIVE_RECORDING_DIR') or \
                p.join(DEEPDRIVE_DIR, 'recordings')
HDF5_SESSION_DIR = p.join(RECORDING_DIR, DATE_STR)
GYM_DIR = p.join(DEEPDRIVE_DIR, 'gym')
LOG_DIR = p.join(DEEPDRIVE_DIR, 'log')
RESULTS_BASE_DIR = p.join(DEEPDRIVE_DIR, 'results')
RESULTS_DIR = p.join(RESULTS_BASE_DIR, DATE_STR + '_' + RUN_ID)
PUBLIC_ARTIFACTS_DIR = p.join(RESULTS_DIR, 'public_artifacts')
LATEST_PUBLIC_ARTIFACTS_DIR = p.join(RESULTS_BASE_DIR,
                                     'latest_public_artifacts')
TENSORFLOW_OUT_DIR = p.join(DEEPDRIVE_DIR, 'tensorflow')
WEIGHTS_DIR = p.join(DEEPDRIVE_DIR, 'weights')
BASELINES_DIR = p.join(DEEPDRIVE_DIR, 'baselines_results')
TFRECORD_DIR_SUFFIX = '_tfrecords'
TF_ENV_EVENT_DIR = p.join(TENSORFLOW_OUT_DIR, 'env', DATE_STR)
makedirs(TF_ENV_EVENT_DIR, exist_ok=True)
