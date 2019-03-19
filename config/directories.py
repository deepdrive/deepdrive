from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

# TODO: Bootstrap future module to enable Python 2 support of install which depends on this file to do below
# from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
#                              int, map, next, oct, open, pow, range, round,
#                              str, super, zip)

import os
import sys

from datetime import datetime


def _get_deepdrive_dir():
    dir_config_file = os.path.join(DEEPDRIVE_CONFIG_DIR, 'deepdrive_dir')
    if os.path.exists(dir_config_file):
        with open(dir_config_file) as dcf:
            ret = dcf.read()
    else:
        default_dir = os.path.join(os.path.expanduser('~'), 'Deepdrive')
        ret = input('Where would you like to store Deepdrive files '
                    '(i.e. sim binaries (1GB), checkpoints (200MB), recordings, and logs)? [Press Enter for %s] '
                    % default_dir)
        deepdrive_dir_set = False
        while not deepdrive_dir_set:
            ret = ret or default_dir
            if 'deepdrive' not in ret.lower():
                ret = os.path.join(ret, 'Deepdrive')
            if not os.path.isabs(ret):
                ret = input('Path: %s is not absolute, please specify a different path [Press Enter for %s] ' %
                            (ret, default_dir))
            if os.path.isfile(ret):
                ret = input('Path: %s is already a file, please specify a different path [Press Enter for %s] ' %
                            (ret, default_dir))
            else:
                deepdrive_dir_set = True
        with open(dir_config_file, 'w') as dcf:
            dcf.write(ret)
            print('%s written to %s' % (ret, dir_config_file))
    ret = ret.replace('\r', '').replace('\n', '')
    os.makedirs(ret, exist_ok=True)
    return ret


def _ensure_python_bin_config():
    py_bin = os.path.join(DEEPDRIVE_CONFIG_DIR, 'python_bin')
    with open(py_bin, 'w') as _dpbf:
        _dpbf.write(sys.executable)


# Directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DEEPDRIVE_DIR = os.environ.get('DEEPDRIVE_DIR')
DEEPDRIVE_CONFIG_DIR = os.path.expanduser('~') + '/.deepdrive'
os.makedirs(DEEPDRIVE_CONFIG_DIR, exist_ok=True)
if DEEPDRIVE_DIR is None:
    DEEPDRIVE_DIR = _get_deepdrive_dir()
_ensure_python_bin_config()

# Data and log directories
DIR_DATE_FORMAT = '%Y-%m-%d__%I-%M-%S%p'
DATE_STR = datetime.now().strftime(DIR_DATE_FORMAT)
RECORDING_DIR = os.environ.get('DEEPDRIVE_RECORDING_DIR') or os.path.join(DEEPDRIVE_DIR, 'recordings')
HDF5_SESSION_DIR = os.path.join(RECORDING_DIR, DATE_STR)
GYM_DIR = os.path.join(DEEPDRIVE_DIR, 'gym')
LOG_DIR = os.path.join(DEEPDRIVE_DIR, 'log')
RESULTS_DIR = os.path.join(DEEPDRIVE_DIR, 'results')
TENSORFLOW_OUT_DIR = os.path.join(DEEPDRIVE_DIR, 'tensorflow')
WEIGHTS_DIR = os.path.join(DEEPDRIVE_DIR, 'weights')
BASELINES_DIR = os.path.join(DEEPDRIVE_DIR, 'baselines_results')
TFRECORD_DIR_SUFFIX = '_tfrecords'

