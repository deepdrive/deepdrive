from __future__ import print_function
import argparse
import os
import tempfile
from subprocess import Popen, PIPE
import sys
import platform
from distutils.spawn import find_executable
from distutils.version import LooseVersion as semvar

DIR = os.path.dirname(os.path.realpath(__file__))

IS_LINUX = sys.platform == 'linux' or sys.platform == 'linux2'
IS_MAC = sys.platform == 'darwin'
IS_UNIX = IS_LINUX or IS_MAC or 'bsd' in sys.platform.lower()
IS_WINDOWS = sys.platform == 'win32'


def run_command(cmd, cwd=None, env=None, throw=True, verbose=False, print_errors=True):
    def say(*args):
        if verbose:
            print(*args)

    say('running command: ' + cmd)
    if not isinstance(cmd, list):
        cmd = cmd.split()
    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd, env=env)
    result, err = process.communicate()
    if not isinstance(result, str):
        result = ''.join(map(chr, result))
    result = result.strip()
    say(result)
    if process.returncode != 0:
        if not isinstance(err, str):
            err = ''.join(map(chr, err))
        err_msg = ' '.join(cmd) + ' finished with error ' + err.strip()
        if throw:
            raise RuntimeError(err_msg)
        elif print_errors:
            say(err_msg)
    return result, process.returncode


def check_py_version():
    version = sys.version_info[:]

    if version[0] == 3 and version[1] >= 5:
        return find_executable('python')
    else:
        raise RuntimeError('Error: Python 3.5+ is required to run deepdrive-agents')


def main():
    print('Checking python version')
    py = check_py_version()
    tf_valid = get_tf_valid()

    if 'ubuntu' in platform.platform().lower():
        # Install tk for dashboard
        run_command('sudo apt-get install -y python3-tk', throw=False, verbose=True)

    run_command(py + ' -m pip install -r requirements.txt', verbose=True)

    if tf_valid:
        print('Starting baseline agent')
        os.system('python main.py --baseline')
    else:
        print('Starting sim in path follower mode')
        os.system('python main.py --let-game-drive')


def get_tf_valid():
    error_msg = '\n\n*** Warning: %s, baseline imitation learning agent will not be available. ' \
                'HINT: Check out our CUDA / cuDNN install tips on the README ' \
                '\n\n'

    print('Checking for valid Tensorflow installation')
    try:
        # noinspection PyUnresolvedReferences
        import tensorflow as tf
        check = tf.constant('string tensors are not tensors but are called tensors in tensorflow')
        with tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                              gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.01,
                                                                        allow_growth=True))) as sess:
            if not get_available_gpus():
                print('\n\n*** Warning: %s \n\n' %
                      'Tensorflow could not find a GPU, performance will be severely degraded on CPU. '
                      'HINT: Try "pipenv install tensorflow-gpu"')
                return False
            sess.run(check)
            print('Tensorflow is working on the GPU.')

    except ImportError:
        print(error_msg % 'Tensorflow not installed', file=sys.stderr)
        return False
    except Exception:
        print(error_msg % 'Tensorflow not working', file=sys.stderr)
        return False

    min_version = '1.1'
    if semvar(tf.__version__) < semvar(min_version):
        warn_msg = 'Tensorflow %s is less than the minimum required version (%s)' % (tf.__version__, min_version)
        print(error_msg % warn_msg, file=sys.stderr)
        return False
    else:
        print('Tensorflow %s detected - meets min version (%s)' % (tf.__version__, min_version))
        return True


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if __name__ == '__main__':
    main()
