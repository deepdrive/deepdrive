from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from future.builtins import (int, open, round,
                             str)
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


def run_command_async(cmd, throw=True):
    from sarge import run, Capture
    # TODO: p = run(..., stdout=Capture(buffer_size=-1), stderr=Capture(buffer_size=-1))
    # TODO: Then log p.stdout. while process not complete in realtime and to file
    p = run(cmd, async_=True)
    # Allow streaming stdout and stderr to user while command executes
    p.close()
    if p.returncode != 0:
        if throw:
            raise RuntimeError('Command failed, see above')

def run_command_no_deps(cmd, cwd=None, env=None, throw=True, verbose=False, print_errors=True):
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
        return sys.executable
    else:
        raise RuntimeError('Error: Python 3.5+ is required to run deepdrive')


def main():
    print('Checking python version...', end='')
    py = check_py_version()
    print('check!')

    tf_valid = get_tf_valid()

    # Install sarge to nicely stream commands
    run_command_no_deps(py + ' -m pip install sarge', verbose=True)


    if 'ubuntu' in platform.platform().lower() and not is_docker():
        # Install tk for dashboard
        run_command_async('sudo apt-get install -y python3-tk', throw=False)

    run_command_async(py + ' -m pip install -r requirements.txt')

    print("""
   ___                  __    _            
  / _ \___ ___ ___  ___/ /___(_)  _____    
 / // / -_) -_) _ \/ _  / __/ / |/ / -_)   
/____/\__/\__/ .__/\_,_/_/ /_/|___/\__/    
  _______ __/_/___/ /_ __                  
 / __/ -_) _ `/ _  / // /                  
/_/  \__/\_,_/\_,_/\_, /                   
                  /___/            
    """)  #  http://patorjk.com/software/taag/#p=display&f=Small%20Slant&t=Deepdrive%0AInstall%0AComplete


def get_tf_valid():
    error_msg = '\n\n*** Warning: %s, baseline imitation learning agent will not be available. ' \
                'HINT: Install Tensorflow or use the python / virtualenv you have it already installed to. If you install, check out our Tensorflow install tips on the README ' \
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
                      'HINT: Try "pip install tensorflow-gpu"')
                return False
            sess.run(check)
            print('Tensorflow is working on the GPU.')

    except ImportError:
        print(error_msg % 'Tensorflow not installed', file=sys.stderr)
        return False
    except Exception:
        print(error_msg % 'Tensorflow not working', file=sys.stderr)
        return False

    min_version = '1.7'
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


def is_docker():
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )


if __name__ == '__main__':
    if 'TEST_RUN_CMD' in os.environ:
        run_command_async('pip install sarge')
    else:
        main()
