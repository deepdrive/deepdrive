from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import argparse
import os
import shutil
import tempfile
from subprocess import Popen, PIPE
import sys
import platform
from distutils.spawn import find_executable
from distutils.version import LooseVersion as semvar

import pkg_resources

DIR = os.path.dirname(os.path.realpath(__file__))

IS_LINUX = sys.platform == 'linux' or sys.platform == 'linux2'
IS_MAC = sys.platform == 'darwin'
IS_UNIX = IS_LINUX or IS_MAC or 'bsd' in sys.platform.lower()
IS_WINDOWS = sys.platform == 'win32'


def run_command_with_sarge(cmd, throw=True):
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

    if not is_docker():
        # Docker does not build with nvidia runtime. We can do this if
        # we want by setting the default docker runtime to nvidia, but
        # we don't want to require that people do this.
        check_tensorflow_gpu()

    # Install sarge to nicely stream commands and wheel for precompiled packages
    run_command_no_deps(py + ' -m pip install sarge wheel', verbose=True)

    if 'ubuntu' in platform.platform().lower() and not is_docker():
        # Install tk for dashboard
        run_command_with_sarge('sudo apt-get install -y python3-tk', throw=False)

    run_command_with_sarge(py + ' -m pip install -r requirements.txt')

    # Create deepdrive directory
    import config as c

    if is_docker():
        pip_args = '--no-cache-dir'
    else:
        pip_args = ''

    # Install correct version of the python bindings
    # # TODO: Remove dev0 once 3.0 is stable
    run_command_with_sarge(py + ' -m pip install {pip_args} "deepdrive > {major_minor_version}.*dev0"'.format(
        major_minor_version=c.MAJOR_MINOR_VERSION_STR, pip_args=pip_args))

    # noinspection PyUnresolvedReferences
    import config.check_bindings

    print("""
   ___                  __    _            
  / _ \___ ___ ___  ___/ /___(_)  _____    
 / // / -_) -_) _ \/ _  / __/ / |/ / -_)   
/____/\__/\__/ .__/\_,_/_/ /_/|___/\__/    
  _______ __/_/___/ /_ __                  
 / __/ -_) _ `/ _  / // /                  
/_/  \__/\_,_/\_,_/\_, /                   
                  /___/            
    """)
    # Gen: https://bit.ly/2SrCVFO


def check_nvidia_docker():
    if is_docker() and not has_nvidia_docker():
        print('WARNING: No nvidia-docker runtime detected', file=sys.stderr)
        return False
    else:
        return True


def check_tensorflow_gpu():
    error_msg = \
        '\n\n*** Warning: %s, Tensorflow agents will not be available. ' \
        'HINT: Install Tensorflow or use the python / virtualenv ' \
        'you have it already installed to. ' \
        'If you install, check out our Tensorflow install ' \
        'tips on the README ' \
        '\n\n'
    print('Checking for valid Tensorflow installation')
    # noinspection PyUnresolvedReferences
    if not check_nvidia_docker():
        print(error_msg % 'Using Docker but not nvidia-docker runtime', file=sys.stderr)
        ret = False
    else:
        import h5py  # importing tensorflow later causes seg faults
        try:
            import tensorflow as tf
        except ImportError:
            print(error_msg % 'Tensorflow not installed', file=sys.stderr)
            ret = False
        else:
            try:
                pkg_resources.get_distribution('tensorflow-gpu')
            except pkg_resources.DistributionNotFound:
                print('\n\n*** Warning: %s \n\n' %
                      'tensorflow-gpu not found, performance will be severely'
                      'degraded if run on CPU. '
                      'HINT: Try "pip install tensorflow-gpu"')
                # TODO: use get_available_gpu's on a given session or
                #  confirm assumption the plain tensorflow package is always cpu
                # TODO: Handle TPU's
            else:
                print('Found tensorflow-gpu - assuming you are running'
                      ' Tensorflow on GPU')

            min_version = '1.7'
            if semvar(tf.__version__) < semvar(min_version):
                warn_msg = 'Tensorflow %s is less than the minimum ' \
                           'required version (%s)' \
                           % (tf.__version__, min_version)
                print(error_msg % warn_msg, file=sys.stderr)
                ret = False
            else:
                print('Tensorflow %s detected - meets min version'
                      ' (%s)' % (tf.__version__, min_version))
                ret = True

    return ret


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


def has_nvidia_docker():
    return 'NVIDIA_VISIBLE_DEVICES' in os.environ


if __name__ == '__main__':
    if 'TEST_RUN_CMD' in os.environ:
        # run_command_no_deps('pip install sarge')
        print(has_nvidia_docker())
    else:
        main()

