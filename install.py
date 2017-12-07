from __future__ import print_function

import argparse
import os
import tempfile
from subprocess import Popen, PIPE
import sys
import platform
from distutils.spawn import find_executable


DIR = os.path.dirname(os.path.realpath(__file__))

IS_LINUX = sys.platform == 'linux' or sys.platform == 'linux2'
IS_MAC = sys.platform == 'darwin'
IS_UNIX = IS_LINUX or IS_MAC or 'bsd' in sys.platform.lower()
IS_WINDOWS = sys.platform == 'win32'


def run_command(cmd, cwd=None, env=None, throw=True):
    print('running %s' % cmd)
    if not isinstance(cmd, list):
        cmd = cmd.split()
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd, env=env)
    result, err = p.communicate()
    if not isinstance(result, str):
        result = ''.join(map(chr, result)).strip()
    if p.returncode != 0 and throw:
        if not isinstance(err, str):
            err = ''.join(map(chr, err)).strip()
        print(result)
        raise RuntimeError(' '.join(cmd) + ' finished with error ' + err)
    return result, p.returncode


def main():
    py, _ = run_command('python -u bin/install/check_py_version.py')
    _, exit_code = run_command('%s -u bin/install/check_tf_version.py' % py, throw=False)
    tf_valid = exit_code == 0
    if 'ubuntu' in platform.platform().lower():
        run_command('sudo apt-get install -y python3-tk', throw=False)
    if not find_executable('pipenv'):
        install_pipenv = '%s -m pip install pipenv' % py
        if IS_LINUX:
            run_command('sudo ' + install_pipenv)
        else:
            run_command(install_pipenv)
    os.system('pipenv install')
    print('Running sanity tests')
    os.system('pipenv run pytest tests/test_sanity.py')
    print('Tests successful')
    if tf_valid:
        print('Starting baseline agent')
        os.system('bin/run_baseline_agent.sh')
    else:
        print('Starting sim in manual mode')
        os.system('bin/drive_manually.sh')


if __name__ == '__main__':
    main()


# Run the tutorial.sh
# Mute with u
# Escape to see menu / controls
# Change cam with 1, 2, 3
# Alt-tab to get back to agent.py
# Change the throttle out in agent.py

# Pause the game, ask them to press j
# Pause the game, ask them to change the camera position
# Pause the game, ask them to change the steering coeff

# To rerun this tutorial, run tutorial.sh
