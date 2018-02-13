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
            print(err_msg)
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
    _, tf_valid = get_tf_valid(py)

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


def get_tf_valid(py, verbose=True):
    flags = ''
    if not verbose:
        flags += ' --version-only'
    cmd_out, exit_code = run_command('%s -u bin/install/check_tf_version.py%s' % (py, flags),
                                     throw=False, verbose=verbose, print_errors=True)
    if exit_code == 0 and not verbose:
        tf_version = cmd_out.splitlines()[-1]
    else:
        tf_version = None
    tf_valid = exit_code == 0
    return tf_version, tf_valid


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
