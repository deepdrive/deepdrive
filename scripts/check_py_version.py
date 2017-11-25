from __future__ import print_function
from distutils.spawn import find_executable
import sys

import os

version = sys.version_info[:]

if version[0] == 3 and version[1] >= 5:
    if 'try3' not in sys.argv:
        print('python')
    exit(0)
elif version[0] < 3:
        py3 = find_executable('python3')
        if py3 and 0 == os.system('%s %s try3' % (py3, os.path.abspath(__file__))):
            # recursive call with python3
            print(py3)
            exit(0)

print('Error: Python 3.5+ is required to run deepdrive-agents', file=sys.stderr)
exit(1)




