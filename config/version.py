from distutils.version import LooseVersion as semvar

from config.directories import *

# Version
VERSION_STR = open(os.path.join(ROOT_DIR, 'VERSION')).read()
MAJOR_MINOR_VERSION = semvar(VERSION_STR).version[:2]
MAJOR_MINOR_VERSION_STR = '.'.join(str(vx) for vx in MAJOR_MINOR_VERSION)
