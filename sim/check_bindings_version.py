import os

import pkg_resources
from distutils.version import LooseVersion as semvar

import config as c


DIR = os.path.dirname(os.path.realpath(__file__))


def check_bindings_version():
    bindings_version = semvar(pkg_resources.get_distribution('deepdrive').version).version[:2]
    client_version = c.MAJOR_MINOR_VERSION
    if bindings_version != client_version:
        print("""ERROR: Python bindings version mismatch. 

Expected {client_version_str}, got {bindings_version_str}

HINT:

For binary sim distributions, try:
pip install package=={client_version_str}.*

For source sim distributions, try:
cd <your-sim-sources>/Plugins/DeepDrivePlugin/Source/DeepDrivePython
python build/build.py --type dev

""".format(client_version=client_version, bindings_version=bindings_version,
           client_version_str='.'.join(str(vx) for vx in client_version),
           bindings_version_str='.'.join(str(vx) for vx in bindings_version), ))
        exit(1)

if __name__ == '__main__':
    print(c.MAJOR_MINOR_VERSION)

