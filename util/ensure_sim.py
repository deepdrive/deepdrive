from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import glob
import os
import stat

from boto.s3.connection import S3Connection
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import deepdrive_api.utils
from deepdrive_api.utils import check_pyarrow_compatibility
import config as c
import logs
from util.run_command import run_command


from util.download import download

log = logs.get_log(__name__)


def ensure_sim(update=False):
    actual_path, expected_path = get_sim_bin_path(return_expected_path=True)
    if update or actual_path is None:
        print('\n--------- Updating to latest simulator ----------')
        if c.IS_LINUX or c.IS_WINDOWS:
            if c.SIM_URL is None:
                log.info('Downloading latest sim')
                url = c.AWS_BUCKET_URL + get_latest_sim_url()
            else:
                log.info(f'Using configured SIM_URL {c.SIM_URL}')
                url = c.SIM_URL
            sim_path = os.path.join(c.DEEPDRIVE_DIR, get_sim_name_from_url(url))
            download(url, sim_path, warn_existing=False, overwrite=False)
        else:
            raise NotImplementedError(
                'Sim download not yet implemented for this OS')
    ensure_executable(get_sim_bin_path())
    ensure_sim_python_binaries()


def get_sim_name_from_url(url, include_file_extension=False):
    ret = url.split('/')[-1]
    if not include_file_extension:
        ret = ''.join(ret.split('.')[:-1])
    return ret


def ensure_sim_python_binaries():
    base_url = c.AWS_BUCKET_URL + '/embedded_python_for_unreal/'
    if c.IS_WINDOWS:
        # These include Python and our requirements
        lib_url = base_url + 'windows/python_bin_with_libs.zip'
        lib_path = os.path.join(get_sim_project_dir(), 'Binaries', 'Win64')
        if not os.path.exists(lib_path) or not os.path.exists(
                os.path.join(lib_path, 'python3.dll')):
            print('Unreal embedded Python not found. Downloading...')
            download(lib_url, lib_path, overwrite=True, warn_existing=False)
    elif c.IS_LINUX:
        # Python is already embedded, however ensure_requirements
        # fails with pip-req-tracker errors
        uepy = deepdrive_api.utils.ensure_uepy_executable(get_sim_path())
        os.system('{uepy} -m pip install pyzmq pyarrow==0.12.1 requests'.
                  format(uepy=uepy))
        log.info('Installed UEPy python dependencies')
    elif c.IS_MAC:
        raise NotImplementedError(
            'Sim does not yet run on OSX, see FAQs /'
            ' running a remote agent in /api.')


def ensure_executable(path):
    if c.IS_UNIX:
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)


def get_sim_bin_path(return_expected_path=False):
    expected_path = None

    def get_from_glob(search_path):
        paths = glob.glob(search_path) or [search_path]
        paths = [p for p in paths
                 if not (p.endswith('.debug') or p.endswith('.sym'))]
        if len(paths) > 1:
            log.warn('Found multiple sim binaries in search directory - '
                     'picking the first from %r', paths)
        if not paths:
            ret_path = None
        else:
            ret_path = paths[0]
        return ret_path

    sim_path = get_sim_path()
    if c.REUSE_OPEN_SIM:
        return None
    elif c.IS_LINUX:
        if os.path.exists(sim_path + '/LinuxNoEditor'):
            expected_path = sim_path + '/LinuxNoEditor/DeepDrive/Binaries/Linux/DeepDrive*'
        else:
            expected_path = sim_path + '/DeepDrive/Binaries/Linux/DeepDrive'
    elif c.IS_MAC:
        raise NotImplementedError('Sim does not yet run on OSX, see FAQs / '
                                  'running a remote agent in /api.')
    elif c.IS_WINDOWS:
        expected_path = os.path.join(
            sim_path, 'WindowsNoEditor', 'DeepDrive', 'Binaries', 'Win64',
            'DeepDrive*.exe')

    path = get_from_glob(expected_path)
    if path and not os.path.exists(path):
        ret = None
    else:
        ret = path

    if return_expected_path:
        return ret, expected_path
    else:
        return ret


def get_latest_sim_url():
    sim_prefix = 'sim/' + c.SIM_PREFIX
    conn = S3Connection(anon=True)
    bucket = conn.get_bucket('deepdrive')
    bucket_search_str = sim_prefix + '-' + c.MAJOR_MINOR_VERSION_STR
    sim_versions = list(bucket.list(bucket_search_str))
    if not sim_versions:
        raise RuntimeError('Could not find a sim version matching %s '
                           'in bucket %s' % (bucket_search_str, c.AWS_BUCKET_URL))
    latest_sim_file, path_version = \
        sorted([(x.name, x.name.split('.')[-2]) for x in sim_versions],
               key=lambda y: y[1])[-1]
    return '/' + latest_sim_file


def get_sim_path() -> str:
    if c.SIM_URL:
        sim_dir = get_sim_name_from_url(c.SIM_URL)
        ret = os.path.join(c.DEEPDRIVE_DIR, sim_dir)
    else:
        paths = glob.glob(
            os.path.join(c.DEEPDRIVE_DIR, 'deepdrive-sim-*-%s.*'
                                          % c.version.MAJOR_MINOR_VERSION_STR))
        paths = [p for p in paths if not p.endswith('.zip')]
        ret = list(sorted(paths))[-1]
    return ret

def get_sim_project_dir():
    if c.REUSE_OPEN_SIM:
        path = input('What is the path to your simulator project directory?'
                     '\n\ti.e. for sources something like ~/src/deepdrive-sim '
                     '\n\tor for packaged binaries, something like ~/Deepdrive/sim/LinuxNoEditor/DeepDrive')
    elif c.IS_LINUX:
        path = os.path.join(get_sim_path(), 'LinuxNoEditor/DeepDrive')
    elif c.IS_MAC:
        raise NotImplementedError('Sim does not yet run on OSX, see FAQs / running a remote agent in /api.')
    elif c.IS_WINDOWS:
        path = os.path.join(get_sim_path(), 'WindowsNoEditor', 'DeepDrive')
    else:
        raise RuntimeError('OS not recognized')

    return path


def check_pyarrow_compat():
    try:
        import ue4cli
        manager = ue4cli.UnrealManagerFactory.create()
        try:
            unreal_root = manager.getEngineRoot()
        except ue4cli.UnrealManagerException.UnrealManagerException:
            log.warning("""
Unable to find Unreal Engine root directory. Please run

ue4 setroot <your-unreal-directory>

""")
            return

        check_pyarrow_compatibility(unreal_root)
    except:
        log.exception('Could not check for pyarrow compatibility, if you see '
                      'segfaults serializing UEPY data, ensure the '
                      'UEPY embedded python pyarrow version matches the '
                      'the pyarrow version being used by this python '
                      'interpreter')
