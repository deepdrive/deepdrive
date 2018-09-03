import glob
import inspect
import os
import stat
import sys
import threading
import time
import zipfile
import tempfile

import h5py
import numpy as np
import pkg_resources
import requests
from clint.textui import progress
from subprocess import Popen, PIPE
from boto.s3.connection import S3Connection

import config as c
import logs

log = logs.get_log(__name__)


def normalize(a):
    amax = a.max()
    amin = a.min()
    arange = amax - amin
    a = (a - amin) / arange
    return a


def preprocess_image(image):
    start = time.time()
    image = (image.astype(np.float32, copy=False)
             ** 0.45  # gamma correct
             * 255.)
    image = np.clip(image, a_min=0, a_max=255)\
        .astype('uint8', copy=False)
    end = time.time()
    log.debug('preprocess_capture_image took %rms', (end - start) * 1000.)
    return image


def preprocess_depth(depth):
    depth = depth.astype('float64', copy=False)
    # x = list(range(depth.size))
    # y = depth.flatten()
    # plt.scatter(x, y)
    # plt.show()
    depth = depth ** -(1 / 3.)
    depth = normalize(depth)
    return depth


def depth_heatmap(depth):
    red = depth
    green = 1.0 - np.abs(0.5 - depth) * 2.
    blue = 1. - depth
    ret = np.array([red, green, blue])
    ret = np.transpose(ret, (1, 2, 0))
    ret = (ret * 255).astype('uint8', copy=False)
    return ret


def obj2dict(obj, exclude=None):
    ret = {}
    exclude = exclude or []
    for name in dir(obj):
        if not name.startswith('__') and name not in exclude:
            value = getattr(obj, name)
            if not inspect.ismethod(value):
                value = getattr(obj, name)
            ret[name] = value
    return ret


def save_hdf5(out, filename):
    if 'DEEPDRIVE_NO_THREAD_SAVE' in os.environ:
        save_hdf5_thread(out, filename)
    else:
        thread = threading.Thread(target=save_hdf5_thread, args=(out, filename))
        thread.start()


def save_hdf5_thread(out, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    log.debug('Saving to %s', filename)
    opts = dict(compression='lzf', fletcher32=True)
    with h5py.File(filename, 'w') as f:
        for i, frame in enumerate(out):
            frame_grp = f.create_group('frame_%s' % str(i).zfill(10))
            add_collision_to_hdf5(frame, frame_grp)
            add_cams_to_hdf5(frame, frame_grp, opts)
            del frame['cameras']
            for k, v in frame.items():
                frame_grp.attrs[k] = v
    log.info('Saved to %s', filename)


def add_cams_to_hdf5(frame, frame_grp, opts):
    for j, camera in enumerate(frame['cameras']):
        camera_grp = frame_grp.create_group('camera_%s' % str(j).zfill(5))
        camera_grp.create_dataset('image', data=camera['image'], **opts)
        camera_grp.create_dataset('depth', data=camera['depth'], **opts)
        del camera['image_data']
        del camera['depth_data']
        del camera['image']
        if 'image_raw' in camera:
            del camera['image_raw']
        del camera['depth']
        for k, v in camera.items():
            # TODO: Move this to a 'props' dataset as attrs can only be 64kB
            camera_grp.attrs[k] = v


def add_collision_to_hdf5(frame, frame_grp):
    clsn_grp = frame_grp.create_group('last_collision')
    clsn = frame['last_collision']
    clsn_grp.attrs['collidee_velocity'] = tuple(clsn.collidee_velocity)
    clsn_grp.attrs['collidee_location'] = clsn.collidee_location if (clsn.time_utc and clsn.collidee_location) else ''
    clsn_grp.attrs['collision_normal'] = tuple(clsn.collision_normal)
    clsn_grp.attrs['time_since_last_collision'] = clsn.time_since_last_collision
    clsn_grp.attrs['time_stamp'] = clsn.time_stamp
    clsn_grp.attrs['time_utc'] = clsn.time_utc
    del frame['last_collision']


def read_hdf5(filename, save_png_dir=None, overfit=False):
    ret = []
    with h5py.File(filename, 'r') as file:
        for i, frame_name in enumerate(file):
            frame = file[frame_name]
            out_frame = dict(frame.attrs)
            out_cameras = []
            for camera_name in frame:
                camera = frame[camera_name]
                out_camera = dict(camera.attrs)
                out_camera['image'] = camera['image'].value
                out_camera['depth'] = camera['depth'].value
                out_cameras.append(out_camera)
                if save_png_dir is not None:
                    save_camera(out_camera['image'], out_camera['depth'], save_dir=save_png_dir, name=str(i).zfill(10))
            out_frame['cameras'] = out_cameras
            ret.append(out_frame)
            if overfit:
                log.info('overfitting to %r, image# %d', filename, i)
                if i == 1:
                    break
    return ret


def save_camera(image, depth, save_dir, name):
    from scipy.misc import imsave
    im_path = os.path.join(save_dir, 'i_' + name + '.png')
    dp_path = os.path.join(save_dir, 'z_' + name + '.png')
    imsave(im_path, image)
    imsave(dp_path, depth)
    log.info('saved image and depth to %s and %s', im_path, dp_path)


def show_camera(image, depth):
    from scipy.misc import toimage
    toimage(image).show()
    toimage(depth).show()
    input('Enter any key to continue')


def read_hdf5_manual(recording_dir=c.RECORDING_DIR):
    save_png_dir = os.path.join(recording_dir, 'test_view')
    os.makedirs(save_png_dir, exist_ok=True)
    read_hdf5(os.path.join(recording_dir, '2018-01-18__05-14-48PM', '0000000001.hdf5'), save_png_dir=save_png_dir)


def is_debugging():
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False


def download(url, directory, warn_existing=True, overwrite=False):
    """Useful for downloading a folder / zip file from dropbox/s3/cloudfront and unzipping it to path"""
    if has_stuff(directory, warn_existing, overwrite):
        return
    else:
        os.makedirs(directory, exist_ok=True)

    log.info('Downloading %s to %s...', url, directory)

    request = requests.get(url, stream=True)
    filename = url.split('/')[-1]
    if '?' in filename:
        filename = filename[:filename.index('?')]
    location = os.path.join(tempfile.gettempdir(), filename)
    with open(location, 'wb') as f:
        if request.status_code == 404:
            raise RuntimeError('Download URL not accessible %s' % url)
        total_length = int(request.headers.get('content-length'))
        for chunk in progress.bar(request.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
            if chunk:
                f.write(chunk)
                f.flush()

    log.info('done.')
    zip_ref = zipfile.ZipFile(location, 'r')
    log.info('Unzipping temp file %s to %s...', location, directory)
    try:
        zip_ref.extractall(directory)
    except Exception:
        print('You may want to close all programs that may have these files open or delete existing '
              'folders this is trying to overwrite')
        raise
    finally:
        zip_ref.close()
        os.remove(location)
        log.info('Removed temp file %s', location)


def dir_has_stuff(path):
    return os.path.isdir(path) and os.listdir(path)


def file_has_stuff(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def has_stuff(path, warn_existing=False, overwrite=False):
    if os.path.exists(path) and (dir_has_stuff(path) or file_has_stuff(path)):
        if warn_existing:
            print('%s exists, do you want to re-download and overwrite the existing files (y/n)?' % path, end=' ')
            overwrite = input()
            if 'n' in overwrite.lower():
                print('USING EXISTING %s - Try rerunning and overwriting if you run into problems.' % path)
                return True
        elif not overwrite:
            return True
    return False


def ensure_executable(path):
    if c.IS_UNIX:
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)


def get_sim_bin_path():
    path = None

    def get_from_glob(search_path):
        paths = glob.glob(search_path)
        if len(paths) > 1:
            log.warn('Found multiple sim binaries in search directory - picking the first from %r', paths)
        if not paths:
            ret = None
        else:
            ret = paths[0]
        return ret

    if c.REUSE_OPEN_SIM:
        return None
    elif c.IS_LINUX:
        path = get_from_glob(c.SIM_PATH + '/LinuxNoEditor/DeepDrive/Binaries/Linux/DeepDrive*')
    elif c.IS_MAC:
        raise NotImplementedError('Support for OSX not yet implemented, see FAQs')
    elif c.IS_WINDOWS:
        path = get_from_glob(os.path.join(c.SIM_PATH, 'WindowsNoEditor', 'DeepDrive', 'Binaries') + '/Win64/*.exe')

    if path and not os.path.exists(path):
        path = None
    return path


def run_command(cmd, cwd=None, env=None, throw=True, verbose=False, print_errors=True):
    def say(*args):
        if verbose:
            print(*args)
    say(cmd)
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


def get_latest_sim_file():
    if c.IS_WINDOWS:
        os_name = 'windows'
    elif c.IS_LINUX:
        os_name = 'linux'
    else:
        raise RuntimeError('Unexpected OS')
    sim_prefix = 'sim/deepdrive-sim-'
    conn = S3Connection(anon=True)
    bucket = conn.get_bucket('deepdrive')
    deepdrive_version = pkg_resources.get_distribution('deepdrive').version
    major_minor = deepdrive_version[:deepdrive_version.rindex('.')]
    sim_versions = list(bucket.list(sim_prefix + os_name + '-' + major_minor))

    latest_sim_file, path_version = sorted([(x.name, x.name.split('.')[-2]) for x in sim_versions],
                                           key=lambda y: y[1])[-1]
    return '/' + latest_sim_file


def download_sim():
    if get_sim_bin_path() is None:
        print('\n--------- Simulator not found, downloading ----------')
        if c.IS_LINUX or c.IS_WINDOWS:
            if os.environ.get('SIM_URL', 'latest') == 'latest':
                log.info('Downloading latest sim')
                url = c.BASE_URL + get_latest_sim_file()
            else:
                url = os.environ['SIM_URL']
            download(url, c.SIM_PATH, warn_existing=False, overwrite=False)
        else:
            raise NotImplementedError('Sim download not yet implemented for this OS')
    ensure_executable(get_sim_bin_path())


def is_docker():
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )


def remotable(f):
    def extract_args(*args, **kwargs):
        return f((args, kwargs), *args, **kwargs)

    return extract_args

if __name__ == '__main__':
    # download('https://d1y4edi1yk5yok.cloudfront.net/sim/asdf.zip', r'C:\Users\a\src\beta\deepdrive-agents-beta\asdf')
    read_hdf5_manual()
