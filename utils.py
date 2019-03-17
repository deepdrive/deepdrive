from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import re

# TODO: Bootstrap future module to enable Python 2 support of install which depends on this file to do below
# from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
#                              int, map, next, oct, open, pow, range, round,
#                              str, super, zip)


import ctypes
import platform
import traceback

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
    assert_disk_space(os.path.dirname(filename))
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
    from box import Box
    clsn_grp = frame_grp.create_group('last_collision')
    clsn = Box(frame['last_collision'], box_it_up=True)
    clsn_grp.attrs['collidee_velocity'] = tuple(clsn.collidee_velocity)
    collidee_location = getattr(clsn, 'collidee_location', None)
    clsn_grp.attrs['collidee_location'] = collidee_location if (clsn.time_utc and collidee_location) else ''
    clsn_grp.attrs['collision_normal'] = tuple(clsn.collision_normal)
    clsn_grp.attrs['time_since_last_collision'] = clsn.time_since_last_collision
    clsn_grp.attrs['time_stamp'] = clsn.time_stamp
    clsn_grp.attrs['time_utc'] = clsn.time_utc
    del frame['last_collision']


def read_hdf5(filename, save_png_dir=None, overfit=False):
    ret = []
    with h5py.File(filename, 'r') as file:
        for i, frame_name in enumerate(file):
            out_frame = read_frame(file, frame_name, i, save_png_dir)
            if out_frame is None:
                log.error('Could not read frame, skipping')
            else:
                ret.append(out_frame)
                if overfit:
                    log.info('overfitting to %r, image# %d', filename, i)
                    if i == 1:
                        break
    return ret


def read_frame(file, frame_name, frame_index, save_png_dir):
    try:
        frame = file[frame_name]
        out_frame = dict(frame.attrs)
        out_cameras = []
        for dataset_name in frame:
            if dataset_name.startswith('camera_'):
                read_camera(dataset_name, frame, frame_index,
                            out_cameras, save_png_dir)
            elif dataset_name == 'last_collision':
                out_frame['last_collision'] = dict(frame[dataset_name].attrs)
        out_frame['cameras'] = out_cameras
    except Exception as e:
        traceback.print_stack()
        log.error('Exception reading frame %s', str(e))
        out_frame = None
    return out_frame


def read_camera(dataset_name, frame, frame_index, out_cameras, save_png_dir):
    camera = frame[dataset_name]
    out_camera = dict(camera.attrs)
    out_camera['image'] = camera['image'].value
    out_camera['depth'] = camera['depth'].value
    out_cameras.append(out_camera)
    if save_png_dir is not None:
        save_camera(out_camera['image'], out_camera['depth'],
                    save_dir=save_png_dir, name=str(frame_index).zfill(10))


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


def upload_hdf5_to_youtube():
    temppngdir = save_hdf5_recordings_to_png()
    log.info(temppngdir)
    # ffmpeg -r 8 -f image2 -s 224x224 -i /tmp/i_%010d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" test.mp4
    # TODO: Mount client_secret.json and credentials into a container somehow
    # PYTHONPATH=. python vendor/youtube-upload/bin/youtube-upload --title=test --privacy=unlisted test.mp4


def save_hdf5_recordings_to_png():
    hdf5_filenames = sorted(glob.glob(c.RECORDING_DIR + '/**/*.hdf5', recursive=True))
    tempdir = tempfile.gettempdir()
    for f in hdf5_filenames:
        try:
            read_hdf5(f, save_png_dir=os.path.join(tempdir, f))
        except OSError as e:
            log.error(e)
    return tempdir


def save_random_hdf5_to_png(recording_dir=c.RECORDING_DIR):
    random_file = np.random.choice(glob.glob(recording_dir + '/*/*.hdf5'))
    if not random_file:
        raise RuntimeError('No hdf5 files found')
    else:
        save_png_dir = os.path.join(os.path.join(recording_dir, 'random_hdf5_view'),
                                    os.path.basename(os.path.dirname(random_file)),
                                    os.path.basename(random_file)[:-5])
        log.info('Saving random files to ' + save_png_dir)
        os.makedirs(save_png_dir, exist_ok=True)
        read_hdf5(os.path.join(recording_dir, random_file), save_png_dir=save_png_dir)


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
    # TODO: Remove overwrite as a parameter, doesn't make sense here.
    if os.path.exists(path) and (dir_has_stuff(path) or file_has_stuff(path)):
        if warn_existing:
            print('%s exists, do you want to re-download and overwrite any existing files (y/n)?' % path, end=' ')
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


def get_sim_bin_path(return_expected_path=False):
    expected_path = None

    def get_from_glob(search_path):
        paths = glob.glob(search_path) or [search_path]
        paths = [p for p in paths if not (p.endswith('.debug') or p.endswith('.sym'))]
        if len(paths) > 1:
            log.warn('Found multiple sim binaries in search directory - picking the first from %r', paths)
        if not paths:
            ret_path = None
        else:
            ret_path = paths[0]
        return ret_path

    if c.REUSE_OPEN_SIM:
        return None
    elif c.IS_LINUX:
        if os.path.exists(c.SIM_PATH + '/LinuxNoEditor'):
            expected_path = c.SIM_PATH + '/LinuxNoEditor/DeepDrive/Binaries/Linux/DeepDrive*'
        else:
            expected_path = c.SIM_PATH + '/DeepDrive/Binaries/Linux/DeepDrive'
    elif c.IS_MAC:
        raise NotImplementedError('Sim does not yet run on OSX, see FAQs / running a remote agent in /api.')
    elif c.IS_WINDOWS:
        expected_path = os.path.join(c.SIM_PATH, 'WindowsNoEditor', 'DeepDrive', 'Binaries', 'Win64', 'DeepDrive*.exe')

    path = get_from_glob(expected_path)
    if path and not os.path.exists(path):
        ret = None
    else:
        ret = path

    if return_expected_path:
        return ret, expected_path
    else:
        return ret


def get_sim_project_dir():
    if c.REUSE_OPEN_SIM:
        path = input('What is the path to your simulator project directory?'
                     '\n\ti.e. for sources something like ~/src/deepdrive-sim '
                     '\n\tor for packaged binaries, something like ~/Deepdrive/sim/LinuxNoEditor/DeepDrive')
    elif c.IS_LINUX:
        path = os.path.join(c.SIM_PATH, 'LinuxNoEditor/DeepDrive')
    elif c.IS_MAC:
        raise NotImplementedError('Sim does not yet run on OSX, see FAQs / running a remote agent in /api.')
    elif c.IS_WINDOWS:
        path = os.path.join(c.SIM_PATH, 'WindowsNoEditor', 'DeepDrive')
    else:
        raise RuntimeError('OS not recognized')

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


def get_sim_url():
    sim_prefix = 'sim/' + c.SIM_PREFIX
    conn = S3Connection(anon=True)
    bucket = conn.get_bucket('deepdrive')
    bucket_search_str = sim_prefix + '-' + c.MAJOR_MINOR_VERSION_STR
    sim_versions = list(bucket.list(bucket_search_str))
    if not sim_versions:
        raise RuntimeError('Could not find a sim version matching %s in bucket %s' % (bucket_search_str, c.BUCKET_URL))
    latest_sim_file, path_version = sorted([(x.name, x.name.split('.')[-2]) for x in sim_versions],
                                           key=lambda y: y[1])[-1]
    return '/' + latest_sim_file


def ensure_sim():
    actual_path, expected_path = get_sim_bin_path(return_expected_path=True)
    if actual_path is None:
        print('\n--------- Simulator not found in %s, downloading ----------' % expected_path)
        if c.IS_LINUX or c.IS_WINDOWS:
            if os.environ.get('SIM_URL', 'latest') == 'latest':
                log.info('Downloading latest sim')
                url = c.BUCKET_URL + get_sim_url()
            else:
                url = os.environ['SIM_URL']
            c.SIM_PATH = os.path.join(c.DEEPDRIVE_DIR, url.split('/')[-1][:-4])
            download(url, c.SIM_PATH, warn_existing=False, overwrite=False)
        else:
            raise NotImplementedError('Sim download not yet implemented for this OS')
    ensure_executable(get_sim_bin_path())
    ensure_sim_python_binaries()


def ensure_sim_python_binaries():
    base_url = c.BUCKET_URL + '/embedded_python_for_unreal/'
    if c.IS_WINDOWS:
        # These include Python and our requirements
        lib_url = base_url + 'windows/python_bin_with_libs.zip'
        lib_path = os.path.join(get_sim_project_dir(), 'Binaries', 'Win64')
        if not os.path.exists(lib_path) or not os.path.exists(os.path.join(lib_path, 'python3.dll')):
            print('Unreal embedded Python not found. Downloading...')
            download(lib_url, lib_path, overwrite=True, warn_existing=False)
    elif c.IS_LINUX:
        # Python is already embedded, however in docker ensure_requirements fails with pip-req-tracker errors
        uepy = os.path.join(c.SIM_PATH, 'Engine/Plugins/UnrealEnginePython/EmbeddedPython/Linux/bin/python3')
        st = os.stat(uepy)
        os.chmod(uepy, st.st_mode | stat.S_IEXEC)
        os.system('{uepy} -m pip install pyzmq pyarrow requests'.format(uepy=uepy))
        log.info('Installed UEPy python depdendencies')
    elif c.IS_MAC:
        raise NotImplementedError('Sim does not yet run on OSX, see FAQs / running a remote agent in /api.')


def is_docker():
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )


def get_free_space_mb(filename):
    """Return folder/drive free space (in megabytes)."""
    if platform.system() == 'Windows':
        drive, _path = os.path.splitdrive(filename)
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(drive), None, None, ctypes.pointer(free_bytes))
        return free_bytes.value / 1024 / 1024
    else:
        path = filename
        while not os.path.exists(path):
            if not path or path == '/':
                raise ValueError('Drive does not exist for filename %s' % filename)
            path = os.path.dirname(path)
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / 1024 / 1024


def remotable(f):
    def extract_args(*args, **kwargs):
        return f((args, kwargs), *args, **kwargs)

    return extract_args


def assert_disk_space(filename, mb=2000):
    try:
        if get_free_space_mb(filename) < mb:
            raise Exception('Less than %dMB left on device, aborting save of %s' % (mb, filename))
    except Exception as e:
        log.error('Could not get free space on the drive containing %s' % filename)
        raise e


def resize_images(input_image_shape, images, always=False):
    import scipy.misc
    for img_idx, img in enumerate(images):
        img = images[img_idx]
        if img.shape != input_image_shape or always:
            # Interesting bug here. Since resize converts mean subtracted floats (~-120 to ~130) to 0-255 uint8,
            # but we don't always resize since randomize_cameras does nothing to the size 5% of the time.
            # This actually worked surprisingly well. Need to test whether this bug actually improves things or not.
            log.debug('invalid image shape %s - resizing', str(img.shape))
            images[img_idx] = scipy.misc.imresize(img, (input_image_shape[0],
                                                        input_image_shape[1]))
    return images


def kill_process(process_to_kill):
    try:
        process_to_kill.terminate()
        time.sleep(0.2)
        i = 0
        while process_to_kill and process_to_kill.poll() is None:
            log.info('Waiting for process to die')
            time.sleep(0.1 * 2 ** i)
            if i > 4:
                # Die!
                log.warn('Forcefully killing process')
                process_to_kill.kill()
                break
            i += 1
    except Exception as e:
        log.error('Error closing process', str(e))


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


if __name__ == '__main__':
    # download('https://d1y4edi1yk5yok.cloudfront.net/sim/asdf.zip', r'C:\Users\a\src\beta\deepdrive-agents-beta\asdf')
    # read_hdf5_manual()
    # ensure_sim()
    # save_random_hdf5_to_png()
    # assert_disk_space(r'C:\Users\a\DeepDrive\recordings\2018-11-03__12-29-33PM\0000000143.hdf5')
    # assert_disk_space('/media/a/data-ext4/deepdrive-data/v2.1/asdf.hd5f')
    # print(get_sim_url())
    print(upload_hdf5_to_youtube())
