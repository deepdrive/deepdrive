import inspect
import logging
import os
import stat
import sys
import threading
import time
import zipfile
from logging.handlers import RotatingFileHandler
from urllib.request import urlretrieve

import h5py
import numpy as np
import requests

import config as c


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
            for j, camera in enumerate(frame['cameras']):
                camera_grp = frame_grp.create_group('camera_%s' % str(j).zfill(5))
                camera_grp.create_dataset('image', data=camera['image'], **opts)
                camera_grp.create_dataset('depth', data=camera['depth'], **opts)
                del camera['image_data']
                del camera['depth_data']
                del camera['image']
                del camera['depth']
                for k, v in camera.items():
                    camera_grp.attrs[k] = v
            del frame['cameras']
            for k, v in frame.items():
                frame_grp.attrs[k] = v
    log.info('Saved to %s', filename)


def read_hdf5(filename, save_png_dir=None):
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
    return ret


def save_camera(image, depth, save_dir, name):
    from scipy.misc import imsave
    imsave(os.path.join(save_dir, 'i_' + name + '.png'), image)
    imsave(os.path.join(save_dir, 'z_' + name + '.png'), depth)


def show_camera(image, depth):
    from scipy.misc import toimage
    toimage(image).show()
    toimage(depth).show()
    input('Enter any key to continue')


os.makedirs(c.LOG_DIR, exist_ok=True)
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_rotator = RotatingFileHandler(os.path.join(c.LOG_DIR, 'log.txt'), maxBytes=(1048576 * 5), backupCount=7)
log_rotator.setFormatter(log_format)


def get_log(namespace, level=logging.INFO, rotator=log_rotator):
    ret = logging.getLogger(namespace)
    ret.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_format)
    ret.addHandler(ch)
    ret.addHandler(rotator)
    return ret


def read_hdf5_manual():
    save_png_dir = os.path.join(c.RECORDINGS_DIR, 'test_view')
    os.makedirs(save_png_dir)
    read_hdf5(os.path.join(c.RECORDINGS_DIR, '2017-11-22_0105_26AM', '0000000001.hdf5'), save_png_dir=save_png_dir)


def log_manual():
    test_log_rotator = RotatingFileHandler(os.path.join(c.LOG_DIR, 'test.txt'), maxBytes=3, backupCount=7)
    log1 = get_log('log1', rotator=test_log_rotator)
    log2 = get_log('log2', rotator=test_log_rotator)
    log1.info('asdf')
    log2.info('zxcv')


def is_debugging():
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False


def download(url, directory, warn_existing=True, overwrite=False):
    """Useful for downloading a folder / zip file from dropbox/s3/cloudfront and unzipping it to path"""
    if has_stuff(directory, warn_existing, overwrite):
        return

    log.info('Downloading %s to %s...', url, directory)
    location = urlretrieve(url)
    log.info('done.')
    location = location[0]
    zip_ref = zipfile.ZipFile(location, 'r')
    log.info('Unzipping temp file %s to %s...', location, directory)
    try:
        zip_ref.extractall(directory)
        print('done.')
    except Exception:
        print('You may want to close all programs that may have these files open or delete existing '
              'folders this is trying to overwrite')
        raise
    finally:
        zip_ref.close()
        os.remove(location)


def download_file(url, path, warn_existing=True, overwrite=False):
    """Good for downloading large files from dropbox as it sets gzip headers and decodes automatically on download"""
    if has_stuff(path, warn_existing, overwrite):
        return

    with open(path, "wb") as f:
        log.info('Downloading %s', url)
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()


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


log = get_log(__name__)

if __name__ == '__main__':
    log_manual()
