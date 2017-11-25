import inspect
import platform
import sys
import threading
import time
from logging.handlers import RotatingFileHandler

import deepdrive
import h5py
import logging

from config import *


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
    depth = depth_heatmap(depth)
    return depth


def depth_heatmap(depth):
    red = depth
    green = 1.0 - np.abs(0.5 - depth) * 2.
    blue = 1. - depth
    ret = np.array([red, green, blue])
    ret = np.transpose(ret, (1, 2, 0))
    ret = (ret * 255).astype('uint8', copy=False)
    return ret


class connection:
    def __enter__(self):
        connected = False
        size = 157286400
        # TODO: Establish some handshake so we don't hardcode size here and in Unreal project
        if platform.system() == 'Linux':
            connected = deepdrive.reset('/tmp/deepdrive_shared_memory', size)
        elif platform.system() == 'Windows':
            connected = deepdrive.reset('Local\DeepDriveCapture_1', size)
        ret = connected == 1
        if ret:
            log.info('Connected to deepdrive')
        else:
            log.error('Could not connect to deepdrive')
            raise Exception('Could not connect to deepdrive')
        return ret

    def __exit__(self, type, value, traceback):
        log.debug('closing connection to deepdrive')
        deepdrive.close()


def obj2dict(obj, exclude=None):
    ret = {}
    exclude = exclude or []
    for name in dir(obj):
        if not name.startswith('__') and not name in exclude:
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
    log.info('saving to %s', filename)
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
    log.info('done saving %s', filename)


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
                    save_camera(out_camera['image'], out_camera['depth'], dir=save_png_dir, name=str(i).zfill(10))
            out_frame['cameras'] = out_cameras
            ret.append(out_frame)
    return ret


def save_camera(image, depth, dir, name):
    from scipy.misc import imsave
    imsave(os.path.join(dir, 'i_' + name + '.png'), image)
    imsave(os.path.join(dir, 'z_' + name + '.png'), depth)


def show_camera(image, depth):
    from scipy.misc import toimage
    toimage(image).show()
    toimage(depth).show()
    input('Enter any key to continue')


log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_rotator = RotatingFileHandler(os.path.join(LOG_DIR, 'log.txt'), maxBytes=(1048576 * 5), backupCount=7)
log_rotator.setFormatter(log_format)


def get_log(namespace, level=logging.INFO, rotator=log_rotator):
    ret = logging.getLogger(namespace)
    ret.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_format)
    ret.addHandler(ch)
    os.makedirs(LOG_DIR, exist_ok=True)
    ret.addHandler(rotator)
    return ret


def read_hdf5_manual():
    save_png_dir = os.path.join(RECORDINGS_DIR, 'test_view')
    os.makedirs(save_png_dir)
    read_hdf5(os.path.join(RECORDINGS_DIR, '2017-11-22_0105_26AM', '0000000001.hdf5'), save_png_dir=save_png_dir)


def log_manual():
    test_log_rotator = RotatingFileHandler(os.path.join(LOG_DIR, 'test.txt'), maxBytes=3, backupCount=7)
    log1 = get_log('log1', rotator=test_log_rotator)
    log2 = get_log('log2', rotator=test_log_rotator)
    log1.info('asdf')
    log2.info('zxcv')


def is_debugging():
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False

log = get_log(__name__)

if __name__ == '__main__':
    log_manual()
