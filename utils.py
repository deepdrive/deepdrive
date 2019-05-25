from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import re

# TODO: Bootstrap future module to enable Python 2 support of install which depends on this file to do below
# from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
#                              int, map, next, oct, open, pow, range, round,
#                              str, super, zip)

import ctypes
import platform
import shutil

import glob
import inspect
import os
import sys
import threading
import time

import numpy as np

import h5py
import requests
import config as c
import logs
from util.anonymize import anonymize_user_home
from util.download import download
from util.ensure_sim import get_sim_path, ensure_sim
from util.run_command import run_command

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
    """
    Converts object properties to a dict.
    This acts as a single level copy, i.e. it's NOT recursive.

    @:param obj - The Object to convert
    @:param exclude - A list of property names to omit from the returned object

    """
    ret = {}
    exclude = exclude or []
    for name in dir(obj):
        if not name.startswith('__') and name not in exclude:
            value = getattr(obj, name)
            if not callable(value):
                ret[name] = value
    return ret


def save_hdf5(out, filename, background=True):
    assert_disk_space(os.path.dirname(filename))
    if 'DEEPDRIVE_NO_THREAD_SAVE' in os.environ or not background:
        return save_hdf5_task(out, filename)
    else:
        thread = threading.Thread(target=save_hdf5_task, args=(out, filename))
        thread.start()
        return thread


def save_hdf5_task(out, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    log.debug('Saving to %s', filename)
    opts = dict(compression='lzf', fletcher32=True)
    with h5py.File(filename, 'w') as f:
        for i, frame in enumerate(out):
            frame_grp = f.create_group('frame_%s' % str(i).zfill(10))
            add_collision_to_hdf5(frame, frame_grp)
            add_score_to_hdf5(frame, frame_grp)
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


def add_score_to_hdf5(frame, frame_grp):
    from sim.score import Score
    score = frame['score']
    score_grp = frame_grp.create_group('score')
    defaults = obj2dict(Score)
    prop_names = defaults.keys()
    for k in prop_names:
        if 'sampler' not in k.lower():
            score_grp.attrs[k] = score.get(k, defaults[k])
    del frame['score']


def add_collision_to_hdf5(frame, frame_grp):
    from box import Box
    clsn_grp = frame_grp.create_group('last_collision')
    clsn = Box(frame['last_collision'], box_it_up=True)
    clsn_grp.attrs['collidee_velocity'] = tuple(clsn.collidee_velocity)
    collidee_location = getattr(clsn, 'collidee_location', None)
    clsn_grp.attrs['collidee_location'] = \
        collidee_location if (clsn.time_utc and collidee_location) else ''
    clsn_grp.attrs['collision_normal'] = tuple(clsn.collision_normal)
    clsn_grp.attrs['time_since_last_collision'] = clsn.time_since_last_collision
    clsn_grp.attrs['time_stamp'] = clsn.time_stamp
    clsn_grp.attrs['time_utc'] = clsn.time_utc
    del frame['last_collision']


def read_hdf5(filename, save_png_dir=None, overfit=False, save_prefix=''):
    ret = []
    with h5py.File(filename, 'r') as file:
        for i, frame_name in enumerate(file):
            out_frame = read_frame(file, frame_name, i, save_png_dir,
                                   save_prefix)
            if out_frame is None:
                log.error('Could not read frame, skipping')
            else:
                ret.append(out_frame)
                if overfit:
                    log.info('overfitting to %r, image# %d', filename, i)
                    if i == 1:
                        break
    return ret


def read_frame(file, frame_name, frame_index, save_png_dir, save_prefix=''):
    try:
        frame = file[frame_name]
        out_frame = dict(frame.attrs)
        out_cameras = []
        for dataset_name in frame:
            if dataset_name.startswith('camera_'):
                read_camera(dataset_name, frame, frame_index,
                            out_cameras, save_png_dir, save_prefix)
            elif dataset_name == 'last_collision':
                out_frame['last_collision'] = dict(frame[dataset_name].attrs)
        out_frame['cameras'] = out_cameras
    except Exception as e:
        traceback.print_stack()
        log.error('Exception reading frame %s', str(e))
        out_frame = None
    return out_frame


def read_camera(dataset_name, frame, frame_index, out_cameras, save_png_dir,
                save_prefix=''):
    camera = frame[dataset_name]
    out_camera = dict(camera.attrs)
    out_camera['image'] = camera['image'][()]
    out_camera['depth'] = camera['depth'][()]
    out_cameras.append(out_camera)
    if save_png_dir is not None:
        if not os.path.exists(save_png_dir):
            os.makedirs(save_png_dir)
        save_camera(out_camera['image'], out_camera['depth'],
                    save_dir=save_png_dir, name=save_prefix + str(frame_index)
                    .zfill(c.HDF5_FRAME_ZFILL))


def save_camera(image, depth, save_dir, name):
    from scipy.misc import imsave
    im_path = os.path.join(save_dir, 'i_' + name + '.png')
    dp_path = os.path.join(save_dir, 'z_' + name + '.png')
    imsave(im_path, image)
    imsave(dp_path, depth)
    log.debug('saved image and depth to %s and %s', im_path, dp_path)


def show_camera(image, depth):
    from scipy.misc import toimage
    toimage(image).show()
    toimage(depth).show()
    input('Enter any key to continue')


def hdf5_to_mp4(fps=c.DEFAULT_FPS, png_dir=None, combine_all=False, sess_dir=None):
    if png_dir is None:
        png_dir = save_hdf5_recordings_to_png(combine_all, sess_dir)
    try:
        file_path = pngs_to_mp4(combine_all, fps, png_dir)
    finally:
        shutil.rmtree(png_dir)
    return file_path


def pngs_to_mp4(combine_all, fps, png_dir):
    # TODO: Add FPS, frame number, run id, date str,
    #  g-forces, episode #, hdf5 #, etc... to this
    #  and rendered views for human interprettability
    log.info('Saved png\'s to ' + png_dir)
    file_path = None
    import distutils.spawn
    ffmpeg_path = distutils.spawn.find_executable('ffmpeg')
    if ffmpeg_path is None:
        log.error('Could not find ffmpeg. Skipping hdf5=>mp4 conversion')
    else:
        zfill_total = c.HDF5_DIR_ZFILL + c.HDF5_FRAME_ZFILL
        pix_fmt = 'yuv420p'  # The pix_fmt does not define resolution (i.e. this is totally different than 480p)
        title = 'deepdrive'
        file_dir = c.RESULTS_DIR
        if not combine_all:
            title += '_' + c.DATE_STR
        file_path = os.path.join(file_dir, '%s.mp4' % title)
        ffmpeg_cmd = ('ffmpeg'
                      ' -y '
                      ' -r {fps}'
                      ' -f image2'
                      ' -i {temp_png_dir}/i_hdf5_%0{zfill_total}d.png'
                      ' -vcodec libx264'
                      ' -crf 25'
                      ' -pix_fmt {pix_fmt}'
                      ' -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"'
                      ' {file_path}'.format(fps=fps, pix_fmt=pix_fmt,
                                            zfill_total=zfill_total,
                                            file_path=file_path,
                                            temp_png_dir=png_dir))
        log.info('PNG=>MP4: ' + ffmpeg_cmd)
        ffmpeg_result = os.system(ffmpeg_cmd)
        if ffmpeg_result == 0:
            log.info('Wrote mp4 to: ' + anonymize_user_home(file_path))
        else:
            file_path = None
    return file_path


def upload_to_gist(name: str, file_paths: list, public: bool):
    files = ' '.join('"%s"' % f for f in file_paths)
    gist_env = os.environ.copy()
    gist_env['YOU_GET_MY_JIST'] = requests.get(c.YOU_GET_MY_JIST_URL).text.strip()
    if os.path.dirname(sys.executable) not in os.environ['PATH']:
        gist_env['PATH'] = os.path.dirname(sys.executable) + ':' + gist_env['PATH']
    opts = '--public' if public else ''
    cmd = 'gist {opts} create {gist_name} {files}'
    cmd = cmd.format(gist_name=name, files=files, opts=opts)
    output, ret_code = run_command(cmd, env=gist_env, verbose=True)
    if ret_code != 0:
        log.warn('Could not upload gist. \n%s' % (output,))
    url = output if ret_code == 0 else None
    return url


def in_home(name):
    p = os.path
    return p.exists(p.join(p.expanduser('~'), name))


def upload_to_youtube(file_path):
    youtube_creds_name = 'youtube-upload-credentials.json'
    client_secrets_name = 'client_secrets.json'
    youtube_creds_exists = in_home(youtube_creds_name)
    client_secrets_exists = in_home(client_secrets_name)
    if not youtube_creds_exists or not client_secrets_exists:
        log.error('Need %s and %s in your home directory to upload to YouTube.',
                  youtube_creds_name, client_secrets_name)
        return False

    # python_path = os.environ['PYTHONPATH']
    # youtube_upload_dir = os.path.join(c.ROOT_DIR, 'vendor', 'youtube_upload')
    # os.environ['PYTHONPATH'] = '%s:%s' % (youtube_upload_dir, python_path)
    import youtube_upload.main
    from box import Box
    options = Box(title=file_path, privacy='unlisted', client_secrets='',
                  credentials_file='', auth_browser=None,
                  description='Deepdrive results for %s' % c.MAIN_ARGS)
    youtube = youtube_upload.main.get_youtube_handler(options)
    video_id = youtube_upload.main.upload_youtube_video(youtube, options, file_path, 1, 0)
    # TODO: Put link to s3 artifacts in description [hdf5, csv, diff,
    #  eventually ue-recording]
    # cmd = '%s %s --title=test --privacy=unlisted %s' % (
    #     sys.executable,
    #     os.path.join(youtube_upload_dir, 'bin', 'youtube_upload'),
    #     file_path
    # )
    # os.environ['PYTHONPATH'] = python_path


    # TODO: Mount client_secret.json and credentials into a container somehow
    # PYTHONPATH=. python vendor/youtube_upload/bin/youtube_upload --title=test --privacy=unlisted test.mp4
    # TODO: Remove temp_dir if TEMP

    return video_id


def save_hdf5_recordings_to_png(combine_all=False, sess_dir=None):
    if combine_all:
        hdf5_filenames = sorted(glob.glob(c.RECORDING_DIR + '/**/*.hdf5',
                                          recursive=True))
    else:
        sess_dir = sess_dir or c.HDF5_SESSION_DIR
        hdf5_filenames = sorted(glob.glob(sess_dir + '/*.hdf5', recursive=True))
    save_dir = os.path.join(c.RECORDING_DIR, 'pngs', c.DATE_STR)
    os.makedirs(save_dir)
    for i, f in enumerate(hdf5_filenames):
        try:
            read_hdf5(f,
                      save_png_dir=save_dir,
                      save_prefix='hdf5_%s' % str(i).zfill(c.HDF5_DIR_ZFILL))
        except OSError as e:
            log.error(e)
    return save_dir


def save_random_hdf5_to_png(recording_dir=c.RECORDING_DIR):
    random_file = np.random.choice(glob.glob(recording_dir + '/*/*.hdf5'))
    if not random_file:
        raise RuntimeError('No hdf5 files found')
    else:
        p = os.path
        save_png_dir = os.path.join(p.join(recording_dir, 'random_hdf5_view'),
                                    p.basename(p.dirname(random_file)),
                                    p.basename(random_file)[:-5])
        log.info('Saving random files to ' + save_png_dir)
        os.makedirs(save_png_dir, exist_ok=True)
        read_hdf5(p.join(recording_dir, random_file),
                  save_png_dir=save_png_dir)


def read_hdf5_manual(recording_dir=c.RECORDING_DIR):
    save_png_dir = os.path.join(recording_dir, 'test_view')
    os.makedirs(save_png_dir, exist_ok=True)
    read_hdf5(os.path.join(recording_dir, '2018-01-18__05-14-48PM',
                           '0000000001.hdf5'), save_png_dir=save_png_dir)


def is_debugging():
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False


def download_weights(url):
    folder = url.split('/')[-1].replace('.zip', '')
    dest = os.path.join(c.WEIGHTS_DIR, folder)
    if not glob.glob(dest + '/*'):
        log.info('Downloading weights %s', folder)
        download(url, dest)
    else:
        log.info('Found cached weights at %s', dest)
    return dest


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
            raise Exception('Less than %dMB left on device, aborting'
                            ' save of %s' % (mb, filename))
    except Exception as e:
        log.error('Could not get free space on the drive containing %s' %
                  filename)
        raise e


def resize_images(input_image_shape, images, always=False):
    import scipy.misc
    for img_idx, img in enumerate(images):
        img = images[img_idx]
        if img.shape != input_image_shape or always:
            # Interesting bug here. Since resize converts mean subtracted
            # floats (~-120 to ~130) to 0-255 uint8,
            # but we don't always resize since randomize_cameras does nothing
            # to the size 5% of the time.
            # This actually worked surprisingly well. Need to test whether
            # this bug actually improves things or not.
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
                return False
            i += 1
        return True

    except Exception as e:
        log.error('Error closing process', str(e))
        return False


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
    # print(save_recordings_to_png_and_mp4(png_dir='/tmp/tmp30zl8ouq'))
    # print(save_hdf5_recordings_to_png())
    # print(upload_to_gist('asdf', ['/home/c2/src/deepdrive/results/2018-05-30__02-40-01PM.csv', '/home/c2/src/deepdrive/results/2019-03-14__06-08-38PM.diff']))
    # log.info('testing %s', os.path.expanduser('~'))
    import traceback

    traceback.print_stack(file=sys.stdout)
    log.info('testing %d', 1234)
    ensure_sim()
