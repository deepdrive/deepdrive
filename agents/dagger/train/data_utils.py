from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from future.builtins import (int, open, round,
                             str)

import glob
import multiprocessing
import threading
from collections import deque
from multiprocessing import Pool


import numpy as np

from utils import read_hdf5
import config as c
import logs

log = logs.get_log(__name__)


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, should_shuffle=False):
        threading.Thread.__init__(self)
        self.queue = deque()
        self.generator = generator
        self.daemon = True
        self.should_shuffle = should_shuffle
        self.cv = threading.Condition()
        self.start()

    def run(self):
        for item in self.generator:
            self._insert_item(item)
        log.debug('inserting none')
        self.queue.append(None)

    def _insert_item(self, item):
        with self.cv:
            log.debug('queue length %r', len(self.queue))
            while len(self.queue) > c.NUM_TRAIN_FILES_TO_QUEUE:
                log.debug('waiting for queue size to decrease')
                self.cv.wait()
            if self.should_shuffle:
                queue_index = c.rng.randint(0, len(self.queue) + 1)
                log.debug('inserting randomly at', queue_index)
                self.queue.insert(queue_index, item)
            else:
                self.queue.append(item)
            log.debug('inserted, queue length is %r item was None %r',
                      len(self.queue), item is None)
            self.cv.notify()  # Tell consumer we have more

    def __iter__(self):
        return self

    def __next__(self):
        log.debug('getting item')
        with self.cv:
            log.debug('in cv getting item')
            while len(self.queue) == 0:
                log.debug('waiting for item')
                self.cv.wait()
                log.debug('out of cv getting item')
            next_item = self.queue.popleft()
            self.cv.notify()   # Tell producer we want more
        if next_item is None:
            log.debug('next item is None, ending!!!!')
            raise StopIteration
        # print('returning item!', next_item)

        return next_item


def get_file_names(hdf5_path, train=True, overfit=False):
    files = glob.glob(hdf5_path + '/**/*.hdf5', recursive=True) \
            or glob.glob(hdf5_path + '*.hdf5', recursive=True)
    c.rng.shuffle(files)
    if overfit:
        files = files[-1:]
    elif train and len(files) > 1:
        files = files[1:]
    elif not train:
        # Eval
        if len(files) == 1:
            egregious_error_message()
        files = files[0:1]
    if len(files) == 0:
        raise Exception('zero %s hdf5 files, aborting!' %
                        'train' if train else 'eval')
    return files


def egregious_error_message():
    log.error("""
 .----------------. 
| .--------------. |
| |              | |
| |      _       | |
| |     | |      | |
| |     | |      | |
| |     | |      | |
| |     |_|      | |
| |     (_)      | |
| '--------------' |
 '----------------'             
            
Only one file in the
      dataset.
                  
Will eval on train!!!
""")


def load_file(h5_filename, overfit=False, mute_spurious_targets=False):
    log.info('loading %s', h5_filename)
    out_images = []
    out_targets = []
    try:
        frames = read_hdf5(h5_filename, overfit=overfit)
        c.rng.shuffle(frames)
        for frame in frames:
            # Just use one camera for now
            out_images.append(frame['cameras'][0]['image'])
            out_targets.append([*normalize_frame(frame, mute_spurious_targets)])
    except Exception as e:
        log.error('Could not load %s - skipping - error was: %r',
                  h5_filename, e)

    log.info('finished loading %s', h5_filename)
    return out_images, out_targets


def normalize_frame(frame, mute_spurious_targets=False):
    spin = frame['angular_velocity'][2]
    if spin <= -c.SPIN_THRESHOLD:
        direction = -1.0
    elif spin >= c.SPIN_THRESHOLD:
        direction = 1.0
    else:
        direction = 0.0
    spin = spin / c.SPIN_NORMALIZATION_FACTOR
    speed = frame['speed'] / c.SPEED_NORMALIZATION_FACTOR
    if mute_spurious_targets:
        speed_change = 0.
        direction = 0.
        spin = 0.
    else:
        speed_change = np.dot(
            frame['acceleration'],
            frame['forward_vector']) / c.SPEED_NORMALIZATION_FACTOR
    steering = frame['steering']
    throttle = frame['throttle']
    return spin, direction, speed, speed_change, steering, throttle


def file_loader(file_stream, overfit=False, mute_spurious_targets=False):
    for h5_filename in file_stream:
        log.info('loading %s', h5_filename)
        yield load_file(h5_filename, overfit, mute_spurious_targets)
    log.info('finished training files')


def batch_gen(file_stream, batch_size, overfit=False,
              mute_spurious_targets=False):
    gen = BackgroundGenerator(
        file_loader(file_stream, overfit, mute_spurious_targets),
        should_shuffle=False)
    for images, targets in gen:
        if overfit:
            images, targets = images[0:batch_size], targets[0:batch_size]
            num_repeats, remainder = divmod(batch_size, len(images))
            if num_repeats > 1:
                images = images * num_repeats
                targets = targets * num_repeats
                if remainder > 0:
                    images = images + images[:remainder]
                    targets = targets + targets[:remainder]
            yield images, targets
        else:
            num_iters = len(images) // batch_size
        # print('num iters', num_iters)
        # print('images', images)

            for i in range(num_iters):
                yield images[i * batch_size:(i+1) * batch_size], \
                      targets[i * batch_size:(i+1) * batch_size]

    print('finished batch gen')


class Dataset(object):
    def __init__(self, files, overfit=False, mute_spurious_targets=False):
        self._files = files
        self.overfit = overfit
        self.mute_spurious_targets = mute_spurious_targets

    def iterate_once(self, batch_size):
        def file_stream():
            for file_name in self._files:
                log.info('queueing data from %s for iterate once', file_name)
                yield file_name
        yield from batch_gen(file_stream(), batch_size,
                             mute_spurious_targets=self.mute_spurious_targets)

    def iterate_forever(self, batch_size):
        def file_stream():
            while True:
                # File order will be the same every epoch
                c.rng.shuffle(self._files)
                for file_name in self._files:
                    log.info('queueing data from %s for iterate forever',
                             file_name)
                    yield file_name
        yield from batch_gen(file_stream(), batch_size, self.overfit,
                             self.mute_spurious_targets)

    def iterate_parallel_once(self, get_callback):
        with Pool(max(multiprocessing.cpu_count() // 2, 1)) as p:
            for i, file in enumerate(self._files):
                # get_callback(i)(load_file(file, self.overfit,
                # self.mute_spurious_targets))
                p.apply_async(load_file,
                              (file, self.overfit, self.mute_spurious_targets),
                              callback=get_callback(i))
            p.close()
            p.join()


def get_dataset(hdf5_path, train=True, overfit=False,
                mute_spurious_targets=False):
    file_names = get_file_names(hdf5_path, train=train, overfit=overfit)
    return Dataset(file_names, overfit, mute_spurious_targets)


def run():
    hdf5_path = c.RECORDING_DIR
    log.info(get_file_names(hdf5_path, train=True))
    dataset = get_dataset(hdf5_path, train=True)
    log.info(dataset)
    log.info('Iterating through recordings')
    for images, targets in dataset.iterate_once(64):
        for image in images:
            if image.shape != (227,227,3):
                log.info('FOUND %r', image.shape)


if __name__ == "__main__":
    run()
