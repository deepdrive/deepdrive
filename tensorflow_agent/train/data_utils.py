import logging as log

import glob
import threading
from collections import deque

from utils import read_hdf5
from config import *


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
            with self.cv:
                print('queue length', len(self.queue))
                while len(self.queue) > NUM_TRAIN_FILES_TO_QUEUE:
                    print('waiting for queue size to decrease')
                    self.cv.wait()
                if self.should_shuffle:
                    queue_index = RNG.randint(0, len(self.queue))
                    print('inserting randomly at', queue_index)
                    self.queue.insert(queue_index, item)
                else:
                    self.queue.append(item)
                print('inserted, queue length is', len(self.queue), 'item was None', item is None)
                self.cv.notify()  # Tell consumer we have more
        print('inserting none')
        self.queue.append(None)

    def __iter__(self):
        return self

    def __next__(self):
        print('getting item')
        with self.cv:
            print('in cv getting item')
            while len(self.queue) == 0:
                print('waiting for item')
                self.cv.wait()
            print('out of cv getting item')
            next_item = self.queue.popleft()
            self.cv.notify()   # Tell producer we want more
        if next_item is None:
            print('next item is None, ending!!!!')
            raise StopIteration
        # print('returning item!', next_item)

        return next_item


def get_file_names(hdf5_path, train=True):
    files = glob.glob(hdf5_path + '/*/*.hdf5')
    if train:
        files = files[1:]
    else:
        files = files[0:1]
    if len(files) == 0:
        raise Exception('zero %s hdf5 files, aborting!' % 'train' if train else 'eval')
    return files


def load_file(h5_filename):
    try:
        frames = read_hdf5(h5_filename)
    except Exception as e:
        log.error('Could not load %s - skipping', h5_filename)
        frames = []
    RNG.shuffle(frames)
    out_images = []
    out_targets = []
    for frame in frames:
        out_images.append(frame['cameras'][0]['image'])  # Just use one camera for now
        out_targets.append([*normalize_frame(frame)])
    return out_images, out_targets


def normalize_frame(frame):
    spin = frame['angular_velocity'][2]
    if spin <= -SPIN_THRESHOLD:
        direction = -1.0
    elif spin >= SPIN_THRESHOLD:
        direction = 1.0
    else:
        direction = 0.0
    spin = spin / SPIN_NORMALIZATION_FACTOR
    speed = frame['speed'] / SPEED_NORMALIZATION_FACTOR
    speed_change = np.dot(frame['acceleration'], frame['forward_vector']) / SPEED_NORMALIZATION_FACTOR
    steering = frame['steering']
    throttle = frame['throttle']
    return spin, direction, speed, speed_change, steering, throttle


def file_loader(file_stream):
    for h5_filename in file_stream:
        log.info('loading %s', h5_filename)
        yield load_file(h5_filename)
    log.info('finished training files')


def batch_gen(file_stream, batch_size):
    gen = BackgroundGenerator(file_loader(file_stream), should_shuffle=False)
    # test_gen = list(gen)
    # print('length of gen', len(test_gen))
    # test_gen = list(gen)
    # print('length of gen2', len(test_gen))
    for images, targets in gen:
        num_iters = len(images) // batch_size
        print('num iters', num_iters)
        # print('images', images)
        for i in range(num_iters):
            yield images[i * batch_size:(i+1) * batch_size], targets[i * batch_size:(i+1) * batch_size]
    print('finished batch gen')


class Dataset(object):
    def __init__(self, files, log):
        # TODO: Avoid passing log object around, https://stackoverflow.com/questions/5974273
        self._files = files
        self.log = log

    def iterate_once(self, batch_size):
        def file_stream():
            for file_name in self._files:
                self.log.info('queueing data from %s for iterate once', file_name)
                yield file_name
        yield from batch_gen(file_stream(), batch_size)

    def iterate_forever(self, batch_size):
        def file_stream():
            while True:
                RNG.shuffle(self._files)  # File order will be the same every epoch
                for file_name in self._files:
                    self.log.info('queueing data from %s for iterate forever', file_name)
                    yield file_name

        # TODO: Make Python 2 compatible with something like
        # for x in batch_gen(file_stream(), batch_size):
        #     yield x
        yield from batch_gen(file_stream(), batch_size)


def get_dataset(hdf5_path, log, train=True):
    file_names = get_file_names(hdf5_path, train=train)
    return Dataset(file_names, log)


def run():
    hdf5_path = os.environ['DEEPDRIVE_HDF5_PATH']
    log.info(get_file_names(hdf5_path))
    dataset = get_dataset(hdf5_path, log)
    log.info(dataset)

if __name__ == "__main__":
    run()
