from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import glob
import os

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import sys
from multiprocessing import Pool
import tensorflow as tf

from agents.dagger.train.data_utils import get_dataset
from agents.dagger.train.train import resize_images
from utils import read_hdf5
import logs
import config as c


log = logs.get_log(__name__)

INPUT_IMAGE_SHAPE = (224, 224, 3)


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floats_feature(value):
    """Wrapper for inserting numpy 32 bit float arrays features into Example proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def add_total_to_tfrecord_files(directory, filename_prefix):
    all_files = glob.glob(os.path.join(directory, filename_prefix) + '_' + '[0-9]' * 5 + '*.tfrecord')
    files_to_rename = []
    for file in all_files:
        if os.path.getsize(file) == 0:
            os.remove(file)
        else:
            files_to_rename.append(file)
    mid_fix = '-of-%s' % str(len(files_to_rename)).zfill(5)
    for i, file in enumerate(sorted(files_to_rename)):
        fname = os.path.basename(file)
        os.rename(file, os.path.join(directory, fname[:16] + str(i).zfill(5) + mid_fix + '.tfrecord'))


def save_dataset(dataset, buffer_size, filename, parallelize=True):
    if parallelize:
        def get_callback(_file_idx):
            log.info('getting callback for %d', _file_idx)
            
            def callback(result):
                log.info('inside callback for %r', _file_idx)
                imgs, tgts = result
                save_tfrecord_file(_file_idx, filename, imgs, tgts)
            return callback

        dataset.iterate_parallel_once(get_callback)
    else:
        file_idx = 0
        for images, targets in dataset.iterate_once(buffer_size):
            log.info('starting file %d', file_idx)
            save_tfrecord_file(file_idx, filename, images, targets)
            file_idx += 1

    add_total_to_tfrecord_files(c.RECORDING_DIR, filename)


def save_tfrecord_file(file_idx, filename, images, targets):
    colorspace = b'RGB'
    channels = 3
    image_format = b'RAW'
    writer = tf.python_io.TFRecordWriter(filename + '_' + str(file_idx).zfill(5) + '.tfrecord')
    valid_target_shape = True
    resize_images(INPUT_IMAGE_SHAPE, images)
    for tgt in targets:
        if len(tgt) != c.NUM_TARGETS:
            log.error('invalid target shape %r skipping' % len(tgt))
            valid_target_shape = False
    if valid_target_shape:
        pass
    for image_idx in range(len(images)):

        # print how many images are saved every 1000 images
        if not image_idx % 1000:
            log.info('Train data: {}/{}'.format(image_idx, len(images)))

        image = images[image_idx]
        # TODO: Add mean pixel back to image, check min is zero, convert to uint8
        image += c.MEAN_PIXEL
        assert(min(image) == 0)

        target = targets[image_idx]

        feature_dict = {
            'image/width': _int64_feature(image.shape[0]),
            'image/height': _int64_feature(image.shape[1]),
            'image/colorspace': _bytes_feature(colorspace),
            'image/channels': _int64_feature(channels),
            'image/format': _bytes_feature(image_format),
            'image/encoded': _bytes_feature(tf.compat.as_bytes(image.tostring()))}

        for name_idx, name in enumerate(c.CONTROL_NAMES):
            feature_dict['image/control/' + name] = _float_feature(target[name_idx])

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()


# TODO: See whether HDF5 images have negative, mean subtracted values. If so, it must be the tf record process that gets rid of those
def encode(parallelize=True):
    hdf5_path = c.RECORDING_DIR
    train_dataset = get_dataset(hdf5_path, train=True)
    eval_dataset = get_dataset(hdf5_path, train=False)
    buffer_size = 1000
    save_dataset(train_dataset, buffer_size, filename=os.path.join(c.RECORDING_DIR, 'deepdrive_train'),
                 parallelize=parallelize)
    save_dataset(eval_dataset, buffer_size, filename=os.path.join(c.RECORDING_DIR, 'deepdrive_eval'),
                 parallelize=parallelize)


def decode():
    data_path = os.path.join(c.RECORDING_DIR, 'deepdrive_train_00000-of-00162.tfrecord')

    with tf.Session() as sess:
        feature = {
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], tf.string),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string)}

        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['image/encoded'], tf.uint8)

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(1000):
            example = sess.run(image)
            print(example)
        coord.request_stop()
        coord.join(threads)
        # image_np = sess.run(image)

        # # Cast label data into int32
        # label = tf.cast(features['train/label'], tf.int32)
        # # Reshape image data into the original shape
        # image = tf.reshape(image, [224, 224, 3])
        #
        # # Any preprocessing here ...
        #
        # # Creates batches by randomly shuffling tensors
        # images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
        #                                         min_after_dequeue=10)


if __name__ == '__main__':
    if 'decode' in sys.argv:
        decode()
    elif 'rename-only' in sys.argv:
        add_total_to_tfrecord_files(c.RECORDING_DIR, 'deepdrive_train')
        add_total_to_tfrecord_files(c.RECORDING_DIR, 'deepdrive_eval')
    else:
        encode(parallelize=('sync' not in sys.argv))
