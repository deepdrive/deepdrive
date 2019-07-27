from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import glob
import os
import sys

import tensorflow as tf

import utils
from agents.dagger.train.data_utils import get_dataset
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
    """
    Wrapper for inserting numpy 32 bit float arrays features
    into Example proto.
    """
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=value.reshape(-1)))


def add_total_to_tfrecord_files(directory, filename_prefix):
    all_files = glob.glob(os.path.join(directory, filename_prefix) +
                          '_' + '[0-9]' * 5 + '*.tfrecord')
    files_to_rename = []
    for file in all_files:
        if os.path.getsize(file) == 0:
            log.warn('Encountered an empty tfrecord file! Deleting %s', file)
            os.remove(file)
        else:
            files_to_rename.append(file)
    mid_fix = '-of-%s' % str(len(files_to_rename) - 1).zfill(5)
    for i, file in enumerate(sorted(files_to_rename)):
        fname = os.path.basename(file)
        os.rename(file, os.path.join(directory, fname[:16] + str(i).zfill(5)
                                     + mid_fix + '.tfrecord'))


def save_dataset(dataset, buffer_size, filename, out_path, parallelize=True):
    if parallelize:
        def get_callback(_file_idx):
            log.debug('getting callback for %d', _file_idx)
            
            def callback(result):
                log.debug('inside callback for %r', _file_idx)
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

    add_total_to_tfrecord_files(out_path, filename)


def save_tfrecord_file(file_idx, filename, images, targets):
    utils.assert_disk_space(filename)
    colorspace = b'RGB'
    channels = 3
    image_format = b'RAW'
    out_filename = filename + '_' + str(file_idx).zfill(5) + '.tfrecord'
    writer = tf.python_io.TFRecordWriter(out_filename)
    utils.resize_images(INPUT_IMAGE_SHAPE, images)
    for image_idx in range(len(images)):

        if not image_idx % 500:
            log.info('{}/{} examples saved to {}'.format(
                image_idx, len(images), out_filename))

        image = images[image_idx]

        # Undo initial subtraction of mean pixel during recording
        image += c.MEAN_PIXEL.astype(image.dtype)

        if image.min() < 0:
            raise ValueError('Min image less than zero')

        target = targets[image_idx]

        feature_dict = {
            'image/width': _int64_feature(image.shape[0]),
            'image/height': _int64_feature(image.shape[1]),
            'image/colorspace': _bytes_feature(colorspace),
            'image/channels': _int64_feature(channels),
            'image/format': _bytes_feature(image_format),
            'image/encoded': _bytes_feature(
                tf.compat.as_bytes(image.tostring()))}

        for name_idx, name in enumerate(c.CONTROL_NAMES):
            feature_dict['image/control/' + name] = \
                _float_feature(target[name_idx])

        example = tf.train.Example(
            features=tf.train.Features(feature=feature_dict))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()
    log.info('Wrote %r examples to %s', len(images), out_filename)


def encode(parallelize=True, hdf5_path=c.RECORDING_DIR, experiment=None):
    # TODO(post v3): Get a couple separate hdf5 files from different
    #  situations / view modes for eval
    hdf5_to_convert_path = get_hdf5_path_to_convert(hdf5_path)
    train_dataset = get_dataset(hdf5_to_convert_path, train=True)
    eval_dataset = get_dataset(hdf5_to_convert_path, train=False)
    buffer_size = 1000
    utils.assert_disk_space(hdf5_path)
    out_path = None
    while experiment is None:
        experiment = utils.get_valid_filename(
            input('Enter a name for your dataset: '))
        out_path = os.path.join(hdf5_path, experiment + c.TFRECORD_DIR_SUFFIX)
        if os.path.exists(out_path):
            print('The path %s exists, please choose a new name.' % out_path)
            experiment = None
    if out_path is None:
        raise RuntimeError('tfrecord output path not set')
    os.makedirs(out_path)
    save_dataset(train_dataset, buffer_size,
                 filename=os.path.join(out_path, 'deepdrive_train'),
                 parallelize=parallelize, out_path=out_path)
    save_dataset(eval_dataset, buffer_size,
                 filename=os.path.join(out_path, 'deepdrive_eval'),
                 parallelize=parallelize, out_path=out_path)


def get_hdf5_path_to_convert(hdf5_path):
    hdf5_dirs = list(get_hdf5_dirs(hdf5_path))
    if not hdf5_dirs:
        raise RuntimeError('No directories with hdf5 files found in %s' %
                           hdf5_path)
    convert_all = 1
    convert_latest = 2
    convert_path = 3
    options = {
        convert_all: 'Convert all HDF5 recording to tfrecords',
        convert_latest: 'Convert latest (%s)' % hdf5_dirs[0],
        convert_path: 'Enter a path to convert', }
    option = None
    if 'DEEPDRIVE_CONVERT_ALL_HDF5' in os.environ:
        option = convert_all
    else:
        done = False
        print('Please choose an option:')
        msg = '\n'.join(['%d) %s' % o for o in options.items()])
        answer = input(msg + '\nOption #: ')
        while not done:
            try:
                option = int(answer)
                _ = options[option]
            except (ValueError, KeyError):
                answer = input('Invalid option. Please try again: ')
            else:
                done = True
    if option is None:
        raise RuntimeError('No option chosen for HDF5 conversion')
    if option == convert_path:
        hdf5_path = input(
            'Please input the absolute path to the directory '
            'containing hdf5 files you want to convert: ').strip()
    elif option == convert_latest:
        hdf5_path = os.path.join(hdf5_path, hdf5_dirs[0])
    return hdf5_path


def get_hdf5_dirs(hdf5_path):
    for d in sorted(next(os.walk(hdf5_path))[1], reverse=True):
        has_hdf5 = glob.glob(os.path.join(hdf5_path, d, '*.hdf5'))
        sensible_year = d[0:4].isdigit() and int(d[0:4]) > 2000
        if has_hdf5 and sensible_year:
            yield d


def test_decode():
    data_path = os.path.join(c.RECORDING_DIR,
                             'deepdrive_train_00000-of-00162.tfrecord')

    with tf.Session() as sess:
        feature = {
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], tf.string),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string)}

        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([data_path],
                                                        num_epochs=1)
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
        # images, labels = tf.train.shuffle_batch([image, label],
        #     batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)


if __name__ == '__main__':
    log.info('Converting HDF5 to tf record')
    if '--decode' in sys.argv:
        test_decode()
    elif '--rename-only' in sys.argv:
        add_total_to_tfrecord_files(c.RECORDING_DIR, 'deepdrive_train')
        add_total_to_tfrecord_files(c.RECORDING_DIR, 'deepdrive_eval')
    else:
        encode(parallelize=('sync' not in sys.argv))
