from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import os

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import sys
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


def save_dataset(dataset, buffer_size, filename):
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(filename)
    colorspace = b'RGB'
    channels = 3
    image_format = b'RAW'
    for images, targets in dataset.iterate_once(buffer_size):

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
                log.info('Train data: {}/{}'.format(image_i-dx, len(images)))

            image = images[image_idx]
            target = targets[image_idx]

            feature_dict = {
                'image/width': _int64_feature(image.shape[0]),
                'image/height': _int64_feature(image.shape[1]),
                'image/colorspace': _bytes_feature(colorspace),
                'image/channels': _int64_feature(channels),
                'image/format': _bytes_feature(image_format),
                'image/encoded':  _bytes_feature(tf.compat.as_bytes(image.tostring()))}

            for name_idx, name in enumerate(c.CONTROL_NAMES):
                feature_dict['image/control/' + name] = _float_feature(target[name_idx])

            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

    writer.close()


def do_it():
    hdf5_path = c.RECORDING_DIR
    train_dataset = get_dataset(hdf5_path, train=True)
    eval_dataset = get_dataset(hdf5_path, train=False)
    buffer_size = 1000
    save_dataset(train_dataset, buffer_size, filename=os.path.join(c.RECORDING_DIR, 'deepdrive_train.tfrecord'))
    save_dataset(eval_dataset, buffer_size, filename=os.path.join(c.RECORDING_DIR, 'deepdrive_eval.tfrecord'))


if __name__ == '__main__':
    do_it()
