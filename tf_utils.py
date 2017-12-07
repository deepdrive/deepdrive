import tensorflow as tf
import time
from tensorflow.python.client import timeline
import numpy as np
import logging as log


IMAGE = tf.placeholder(tf.float64)
DEPTH = tf.placeholder(tf.float64)


def _image_op(x):
    y = x ** 0.45  # gamma correction
    y = tf.clip_by_value(y, 0, 1)
    y = y * 255.
    y = tf.cast(y, tf.uint8)
    return y


def _depth_op(x):
    x = x ** -(1 / 3.)
    x = _normalize_op(x)
    x = _heatmap_op(x)
    return x


def _normalize_op(x):
    amax = tf.reduce_max(x)
    amin = tf.reduce_min(x)
    arange = amax - amin
    x = (x - amin) / arange
    return x


def _heatmap_op(x):
    red = x
    green = 1.0 - tf.abs(0.5 - x) * 2.
    blue = 1. - x
    y = tf.stack([red, green, blue])
    y = tf.transpose(y, (1, 2, 0))
    y = tf.cast(y * 255, tf.uint8)
    return y


image_op = _image_op(IMAGE)
depth_op = _depth_op(DEPTH)


def preprocess_image(image, sess, trace=False):
    return _run_op(sess, image_op, IMAGE, image, trace, op_name='preprocess_image')


def preprocess_depth(depth, sess, trace=False):
    return _run_op(sess, depth_op, DEPTH, depth, trace, op_name='preprocess_depth')


def _run_op(sess, op, X, x, trace=False, op_name='tf_op'):
    start = time.time()
    if trace:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        ret = sess.run(op, feed_dict={X: x}, options=run_options, run_metadata=run_metadata)

        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
    else:
        ret = sess.run(op, feed_dict={X: x})
    end = time.time()
    log.debug('%r took %rms', op_name, (end - start) * 1000)
    return ret


def _main():
    h = w = 227
    import sys
    log.basicConfig(level=log.DEBUG, stream=sys.stdout, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    with tf.Session() as sess:
        preprocess_image(np.random.rand(h, w, 3), sess)
        preprocess_image(np.random.rand(h, w, 3), sess)
        preprocess_image(np.random.rand(h, w, 3), sess)
        preprocess_depth(np.random.rand(h, w,), sess)
        preprocess_depth(np.random.rand(h, w,), sess)
        preprocess_depth(np.random.rand(h, w,), sess)


if __name__ == '__main__':
    _main()
