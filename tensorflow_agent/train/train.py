from __future__ import print_function

import logging
import tensorflow as tf
from tensorflow_agent.train.data_utils import get_dataset
from utils import get_log

from config import *
from tensorflow_agent.net import Net

log = get_log(__name__, logging.DEBUG)


def visualize_model(model, y):
    names = ["spin", "direction", "speed", "speed_change", "steering", "throttle"]
    for i in range(6):
        p = tf.reduce_mean(model.p[:, i])
        tf.summary.scalar("losses/{}/p".format(names[i]), tf.reduce_mean(p))
        err = 0.5 * tf.reduce_mean(tf.square(model.p[:, i] - y[:, i]))
        tf.summary.scalar("losses/{}/error".format(names[i]), err)
    tf.summary.image("model/x", model.x, max_outputs=10)


def visualize_gradients(grads_and_vars):
    grads = [g for g, v in grads_and_vars]
    var_list = [v for g, v in grads_and_vars]
    for g, v in grads_and_vars:
        if g is None:
            continue
        tf.summary.histogram(v.name, v)
        tf.summary.histogram(v.name + "/grad", g)
        tf.summary.scalar("norms/" + v.name, tf.global_norm([v]))
        tf.summary.scalar("norms/" + v.name + "/grad", tf.global_norm([g]))
    grad_norm = tf.global_norm(grads)
    tf.summary.scalar("model/grad_global_norm", grad_norm)
    tf.summary.scalar("model/var_global_norm", tf.global_norm(var_list))


def run(resume_dir=None):
    os.makedirs(TENSORFLOW_OUT_DIR, exist_ok=True)
    if resume_dir is not None:
        date_str = resume_dir[:resume_dir.rindex('_')]
    else:
        date_str = DATE_STR
    sess_train_dir = date_str + '_train'
    sess_eval_dir = date_str + '_eval'
    os.makedirs(sess_train_dir, exist_ok=True)
    os.makedirs(sess_eval_dir, exist_ok=True)
    batch_size = 64  # Change this to fit in your GPU's memory
    x = tf.placeholder(tf.float32, (None,) + IMAGE_SHAPE)
    y = tf.placeholder(tf.float32, (None, NUM_TARGETS))
    log.info('creating model')
    with tf.variable_scope("model") as vs:
        model = Net(x, NUM_TARGETS)
        vs.reuse_variables()
        eval_model = Net(x, NUM_TARGETS, is_training=False)

    l2_norm = tf.global_norm(tf.trainable_variables())
    loss = 0.5 * tf.reduce_sum(tf.square(model.p - y)) / tf.to_float(tf.shape(x)[0])
    tf.summary.scalar("model/loss", loss)
    tf.summary.scalar("model/l2_norm", l2_norm)
    total_loss = loss + 0.0005 * l2_norm
    tf.summary.scalar("model/total_loss", total_loss)
    starter_learning_rate = 2e-6

    # TODO: add polyak averaging.
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step=model.global_step,
                                               decay_steps=73000, decay_rate=0.5, staircase=True)

    opt = tf.train.AdamOptimizer(learning_rate)
    tf.summary.scalar("model/learning_rate", learning_rate)
    grads_and_vars = opt.compute_gradients(total_loss)
    visualize_model(model, y)
    visualize_gradients(grads_and_vars)
    summary_op = tf.summary.merge_all()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads_and_vars, model.global_step)

    init_op = tf.global_variables_initializer()
    pretrained_var_map = {}
    for v in tf.trainable_variables():
        found = False
        for bad_layer in ["fc6", "fc7", "fc8"]:
            if bad_layer in v.name:
                found = True
        if found:
            continue

        pretrained_var_map[v.op.name[6:]] = v

    alexnet_saver = tf.train.Saver(pretrained_var_map)

    def init_fn(ses):
        log.info("Initializing parameters.")
        ses.run(init_op)
        alexnet_saver.restore(ses, "bvlc_alexnet.ckpt")

    saver = tf.train.Saver()
    sv = tf.train.Supervisor(is_chief=True,
                             logdir=sess_train_dir,
                             summary_op=None,  # Automatic summaries don't work with placeholders.
                             saver=saver,
                             global_step=model.global_step,
                             save_summaries_secs=30,
                             save_model_secs=60,
                             init_op=None,
                             init_fn=init_fn)

    eval_sw = tf.summary.FileWriter(sess_eval_dir)

    train_dataset = get_dataset(RECORDINGS_DIR, log)
    eval_dataset = get_dataset(RECORDINGS_DIR, log, train=False)
    config = tf.ConfigProto(allow_soft_placement=True)
    with sv.managed_session(config=config) as sess, sess.as_default():
        train_data_provider = train_dataset.iterate_forever(batch_size)
        log.info('Start tensorboard with \n\ttensorboard --logdir="' + TENSORFLOW_OUT_DIR +
              '" (In Windows tensorboard will be in your python env\'s Scripts folder, '
              'i.e. C:\\Users\\<YOU>\\Miniconda3\\envs\\tensorflow\\Scripts)\n'
              'Then navigate to http://localhost:6006 - You may see errors if Tensorboard was already '
              'started / has tabs open. If so, shut down Tenosrboard first and close all Tensorboard tabs. '
              'Sometimes you may just need to restart if you get CUDA device errors.')
        while True:
            for i in range(1000):
                images, targets = next(train_data_provider)
                feed_dict = {x: images, y: targets}  # , 'phase:0': 1}
                if i % 10 == 0 and i > 0:
                    # Summarize: Do this less frequently to speed up training time, more frequently to debug issues
                    _, summ = sess.run([train_op, summary_op], feed_dict)
                    sv.summary_computed(sess, summ)
                    sv.summary_writer.flush()
                else:
                    sess.run(train_op, feed_dict)
                step = model.global_step.eval()
                log.info('step %d', step)

            step = model.global_step.eval()
            # Do evaluation
            losses = []
            for images, targets in eval_dataset.iterate_once(batch_size):
                preds = sess.run(eval_model.p, {x: images})
                losses += [np.square(targets - preds)]
            losses = np.concatenate(losses)
            summary = tf.Summary()
            summary.value.add(tag="eval/loss", simple_value=float(0.5 * losses.sum() / losses.shape[0]))
            names = ["spin", "direction", "speed", "speed_change", "steering", "throttle"]
            for i in range(len(names)):
                summary.value.add(tag="eval/{}".format(names[i]), simple_value=float(0.5 * losses[:, i].mean()))
            eval_sw.add_summary(summary, step)
            eval_sw.flush()

if __name__ == "__main__":
    run()
