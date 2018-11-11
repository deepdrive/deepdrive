from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

from datetime import datetime
import glob
import os
import sys
from multiprocessing import Process

import config as c
import utils
from agents.dagger.net import MOBILENET_V2_SLIM_NAME
from agents.dagger.train import hdf5_to_tfrecord
from install import get_tf_valid
from vendor.tensorflow.models.research.slim.eval_image_nn import slim_eval_image_nn
from vendor.tensorflow.models.research.slim.train_image_nn import slim_train_image_nn
import logs

log = logs.get_log(__name__)


IMG_SIZE = 224

def train_mobile_net(data_dir, resume_dir=None):
    if not get_tf_valid():
        raise RuntimeError('Invalid Tensorflow version detected. See above for details.')

    """# Should see eval steering error of about 0.1135 / Original Deepdrive 2.0 baseline steering error eval was ~0.2, 
    train steering error: ~0.08"""


    if not os.path.exists(c.MNET2_PRETRAINED_PATH + '.meta'):
        utils.download(c.MNET2_PRETRAINED_URL + '?cache_bust=1', c.WEIGHTS_DIR, warn_existing=False, overwrite=True)

    if not glob.glob(data_dir + '/*.tfrecord') and glob.glob(data_dir + '/*/*.hdf5'):
        log.warn('Performing one time translation of HDF5 to TFRecord')
        hdf5_to_tfrecord.encode()


    # Execute sessions in separate processes to ensure Tensorflow cleans up nicely
    # Without this, fine_tune_all_layers would crash towards the end with
    #  Error polling for event status: failed to query event: CUDA_ERROR_LAUNCH_FAILED:

    if resume_dir is None:
        train_dir = datetime.now().strftime(os.path.join(c.TENSORFLOW_OUT_DIR, '%Y-%m-%d__%I-%M-%S%p'))
        print('train_dir is ', train_dir)
        isolate_in_process(fine_tune_new_layers, args=(data_dir, train_dir))
        isolate_in_process(eval_mobile_net, args=(data_dir,))
    else:
        train_dir = resume_dir  # TODO: Fix MNET2/tf-slim issue resuming with train_dir
        print('resume_dir is ', resume_dir)

    isolate_in_process(fine_tune_all_layers, args=(data_dir, train_dir))
    isolate_in_process(eval_mobile_net, args=(data_dir,))
    log.info('Finished training')


def isolate_in_process(target, args):
    p = Process(target=target, args=args)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError("""
        Process finished with error. See above for details. HINTS: 
        
        1) If you see CUDA errors like:
        
                Error polling for event status: failed to query event: CUDA_ERROR_LAUNCH_FAILED
        
            try running with --use-latest-model to resume training from the last checkpoint. 
        
        2) Running training outside of IDE's like PyCharm seems to be more stable.
        """)


def fine_tune_new_layers(data_dir, train_dir):
    slim_train_image_nn(
        dataset_name='deepdrive',
        dataset_split_name='train',
        train_dir=train_dir,
        dataset_dir=data_dir,
        model_name=MOBILENET_V2_SLIM_NAME,
        train_image_size=IMG_SIZE,
        checkpoint_path=c.MNET2_PRETRAINED_PATH,
        checkpoint_exclude_scopes='MobilenetV2/Logits,MobilenetV2/Predictions,MobilenetV2/predics',
        trainable_scopes='MobilenetV2/Logits,MobilenetV2/Predictions,MobilenetV2/predics',
        max_number_of_steps=2000,
        batch_size=32,
        learning_rate=0.0001,
        learning_rate_decay_type='fixed',
        save_interval_secs=10,
        save_summaries_secs=60,
        log_every_n_steps=20,
        optimizer='rmsprop',
        weight_decay=0.00004)


def fine_tune_all_layers(data_dir, train_dir):
    slim_train_image_nn(
        dataset_name='deepdrive',
        checkpoint_path=train_dir,
        dataset_split_name='train',
        dataset_dir=data_dir,
        model_name=MOBILENET_V2_SLIM_NAME,
        train_image_size=IMG_SIZE,
        max_number_of_steps=10**5,
        batch_size=16,
        learning_rate=0.00004,
        learning_rate_decay_type='fixed',
        save_interval_secs=180,
        save_summaries_secs=60,
        log_every_n_steps=20,
        optimizer='rmsprop',
        weight_decay=0.00004)


def eval_mobile_net(data_dir):
    slim_eval_image_nn(dataset_name='deepdrive', dataset_split_name='eval', dataset_dir=data_dir,
                       model_name=MOBILENET_V2_SLIM_NAME, eval_image_size=IMG_SIZE)

