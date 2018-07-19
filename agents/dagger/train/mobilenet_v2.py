from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import os

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import config as c
import utils
from agents.dagger.net import MOBILENET_V2_SLIM_NAME
from vendor.tensorflow.models.research.slim.train_image_nn import slim_train_image_nn

IMG_SIZE = 224


def train_mobile_net(data_dir):

    if not os.path.exists(c.MNET2_CKPT_PATH + '.meta'):
        utils.download(c.MNET2_CKPT_URL + '?cache_bust=1', c.WEIGHTS_DIR, warn_existing=False, overwrite=True)

    # Fine-tune only the new layers
    slim_train_image_nn(
        dataset_name='deepdrive',
        dataset_split_name='train',
        dataset_dir=data_dir,
        model_name=MOBILENET_V2_SLIM_NAME,
        train_image_size=IMG_SIZE,
        checkpoint_path=c.MNET2_CKPT_PATH,
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

    eval_mobile_net(data_dir)

    # Fine-tune all layers
    slim_train_image_nn(
        dataset_name='deepdrive',
        resume_deepdrive=True,
        dataset_split_name='train',
        dataset_dir=data_dir,
        model_name=MOBILENET_V2_SLIM_NAME,
        train_image_size=IMG_SIZE,
        max_number_of_steps=49147,
        batch_size=16,
        learning_rate=0.00004,
        learning_rate_decay_type='fixed',
        save_interval_secs=180,
        save_summaries_secs=60,
        log_every_n_steps=20,
        optimizer='rmsprop',
        weight_decay=0.00004)

    eval_mobile_net(data_dir)


def eval_mobile_net(data_dir):
    from vendor.tensorflow.models.research.slim.eval_image_nn import slim_eval_image_nn

    slim_eval_image_nn(dataset_name='deepdrive', dataset_split_name='eval', dataset_dir=data_dir,
                       model_name=MOBILENET_V2_SLIM_NAME, eval_image_size=IMG_SIZE)
