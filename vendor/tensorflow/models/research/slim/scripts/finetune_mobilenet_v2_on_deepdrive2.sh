#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 2. Fine-tunes Mobilenet v2 on the Deepdrive train set.
# 3. Evaluates the model on the Deepdrive eval set.
#
# Usage:
# scripts/finetune_mobilenet_v2_on_deepdrive.sh
#
# Should see steering error of about 0.1135 / Original Deepdrive 2.0 baseline steering error eval was ~0.2, train steering error: ~0.08
set -ev

# TODO: Move this to Python

# Where the dataset is saved to.
DATASET_DIR=/media/a/data-ext4/deepdrive-data
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SLIM_DIR="$(dirname "$DIR")"
export PYTHONPATH=${PYTHONPATH}:

### Fine-tune only the new layers
python ${SLIM_DIR}/train_image_classifier.py \
  --dataset_name=deepdrive \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v2_deepdrive \
  --train_image_size=224 \
  --checkpoint_path=/home/a/mnet-test/checkpoints/mobilenet_v2_1.0_224.ckpt \
  --checkpoint_exclude_scopes=MobilenetV2/Logits,MobilenetV2/Predictions,MobilenetV2/predics \
  --trainable_scopes=MobilenetV2/Logits,MobilenetV2/Predictions,MobilenetV2/predics \
  --max_number_of_steps=2000 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=10 \
  --save_summaries_secs=60 \
  --log_every_n_steps=20 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

python ${SLIM_DIR}/eval_image_classifier.py \
  --dataset_name=deepdrive \
  --dataset_split_name=eval \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v2_deepdrive \
  --eval_image_size=224

# Fine-tune all layers
python ${SLIM_DIR}/train_image_classifier.py \
  --dataset_name=deepdrive \
  --resume_deepdrive \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v2_deepdrive \
  --train_image_size=224 \
  --max_number_of_steps=49147 \
  --batch_size=16 \
  --learning_rate=0.00004 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=180 \
  --save_summaries_secs=60 \
  --log_every_n_steps=20 \
  --optimizer=rmsprop \
  --weight_decay=0.00004


python ${SLIM_DIR}/eval_image_classifier.py \
  --dataset_name=deepdrive \
  --dataset_split_name=eval \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v2_deepdrive \
  --eval_image_size=224


#  --dataset_name=deepdrive 
#  --dataset_split_name=train 
#  --dataset_dir=${DATASET_DIR} 
#  --model_name=mobilenet_v2_deepdrive 
#  --train_image_size=224 
#  --checkpoint_path=/home/a/mnet-test/checkpoints/mobilenet_v2_1.0_224.ckpt 
#  --checkpoint_exclude_scopes=MobilenetV2/Logits,MobilenetV2/Predictions,MobilenetV2/predics 
#  --trainable_scopes=MobilenetV2/Logits,MobilenetV2/Predictions,MobilenetV2/predics 
#  --max_number_of_steps=2000 
#  --batch_size=32 
#  --learning_rate=0.0001 
#  --learning_rate_decay_type=fixed 
#  --save_interval_secs=10 
#  --save_summaries_secs=60 
#  --log_every_n_steps=20 
#  --optimizer=rmsprop 
#  --weight_decay=0.00004
