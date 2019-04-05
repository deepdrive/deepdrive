#!/usr/bin/env bash

set -e

python main.py \
      --net-path="https://s3-us-west-1.amazonaws.com/deepdrive/weights/domain_randomization_2019-03-04__02-11-53PM.zip" \
      --view-mode-period=16 \
      --max-steps=100 \
      --max-episodes=1 \
      --sync \
      --record \
      --eval-only \
      --remote \
      --render \
      --eval-only \
      --upload-gist