#!/usr/bin/env bash

python main.py \
      --net-path="https://s3-us-west-1.amazonaws.com/deepdrive/weights/domain_randomization_2019-03-04__02-11-53PM.zip" \
      --view-mode-period=16 \
      --max-steps=100 \
      --max-episodes=2 \
      --sync \
      --record \
      --eval-only \
      --remote \
      --render \
      --eval-only \
      --upload-gist \
      --public