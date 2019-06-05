#!/usr/bin/env bash

set -e

export DEEPDRIVE_PUBLIC=true

# TODO!!! Think about how you'll fan out and fan in results.

python main.py \
      --server \
      --botleague \
      --view-mode-period=16 \
      --max-steps=100 \
      --max-episodes=1 \
      --sync \
      --record \
      --eval-only \
      --render \
      --public \
      --upload-gist \
      --image-resize-dims="[224,224,3]"
