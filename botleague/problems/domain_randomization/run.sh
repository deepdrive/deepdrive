#!/usr/bin/env bash

set -e

export DEEPDRIVE_PUBLIC=true

# TODO!!! Think about how you'll fan out and fan in results for an individual bot.

python main.py \
      --server \
      --view-mode-period=16 \
      --max-steps=100 \
      --max-episodes=1 \
      --sync \
      --record \
      --eval-only \
      --render \
      --upload-gist \
      --image-resize-dims="[224,224,3]"
