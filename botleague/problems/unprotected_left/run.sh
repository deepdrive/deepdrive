#!/usr/bin/env bash

set -e

export DEEPDRIVE_PUBLIC=true

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
      --image-resize-dims="[224,224,3]" \
      --scenario 2 \
      && echo "Sim finished running successfully. Check above for results."
