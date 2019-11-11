#!/usr/bin/env bash

set -e

export DEEPDRIVE_PUBLIC=true

python main.py \
      --server \
      --path-follower \
      --max-steps=500 \
      --max-episodes=1 \
      --record \
      --eval-only \
      --render \
      --eval-only \
      --upload-gist \
      --map=kevindale_bare \
      --scenario=2 \
      --image-resize-dims="[224,224,3]" \
      && echo "Sim finished running successfully. Check above for results."