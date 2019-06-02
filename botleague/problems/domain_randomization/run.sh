#!/usr/bin/env bash

set -e

# Run with DEEPDRIVE_PUBLIC=true on eval servers / botleague problem endpoint servers

# TODO!!! Think about how you'll fan out and fan in results.

python main.py \
      --server \
      --view-mode-period=16 \
      --max-steps=100 \
      --max-episodes=1 \
      --sync \
      --record \
      --eval-only \
      --render \
      --eval-only \
      --upload-gist