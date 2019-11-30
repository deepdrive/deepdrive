#!/usr/bin/env bash

set -e

export DEEPDRIVE_PUBLIC=true

python main.py \
	--agent dagger_mobilenet_v2 \
	--use-latest-model \
	--max-steps=100 \
	--max-episodes=1 \
	--record \
	--eval-only \
	--upload-gist \
	--scenario=2 \
	--map=kevindale_bare \
	--camera-rigs=default_rig_1080p \
	--image-resize-dims="[224,224,3]"
	&& echo "Sim finished running successfully. Check above for results."
