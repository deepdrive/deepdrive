# Runs agents only. Running Unreal in Docker currently results in a Segmentation Fault using SDL2 offscreen mode.

# Usage:
#     python api/server.py
#     docker run --runtime=nvidia --net=host -it deepdrive python main.py --baseline --is-remote-client
#
#     ppo:
#     mnet weights: https://s3-us-west-1.amazonaws.com/deepdrive/weights/2018-05-22__03-04-53PM.zip
#     baselines results with weights: https://s3-us-west-1.amazonaws.com/deepdrive/weights/baselines_results.zip and change PPO_RESUME_PATH in config.py # TODO: Make this nicer
#     docker run -v /mnet_weight_dir/2018-05-22__03-04-53PM:/mnet_weight_dir/2018-05-22__03-04-53PM -v /baselines_results_dir/openai-2018-06-22-00-00-21-866205/checkpoints:/baselines_results_dir/openai-2018-06-22-00-00-21-866205/checkpoints --runtime=nvidia --net=host -it deepdrive python main.py --agent bootstrapped_ppo2 --experiment bootstrap --train --net-path=/mnet_weight_dir/2018-05-22__03-04-53PM/model.ckpt-45466 --sync --is-remote-client

FROM tensorflow/tensorflow:1.8.0-gpu-py3

# Minimize re-downloading / re-installing TODO: Cleanup / do this in python
###########################################################################
WORKDIR /src/deepdrive
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt
ENV DEEPDRIVE_DIR=/Deepdrive
ENV DEEPDRIVE_REMOTE_CLIENT=true
###########################################################################

ENV PYTHONPATH=/src/deepdrive

EXPOSE 5557/tcp

COPY . .