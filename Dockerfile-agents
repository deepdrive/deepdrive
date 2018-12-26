# Build (automated on dockerhub, so just use for local testing, don't push):
#     docker build -t deepdriveio/deepdrive:agents-2.1 -f Dockerfile-agents .
# Usage:
#     Run API server somewhere, i.e. python api/server.py
#
#     Then run the client:
#     docker run --runtime=nvidia --net=host -it deepdriveio/deepdrive:agents-2.1 python3 main.py --baseline --is-remote-client
#
#     ppo train:
#     baselines results with weights: https://s3-us-west-1.amazonaws.com/deepdrive/weights/baselines_results.zip and change PPO_RESUME_PATH in config.py # TODO: Promote PPO resume path to command line arg
#     docker run -v /baselines_results_dir/openai-2018-06-22-00-00-21-866205/checkpoints:/baselines_results_dir/openai-2018-06-22-00-00-21-866205/checkpoints --runtime=nvidia --net=host -it deepdrive python3 main.py --agent bootstrapped_ppo2 --experiment bootstrap --train --sync --is-remote-client
#
#     ppo run:
#     docker run --runtime=nvidia --net=host -it deepdrive python3 main.py --agent bootstrapped_ppo2 --experiment bootstrap --eval-only --sync --is-remote-client
#
#     view:
#     http://localhost:5558

FROM tensorflow/tensorflow:1.8.0-gpu-py3

RUN apt-get update; apt-get install -y python-opencv

# Minimize re-downloading / re-installing TODO: Cleanup / do this in python
###########################################################################
WORKDIR /src/deepdrive
RUN pip3 install "https://s3-us-west-1.amazonaws.com/deepdrive/wheels/deepdrive/deepdrive-sim/140/140.1/Plugins/DeepDrivePlugin/Source/wheelhouse/deepdrive-2.0.20180812221209-cp35-cp35m-manylinux1_x86_64.whl"
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Dependencies for streaming agent sensors
RUN pip3 install --no-cache-dir opencv-python flask

ENV DEEPDRIVE_DIR=/Deepdrive
ENV DEEPDRIVE_REMOTE_CLIENT=true
#TODO: Download baseline weights to avoid re-downloading every training run
###########################################################################

ENV PYTHONPATH=/src/deepdrive

# API Port
EXPOSE 5557/tcp

# Render stream port
EXPOSE 5558/tcp

COPY . .

CMD python3 main.py