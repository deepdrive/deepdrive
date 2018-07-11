# Runs agents only. Running Unreal in Docker currently results in a Segmentation Fault using SDL2 offscreen mode.

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