# Requires NVIDIA drivers 390+
#
# Build (automated on dockerhub, so just use for local testing, don't push):
#
# Local usage: See Makefile
#
# VERSION=`cat VERSION | sed 's/ //g'`
# docker build --build-arg version=$VERSION -t deepdriveio/deepdrive:$VERSION -f Dockerfile .
#
# Usage:
# VERSION=`cat VERSION | sed 's/ //g
# docker run -it --net=host --runtime=nvidia deepdriveio/deepdrive:$VERSION

# Ubuntu 18
FROM adamrehn/ue4-runtime:tensorflow-virtualgl

#FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04
# For Ubuntu 16 also install
# python-software-properties

ARG version=3.0
USER root

# OS dependencies
RUN apt-get update; apt-get install -y \
        libsdl2-2.0 \
        software-properties-common \
        sudo \
        python3-venv \
        python3-pip \
        python3-dev \
        ffmpeg \
        git \
        vim \
      && cd /usr/local/bin \
      && ln -s /usr/bin/python3 python \
      && pip3 install --upgrade pip \
      && apt-get clean && rm -rf /var/lib/apt/lists/*

# Give the ue4 user the ability to install pip packages
ARG user=ue4
ENV DEEPDRIVE_USER=$user
#RUN chown -R $user:$user /home/$user
RUN adduser $user sudo
RUN echo "$user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Ensure ue4 owns directories needed to run Deepdrive
USER $user
#RUN sudo chown -R $user:$user /home/$user
WORKDIR /home/$user/src/deepdrive
#RUN sudo chown -R $user:$user .
#RUN sudo chmod -R 775 /home/$user

# Create our virtual environment
ARG venv=deepdrive_venv
RUN python -m venv /home/$user/$venv
ENV VIRTUAL_ENV /home/$user/$venv
ENV PATH /home/$user/$venv/bin:$PATH
RUN which pip

# Get the latest pip within the virtual environment
RUN pip install --upgrade pip
RUN which pip

# Install tensorflow
RUN pip install tensorflow-gpu

# Install OpenCV for web renderer
RUN pip install opencv-python

# Nice to have ipython in the container for bashing in
RUN pip install ipython

# Set Deepdrive directory so that we are not prompted to enter it interactively
ENV DEEPDRIVE_DIR=/home/$user/Deepdrive

# Cache pip requirements
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy minimum files needed for sim download
COPY config/ ./config/
COPY util/anonymize.py ./util/anonymize.py
COPY util/run_command.py ./util/run_command.py
COPY util/download.py ./util/download.py
COPY util/ensure_sim.py ./util/ensure_sim.py
COPY util/get_directories.py ./util/get_directories.py
COPY VERSION logs.py ./

# Download sim
RUN python -c "from util.ensure_sim import ensure_sim; ensure_sim();"

# Install
COPY install.py ./
RUN python -u install.py

# Commands useful for debugging
#ENV PYTHONPATH=/home/$user/src/deepdrive:$PYTHONPATH
#RUN which pip
#RUN pip install wheel
#RUN pip install "deepdrive > $version.*dev0" # TODO: Remove dev0 after 3.0 stable release
#RUN pip install sarge
#RUN pip install -r requirements.txt
#COPY logs.py utils.py install.py ./


# LibSDL2 offscreen opengl mode
ENV SDL_VIDEODRIVER=offscreen

# API
EXPOSE 5557/tcp

COPY . .

ENTRYPOINT ["/bin/bash", "docker/entrypoint.bash"]
CMD python main.py --server


# TODO: Support
# 1) Eval
# 2) Experiement
# 3) CI