# Make sure drivers are >= 390
# sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 deepdrive:latest python main.py

FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04 as glvnd

FROM nvidia/cuda:9.1-runtime-ubuntu16.04

COPY --from=glvnd /usr/local/lib/x86_64-linux-gnu /usr/local/lib/x86_64-linux-gnu
COPY --from=glvnd /usr/local/lib/i386-linux-gnu /usr/local/lib/i386-linux-gnu
COPY --from=glvnd /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json
COPY --from=glvnd /etc/ld.so.conf.d/glvnd.conf /etc/ld.so.conf.d/glvnd.conf

RUN ldconfig

ENV LD_LIBRARY_PATH /usr/local/lib/x86_64-linux-gnu:/usr/local/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

RUN apt-get update; apt-get install -y software-properties-common python-software-properties

RUN apt-get install -y python3-pip python3-dev \
      && cd /usr/local/bin \
      && ln -s /usr/bin/python3 python \
      && pip3 install --upgrade pip

RUN apt-get install -y mesa-utils
RUN apt-get update; apt-get install -y libsdl2-2.0

ENV SDL_VIDEODRIVER=dummy

# Minimize re-downloading / re-installing TODO: Cleanup / do this in python
###########################################################################
COPY requirements.txt src/deepdrive/
COPY config.py src/deepdrive/
COPY logs.py src/deepdrive/
COPY utils.py src/deepdrive/

WORKDIR src/deepdrive

RUN pip install -r requirements.txt

ENV DEEPDRIVE_DIR=/appuser/Deepdrive
RUN python -c "import utils; utils.download_sim()"
###########################################################################

RUN apt-get install -y libxxf86vm1 libglu1-mesa libopenal1 libssl1.0.0

COPY . .

#RUN python -u install.py --unattended

RUN chmod +x /appuser/Deepdrive/sim/LinuxNoEditor/DeepDrive/Binaries/Linux/DeepDrive-Linux-Shipping

RUN groupadd -g 999 appuser && \
    useradd -r -u 999 -g appuser appuser

RUN mkdir -p "/home/appuser/.config"

RUN chown -R appuser "/home/appuser"
RUN chgrp -R appuser "/home/appuser"

ENV SDL_VIDEODRIVER=offscreen

USER appuser

#RUN echo "root:Docker!" | chpasswd

#RUN apt-get /media/b/data-ext4/src/CarlaUE4clean && rm -rf /var/lib/apt/lists/*