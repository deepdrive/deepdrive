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

# TODO Remove this
COPY snake.py snake.py

COPY . deepdrive

RUN python deepdrive/install.py

#RUN apt-get clean && rm -rf /var/lib/apt/lists/*