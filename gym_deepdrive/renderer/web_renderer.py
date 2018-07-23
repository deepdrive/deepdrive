#!/usr/bin/env python
#
# Project: Video Streaming with Flask
# Author: Log0 <im [dot] ckieric [at] gmail [dot] com>
# Date: 2014/12/21
# Website: http://www.chioka.in/
# Description:
# Modified to support streaming out with webcams, and not just raw JPEGs.
# Most of the code credits to Miguel Grinberg, except that I made a small tweak. Thanks!
# Credits: http://blog.miguelgrinberg.com/post/video-streaming-with-flask
#
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python main.py".
# 3. Navigate the browser to the local webpage.
import time

import cv2
import multiprocessing

import pyarrow
import numpy as np
from flask import Flask, render_template, Response

import config as c
import logs
from gym_deepdrive.renderer.base_renderer import Renderer

log = logs.get_log(__name__)
app = Flask(__name__)


class WebRenderer(Renderer):
    def __init__(self):
        # TODO: Move source ZMQ to base renderer and replace pyglet renderer's use or multiprocessing
        import zmq
        self.prev_render_time = None
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        conn_string = 'tcp://localhost:%s' % c.STREAM_PORT
        log.info('Sending images to %s', conn_string)

        self.socket.connect(conn_string)
        self.web_server_process = multiprocessing.Process(target=background_server_process, name='streaming server')
        self.web_server_process.start()

    def __del__(self):
        self.socket.close()
        self.context.term()
        self.web_server_process.join()

    def render(self, obz):
        now = time.time()
        if self.prev_render_time:
            delta = now - self.prev_render_time
            log.debug('send image period %f', delta)
        self.prev_render_time = now
        if obz is not None:
            self.socket.send(pyarrow.serialize(obz['cameras']).to_buffer())


def background_server_process():
    StreamServer()

class StreamServer(object):
    """Note that this will only send frames to one viewer as we use a ZMQ pair server in the background for speed vs
    pub/sub"""
    def __init__(self):
        import zmq
        log.info('Pairing to zmq image stream')
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        conn_string = 'tcp://*:%s' % c.STREAM_PORT
        socket.bind(conn_string)
        log.info('Grabbing images from %s', conn_string)
        self.socket = socket
        self.context = context

        gen = self.gen

        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/video_feed')
        def video_feed():
            return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

        app.run(host='0.0.0.0', port=5000)

    def gen(self):
        prev_time = None
        while True:
            msg = self.socket.recv()

            # TODO: Add to deque of 1, then pull from the q at FPS

            now = time.time()
            if prev_time:
                delta = now - prev_time
                log.debug('recv image period %f', delta)
            prev_time = now
            if msg:
                cameras = pyarrow.deserialize(msg)

                if cameras is None:
                    continue
                else:
                    image = cameras[0]['image_raw']

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    ret, jpeg = cv2.imencode('.jpg', image)

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    def __del__(self):
        if self.socket is not None:
            # In background process
            self.socket.close()
            self.context.term()




