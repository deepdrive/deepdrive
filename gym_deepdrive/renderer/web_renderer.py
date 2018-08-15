import threading
import time

import cv2
import multiprocessing

import pyarrow
import numpy as np
from flask import Flask, render_template
from werkzeug.wrappers import Response

from collections import deque

import config as c
import logs
import utils
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
        self.frame_queue = deque(maxlen=1)

        gen = self.gen

        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/video_feed')
        def video_feed():
            return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

        self.frame_worker = threading.Thread(target=frame_worker, args=(socket, self.frame_queue))
        self.frame_worker.start()

        self.start_server()

    def start_server(self):
        # To see various options explored, go here: https://github.com/crizCraig/mjpg_server_test
        use_cherrypy_wsgi = True
        if use_cherrypy_wsgi:
            import cherrypy
            from paste.translogger import TransLogger

            # Enable WSGI access logging via Paste
            app_logged = TransLogger(app)

            # Mount the WSGI callable object (app) on the root directory
            cherrypy.tree.graft(app_logged, '/')

            # Set the configuration of the web server
            cherrypy.config.update({
                'engine.autoreload.on': False,
                'checker.on': False,
                'tools.log_headers.on': False,
                'request.show_tracebacks': False,
                'request.show_mismatched_params': False,
                'log.screen': False,
                'server.socket_port': 5000,
                'server.socket_host': '0.0.0.0',
                'server.thread_pool': 30,
                'server.socket_queue_size': 30,
                'server.accepted_queue_size': -1,
            })

            # Start the CherryPy WSGI web server
            cherrypy.engine.start()
            cherrypy.engine.block()
        else:
            app.run(host='0.0.0.0', port=5000)

    def gen(self):
        jpeg_bytes = None
        while True:
            if len(self.frame_queue) > 0:
                if jpeg_bytes is not self.frame_queue[0]:
                    yield self.frame_queue[0]
                    jpeg_bytes = self.frame_queue[0]
                time.sleep(0.001)

    def __del__(self):
        if self.socket is not None:
            # In background process
            self.socket.close()
            self.context.term()

# def frame_worker(socket, queue):
#     while True:
#         msg = socket.recv()
#         if msg:
#             cameras = pyarrow.deserialize(msg)
#             if cameras is not None:
#
#                 image = cameras[0]['image_raw']
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#                 ret, jpeg = cv2.imencode('.jpg', image)
#                 queue.append(b'--frame\r\n'
#                              b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


def frame_worker(socket, queue):
    while True:
        msg = socket.recv()
        if msg:
            cameras = pyarrow.deserialize(msg)
            all_cams_image = None
            if cameras is not None:
                for cam_idx, cam in enumerate(cameras):
                    image = cam['img_raw'] if 'img_raw' in cam else cam['image']
                    depth = np.ascontiguousarray(utils.depth_heatmap(np.copy(cam['depth'])))
                    try:
                        image = np.concatenate((image, depth), axis=1)
                    except ValueError as e:
                        log.error('Could not concatenate image with shape %r and depth with shape %r %s',
                                  image.shape, depth.shape, str(e))

                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if all_cams_image is None:
                        all_cams_image = image
                    else:
                        all_cams_image = np.concatenate((all_cams_image, image), axis=0)

                ret, jpeg = cv2.imencode('.jpg', all_cams_image)
                queue.append(b'--frame\r\n'
                             b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

