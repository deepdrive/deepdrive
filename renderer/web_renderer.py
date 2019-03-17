from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
import time
import multiprocessing
import pyarrow
from flask import Flask

import config as c
import logs
import utils
from renderer.base_renderer import Renderer

log = logs.get_log(__name__)
app = Flask(__name__)

web_renderer = None

"""
Usage:
Call main.py with --render and navigate to http://0.0.0.0:5000
"""

# TODO: Support rendering saved HDF5 and tfrecord's


def get_web_renderer():
    # Singleton is a hack around difficulties setting up multiple ZMQ contexts on same port in the same process
    global web_renderer
    if web_renderer is None:
        web_renderer = WebRenderer()
    return web_renderer


class WebRenderer(Renderer):

    def __init__(self):

        # TODO: Move source ZMQ to base renderer and replace pyglet renderer's use of multiprocessing
        import zmq

        self.prev_render_time = None
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        conn_string = 'tcp://localhost:%s' % c.STREAM_PORT
        log.debug('Sending images over ZMQ to %s', conn_string)

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
    from renderer.stream_server import StreamServer
    StreamServer(app)

