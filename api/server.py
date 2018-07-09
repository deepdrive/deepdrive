from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import zmq
import random
import sys
import time
import numpy as np
import pyarrow

import logs
import utils

import deepdrive
from gym_deepdrive.envs.deepdrive_gym_env import DeepDriveEnv, Action
import config as c

log = logs.get_log(__name__)

STEP = 'step'
RESET = 'reset'
START = 'start'

CONN_STRING = "tcp://*:%s" % c.API_PORT


class Server(object):
    def __init__(self):
        self.socket = None
        self.context = None
        self.env = None

    def create_socket(self):
        if self.socket is not None:
            log.info('Closed server socket')
            self.socket.close()
        if self.context is not None:
            log.info('Destroyed context')
            self.context.destroy()

        self.context = zmq.Context()
        socket = self.context.socket(zmq.PAIR)
        # socket.RCVTIMEO = c.API_TIMEOUT_MS
        # socket.SNDTIMEO = c.API_TIMEOUT_MS
        socket.bind(CONN_STRING)
        self.socket = socket
        return socket

    def run(self):
        self.create_socket()
        log.info('Environment server started at %s', CONN_STRING)
        while True:
            try:
                msg = self.socket.recv()
                method, args, kwargs = pyarrow.deserialize(msg)
                resp = None
                if self.env is None and method != START:
                    resp = 'No environment started, please send start request'
                    log.error('Client sent request with no environment started')
                elif method == START:
                    allowed_args = ['experiment_name', 'env', 'cameras', 'combine_box_action_spaces', 'is_discrete',
                                    'preprocess_with_tensorflow', 'is_sync']
                    if c.IS_EVAL:
                        allowed_args.remove('env')
                        allowed_args.remove('is_sync')
                    for key in kwargs:
                        if key not in allowed_args:
                            del kwargs[key]
                    self.env = deepdrive.start(**kwargs)
                elif method == STEP:
                    resp = self.env.step(Action(*args, **kwargs).as_gym())
                elif method == RESET:
                    resp = self.env.reset()
                else:
                    log.error('Invalid API method')

                self.socket.send(pyarrow.serialize(resp).to_buffer())

            except zmq.error.Again:
                log.info('Waiting for client')
                self.create_socket()


def start():
    server = Server()
    server.run()


if __name__ == '__main__':
    start()
