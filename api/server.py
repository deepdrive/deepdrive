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

METHODS = {
    STEP: DeepDriveEnv.step,
    RESET: DeepDriveEnv.reset
}


def create_socket(socket=None):
    if socket:
        socket.close()
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    # socket.RCVTIMEO = c.API_TIMEOUT_MS
    # socket.SNDTIMEO = c.API_TIMEOUT_MS
    socket.bind("tcp://*:%s" % c.API_PORT)
    return socket


def start():
    socket = create_socket()
    env = deepdrive.start()
    while True:
        try:
            msg = socket.recv()
            method, args = pyarrow.deserialize(msg)
            if method == STEP:
                resp = env.step(Action(*args).as_gym())
            socket.send(pyarrow.serialize(resp).to_buffer())
        except zmq.error.Again:
            log.info('Waiting for client')
            socket = create_socket(socket)

if __name__ == '__main__':
    start()