from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import zmq
import random
import sys
import time
import pyarrow

import config as c
import logs
from gym_deepdrive.envs.deepdrive_gym_env import Action

log = logs.get_log(__name__)


class RemoteEnv(object):
    def __init__(self, **kwargs):
        self.socket = None
        self.create_socket()
        self._send('start', kwargs=kwargs)

    def _send(self, method, args=None, kwargs=None):
        args = args or []
        kwargs = kwargs or {}
        try:
            msg = pyarrow.serialize([method, args, kwargs]).to_buffer()
            self.socket.send(msg)
            return pyarrow.deserialize(self.socket.recv())
        except zmq.error.Again:
            log.info('Waiting for server')
            self.create_socket()
            return None

    def create_socket(self):
        if self.socket:
            self.socket.close()
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        # socket.RCVTIMEO = c.API_TIMEOUT_MS
        # socket.SNDTIMEO = c.API_TIMEOUT_MS
        socket.connect("tcp://localhost:%s" % c.API_PORT)
        self.socket = socket
        return socket

    def step(self, action):
        if isinstance(action, list):
            action = Action.from_gym(action)
        resp = self._send('step', kwargs=dict(
            steering=action.steering, throttle=action.throttle, handbrake=action.handbrake, brake=action.brake,
            has_control=action.has_control))
        return resp

    def reset(self):
        self._send('reset')
