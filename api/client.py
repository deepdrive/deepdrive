from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import zmq

import numpy as np
import pyarrow
from gym import spaces

import config as c
import api.methods as m
import logs
from gym_deepdrive.envs.deepdrive_gym_env import Action

log = logs.get_log(__name__)


def deserialize_space(resp):
    if resp['type'] == "<class 'gym.spaces.box.Box'>":
        ret = spaces.Box(resp['low'], resp['high'], dtype=resp['dtype'])
    else:
        raise RuntimeError('Unsupported action space type')
    return ret


class RemoteEnv(object):
    def __init__(self, **kwargs):
        self.socket = None
        self.create_socket()
        self._send(m.START, kwargs=kwargs)

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
        resp = self._send(m.STEP, args=[action])
        return resp

    def reset(self):
        return self._send(m.RESET)

    @property
    def action_space(self):
        resp = self._send(m.ACTION_SPACE)
        ret = deserialize_space(resp)
        return ret

    @property
    def observation_space(self):
        resp = self._send(m.OBSERVATION_SPACE)
        ret = deserialize_space(resp)
        return ret

    @property
    def metadata(self):
        return self._send(m.METADATA)

    @property
    def reward_range(self):
        return self._send(m.REWARD_RANGE)
