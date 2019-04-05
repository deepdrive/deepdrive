from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import zmq

import pyarrow
from gym import spaces

import config as c
import api.methods as m
import logs
from sim.action import Action
from renderer import renderer_factory

log = logs.get_log(__name__)


def deserialize_space(resp):
    if resp['type'] == "<class 'gym.spaces.box.Box'>":
        ret = spaces.Box(resp['low'], resp['high'], dtype=resp['dtype'])
    else:
        raise RuntimeError('Unsupported action space type')
    return ret


class Client(object):
    """
    A Client object acts as a remote proxy to the deepdrive gym environment.
    Methods that you would call on the env, like step() are also called on
    this object, with communication over the network -
    rather than over shared memory (for observations) and network
    (for transactions like reset) as is the case with the locally run
    sim/gym_env.py.
    This allows the agent and environment to run on separate machines, but
    with the same API as a local agent, namely the gym API.

    The local gym environment is then run by api/server.py which proxies
    RPC's from this client to the local environment.

    All network communication happens over ZMQ to take advantage of their
    highly optimized cross-language / cross-OS sockets.

    NOTE: This will obviously run more slowly than a local agent which
    communicates sensor data over shared memory.
    """
    def __init__(self, **kwargs):
        self.socket = None
        self.last_obz = None
        self.create_socket()
        self.should_render = kwargs.get('client_render', False)
        if kwargs['cameras'] is None:
            kwargs['cameras'] = [c.DEFAULT_CAM]
        if self.should_render:
            self.renderer = renderer_factory(cameras=kwargs['cameras'])
        else:
            self.renderer = None
        log.info('Waiting for sim to start on server...')
        # TODO: Fix connecting to an open sim
        self._send(m.START, kwargs=kwargs)
        log.info('===========> Deepdrive sim started')

    def _send(self, method, args=None, kwargs=None):
        args = args or []
        kwargs = kwargs or {}
        try:
            msg = pyarrow.serialize([method, args, kwargs]).to_buffer()
            self.socket.send(msg)
            return pyarrow.deserialize(self.socket.recv())
        except zmq.error.Again:
            log.info('Waiting for Deepdrive API server...')
            self.create_socket()
            return None

    def create_socket(self):
        if self.socket:
            self.socket.close()
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)

        # Creating a new socket on timeout is not working when other ZMQ
        # connections are present in the process.
        # socket.RCVTIMEO = c.API_TIMEOUT_MS
        # socket.SNDTIMEO = c.API_TIMEOUT_MS

        socket.connect("tcp://localhost:%s" % c.API_PORT)
        self.socket = socket
        return socket

    def step(self, action):
        if isinstance(action, Action):
            action = action.as_gym()
        obz, reward, done, info = self._send(m.STEP, args=[action])
        if not obz:
            obz = None
        self.last_obz = obz
        if self.should_render:
            self.render()
        return obz, reward, done, info

    def reset(self):
        return self._send(m.RESET)

    def render(self):
        """
        We pass the obz through an instance variable to comply with
        the gym api where render() takes 0 arguments
        """
        if self.last_obz is not None:
            self.renderer.render(self.last_obz)

    def change_cameras(self, cameras):
        return self._send(m.CHANGE_CAMERAS, args=[cameras])

    def close(self):
        self._send(m.CLOSE)
        self.socket.close()

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
