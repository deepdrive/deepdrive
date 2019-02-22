from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (str)

import zmq
import pyarrow
from gym import spaces

import logs

import config as c
import api.methods as m
import sim

log = logs.get_log(__name__)


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
                if self.env is None and method != m.START:
                    resp = 'No environment started, please send start request'
                    log.error('Client sent request with no environment started')
                elif method == m.START:
                    allowed_args = ['experiment', 'env', 'cameras', 'combine_box_action_spaces', 'is_discrete',
                                    'preprocess_with_tensorflow', 'is_sync']
                    if c.IS_EVAL:
                        allowed_args.remove('env')
                        allowed_args.remove('is_sync')
                    for key in list(kwargs):
                        if key not in allowed_args:
                            del kwargs[key]
                    self.env = sim.start(**kwargs)
                elif method == m.STEP:
                    resp = self.env.step(args[0])
                elif method == m.RESET:
                    resp = self.env.reset()
                elif method == m.ACTION_SPACE or method == m.OBSERVATION_SPACE:
                    resp = self.serialize_space(resp)
                elif method == m.REWARD_RANGE:
                    resp = self.env.reward_range
                elif method == m.METADATA:
                    resp = self.env.metadata
                else:
                    log.error('Invalid API method')

                self.socket.send(pyarrow.serialize(resp).to_buffer())

            except zmq.error.Again:
                log.info('Waiting for client')
                self.create_socket()

    def serialize_space(self, resp):
        space = self.env.action_space
        space_type = type(space)
        if space_type == spaces.Box:
            resp = {'type': str(space_type),
                    'low': space.low,
                    'high': space.high,
                    'dtype': str(space.dtype)
                    }
        else:
            raise RuntimeError(str(space_type) + ' not supported')
        return resp


def start():
    server = Server()
    server.run()


if __name__ == '__main__':
    start()
