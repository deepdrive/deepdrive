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


class APIClient(object):
    def __init__(self):
        self.socket = None
        self.create_socket()

    def send(self, method, args=None):
        args = args or []
        try:
            msg = pyarrow.serialize([method, args]).to_buffer()
            self.socket.send(msg)
            return pyarrow.deserialize(self.socket.recv())
        except zmq.error.Again:
            print('Waiting for server')
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

def main():
    client = APIClient()
    obz, reward, done, info = client.send('step', args=[1, 1])
    pass

if __name__ == '__main__':
    main()