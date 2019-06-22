from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import sys
import time

import zmq
import pyarrow

import logs
from utils import sizeof_fmt

log = logs.get_log(__name__)

API_PORT = 5657


class UEPyClient(object):
    """
    Call Unreal API via RPC  -
    Deepdrive sim API: http://bit.ly/2I1XB5u
    Unreal+Python integration: https://github.com/20tab/UnrealEnginePython
    """

    def __init__(self, **kwargs):
        self.socket = None
        self.last_obz = None
        self.create_socket()
        # self._send(m.START, kwargs=kwargs)

    def call(self, method: str, *args, **kwargs):
        """
        Eval expressions against Unreal Python API
        :param method API method to execute
        :param args - tuple or list of args to pass
        :param kwargs - dict of kwargs to pass
        :return: dict of form {'success': <False iff exception thrown>,
         'result': <eval(...) return value>}
        """
        ret = None
        try:
            start_serialize = time.time()
            msg = pyarrow.serialize([method, args, kwargs]).to_buffer()

            start_send = time.time()
            self.socket.send(msg)
            log.debug('send took %r' % (time.time() - start_send))

            start_receive = time.time()
            resp = self.socket.recv()
            log.debug('receive took %r', (time.time() - start_receive))

            size_formatted = sizeof_fmt(sys.getsizeof(resp))
            log.debug('receive size was %s', size_formatted)

            start_deserialize = time.time()
            ret = pyarrow.deserialize(resp)
            log.debug('deserialize took %r', (time.time() - start_deserialize))
        except zmq.error.Again:
            print('Waiting for uepy server')
            self.create_socket()
            return None
        finally:
            if ret is None:
                raise RuntimeError(
                    'Could not get response from uepy server. '
                    'Ensure your Arrow/pyarrow versions are compatible, and/or '
                    'try restarting sim or Unreal Editor. ')
            if not ret['success']:
                log.error(ret['result'])
                raise RuntimeError(
                    'Error executing %s(%s, %s) in Unreal - '
                    'Traceback above' % (method, str(args), str(kwargs)))
            return ret

    def create_socket(self):
        if self.socket:
            self.socket.close()
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)

        # Creating a new socket on timeout is not working when other ZMQ
        # connections are present in the process.
        socket.RCVTIMEO = 5000
        # socket.SNDTIMEO = c.API_TIMEOUT_MS

        socket.connect("tcp://localhost:%s" % API_PORT)
        self.socket = socket
        return socket

    def close(self):
        self.socket.close()


# TODO: Don't use a global singleton client
CLIENT = None


def rpc(method_name, *args, **kwargs):
    """Calls a method defined via api_methods.py in deepdrive-sim and run via
    the UnrealEnginePython embedded python interpreter"""
    global CLIENT
    if CLIENT is None:
        CLIENT = UEPyClient()

    start_call = time.time()
    ret = CLIENT.call(method_name, *args, **kwargs)
    log.debug('call took %r' % (time.time() - start_call))
    return ret


def main():
    answer = rpc('get_42')
    print('UnrealEnginePython evaluated answer to ', answer)

    answer = rpc('get_world')
    print('UnrealEnginePython evaluated answer to ', answer)


if __name__ == '__main__':
    main()
