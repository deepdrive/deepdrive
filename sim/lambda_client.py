from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import inspect

import zmq
import pyarrow

import logs

log = logs.get_log(__name__)

API_PORT = 5657


class LambdaClient(object):
    """
    Call Unreal API with Python expressions - see: https://github.com/20tab/UnrealEnginePython

    The Python that executes passed expressions is embedded in Unreal Engine
    and currently is Python 3.5.

    Variables server has access to:
    `ue` - unreal_engine Python module
    `world` - current running world

    Examples:

    ue.all_worlds()

    dir(world)

    world.all_actors()

    location = world.all_actors()[0].get_actor_location()

    location.z += 100

    world.all_actors()[0].set_actor_location(location)

    # TODO: Show how to access vehicle and tire models

    """

    def __init__(self, **kwargs):
        self.socket = None
        self.last_obz = None
        self.create_socket()
        # self._send(m.START, kwargs=kwargs)

    def call(self, method, *args, **kwargs):
        """
        Eval expressions against Unreal Python API
        :param method API method to execute
        :param args - tuple or list of args to pass
        :param kwargs - dict of kwargs to pass
        :return: dict of form {'success': <False iff exception thrown>, 'result': <eval(...) return value>}
        """
        ret = None
        try:
            msg = pyarrow.serialize([method, args, kwargs]).to_buffer()
            self.socket.send(msg)
            ret = pyarrow.deserialize(self.socket.recv())
        except zmq.error.Again:
            print('Waiting for lambda uepy server')
            self.create_socket()
            return None
        finally:
            if ret is None:
                raise RuntimeError('Could not get response from lambda server. Try restarting sim or Unreal Editor.')
            if not ret['success']:
                log.error(ret['result'])
                raise RuntimeError(
                    'Error executing %s(%s, %s) in Unreal - Traceback above' % (method, str(args), str(kwargs)))
            return ret

    def create_socket(self):
        if self.socket:
            self.socket.close()
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)

        # Creating a new socket on timeout is not working when other ZMQ connections are present in the process.
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
    """Calls a method defined via api_methods.py in deepdrive-sim and run via the UnrealEnginePython
    embedded python interpreter"""
    global CLIENT
    if CLIENT is None:
        CLIENT = LambdaClient()

    return CLIENT.call(method_name, *args, **kwargs)


def main():
    answer = rpc('get_42')
    print('UnrealEnginePython evaluated answer to ', answer)

    answer = rpc('get_world')
    print('UnrealEnginePython evaluated answer to ', answer)


if __name__ == '__main__':
    main()
