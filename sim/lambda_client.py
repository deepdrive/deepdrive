from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import inspect

import zmq
import pyarrow

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

    """
    def __init__(self, **kwargs):
        self.socket = None
        self.last_obz = None
        self.create_socket()
        # self._send(m.START, kwargs=kwargs)

    def eval(self, expression_str, local_vars=None):
        """
        Eval expressions against Unreal Python API
        :param expression_str: Expression to eval against embedded Python
            Hint: To write expressions in Python, you can use lambda_to_expr_str, i.e.
            expression_str = lambda_to_expr_str(lambda: [w.get_name() for w in ue.all_worlds()])
            worlds = client.eval(expression_str)

            which would be the same as:

            worlds = client.eval('[w.get_name() for w in ue.all_worlds()]')

        :param local_vars:
        :return:
        """
        local_vars = local_vars or {}
        try:
            msg = pyarrow.serialize([expression_str, local_vars]).to_buffer()
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

        # Creating a new socket on timeout is not working when other ZMQ connections are present in the process.
        # socket.RCVTIMEO = c.API_TIMEOUT_MS
        # socket.SNDTIMEO = c.API_TIMEOUT_MS

        socket.connect("tcp://localhost:%s" % API_PORT)
        self.socket = socket
        return socket

    def close(self):
        self.socket.close()


def lambda_to_expr_str(lambda_fn):
    """c.f. https://stackoverflow.com/a/52615415/134077"""
    if not lambda_fn.__name__ == "<lambda>":
        raise ValueError('Tried to convert non-lambda expression to string')
    else:
        lambda_str = inspect.getsource(lambda_fn).strip()
        curr_fn_name = inspect.stack()[0][3]
        lambda_no_vars = ''.join(lambda_str.split()).startswith('lambda:')
        if not (lambda_no_vars or lambda_str.startswith(('lambda ', curr_fn_name))):
            raise ValueError('lambda_fn was not declared on its own line '
                             '- getsource() returned\n\t' + lambda_str)
        expression_start = lambda_str.index(':') + 1
        expression_str = lambda_str[expression_start:].strip()
        if expression_str.endswith(')') and '(' not in expression_str:
            # i.e. l = lambda2str(lambda x: x + 1) => x + 1)
            expression_str = expression_str[:-1]
        return expression_str


def main():
    client = LambdaClient()
    answer = client.eval('x**2', {'x': 2})
    expr_str = lambda_to_expr_str(lambda x: x**2)
    client.eval(expr_str, {'x': 2})


if __name__ == '__main__':
    main()
