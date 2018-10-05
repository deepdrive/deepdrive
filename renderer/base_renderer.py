from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)


class Renderer(object):
    def render(self, obz):
        raise not NotImplementedError('Please define render in your child class')


