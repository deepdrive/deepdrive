from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

from enum import Enum

from gym_deepdrive.renderer.pyglet_renderer import PygletRenderer
from gym_deepdrive.renderer.web_renderer import WebRenderer


class RendererType(Enum):
    WEB = 1
    PYGLET = 2


def renderer_factory(renderer_type=RendererType.WEB, cameras=None):
    if renderer_type is RendererType.WEB:
        return WebRenderer()
    elif renderer_type is RendererType.PYGLET:
        return PygletRenderer(cameras)
    else:
        raise NotImplementedError('Renderer type not recognized')
