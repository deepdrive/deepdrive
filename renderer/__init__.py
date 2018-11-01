from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from enum import Enum


import logs

log = logs.get_log(__name__)

class RendererType(Enum):
    WEB = 1
    PYGLET = 2


def renderer_factory(renderer_type=None, cameras=None):
    if renderer_type is None:
        try:
            import cv2
            renderer_type = RendererType.WEB
        except ImportError as e:
            log.warn(
                'Could not find opencv - install with `pip install opencv-python`. Falling back to pyglet renderer')
            renderer_type = RendererType.PYGLET

    if renderer_type is RendererType.WEB:
        from renderer.web_renderer import get_web_renderer
        return get_web_renderer()
    elif renderer_type is RendererType.PYGLET:
        from renderer.pyglet_renderer import PygletRenderer
        return PygletRenderer(cameras)
    else:
        raise NotImplementedError('Renderer type not recognized')
