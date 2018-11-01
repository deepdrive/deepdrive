from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import time

import numpy as np

from multiprocessing import Process, Queue

import utils
import logs
from renderer.base_renderer import Renderer

log = logs.get_log(__name__)

DRAW_FPS = True


class PygletRenderer(Renderer):
    def __init__(self, cameras):
        self.prev_render_time = None
        # TODO: Use ZMQ/pyarrow instead of multiprocessing for faster transfer
        q = Queue(maxsize=1)
        p = Process(target=render_cameras, args=(q, cameras))
        p.start()
        self.pyglet_process = p
        self.pyglet_queue = q

    def render(self, obz):
        now = time.time()
        if self.prev_render_time:
            log.info(now - self.prev_render_time)
        self.prev_render_time = now
        if obz is not None:
            self.pyglet_queue.put(obz['cameras'])


def render_cameras(render_queue, cameras):
    import pyglet
    from pyglet.gl import GLubyte

    widths = []
    heights = []
    for camera in cameras:
        widths += [camera['capture_width']]
        heights += [camera['capture_height']]

    width = max(widths) * 2  # image and depths
    height = sum(heights)
    window = pyglet.window.Window(width, height)
    fps_display = pyglet.clock.ClockDisplay() if DRAW_FPS else None

    @window.event
    def on_draw():
        window.clear()
        cams = render_queue.get(block=True)
        channels = 3
        bytes_per_channel = 1
        for cam_idx, cam in enumerate(cams):
            img_raw = cam['img_raw'] if 'img_raw' in cam else cam['image']
            img_data = np.copy(img_raw)
            depth_data = np.ascontiguousarray(utils.depth_heatmap(np.copy(cam['depth'])))
            img_data.shape = -1
            depth_data.shape = -1
            img_texture = (GLubyte * img_data.size)(*img_data.astype('uint8'))
            depth_texture = (GLubyte * depth_data.size)(*depth_data.astype('uint8'))
            image = pyglet.image.ImageData(
                cam['capture_width'],
                cam['capture_height'],
                'RGB',
                img_texture,
                pitch= -1 * cam['capture_width'] * channels * bytes_per_channel)
            depth = pyglet.image.ImageData(
                cam['capture_width'],
                cam['capture_height'],
                'RGB',
                depth_texture,
                pitch= -1 * cam['capture_width'] * channels * bytes_per_channel)
            if image is not None:
                image.blit(0, cam_idx * cam['capture_height'])
            if depth is not None:
                depth.blit(cam['capture_width'], cam_idx * cam['capture_height'])
        if DRAW_FPS:
            fps_display.draw()

    while True:
        pyglet.clock.tick()

        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event('on_draw')
            window.flip()
