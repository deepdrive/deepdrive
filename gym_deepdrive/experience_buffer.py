from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

from collections import deque


class ExperienceBuffer(object):
    def __init__(self, step_seconds=0.25, seconds_to_keep=2, fade_fn=None):
        self.step_seconds = step_seconds  # type: float
        self.seconds_to_keep = seconds_to_keep  # type: int

        self.max_length = int(self.seconds_to_keep / self.step_seconds)
        self.buffer = deque(maxlen=self.max_length)
        self.last_capture_time = None
        self.fade_fn = fade_fn or (lambda j: 2 * j)

        self.fade_length = 0
        while self.fade_fn(self.fade_length) < self.max_length:
            self.fade_length += 1

    def maybe_add(self, x, t):
        if self.last_capture_time is None or t > (self.last_capture_time + self.step_seconds):
            self.buffer.append((x, t))

    def size(self):
        return len(self.buffer)

    def __len__(self):
        return self.size()

    def get_fading(self):
        if self.size() > 0 and self.size() == self.max_length:
            i = self.size() - 1
            ret = []
            while i >= 0:
                ret.append(self.buffer[i])
                i -= self.fade_fn(i)
        else:
            ret = None

        if len(ret) < self.fade_length:
            # Don't return a partial buffer
            ret = None

        return ret


