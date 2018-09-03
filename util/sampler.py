from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import random
import time
from random import randint

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

from collections import deque
import numpy as np

class Sampler(object):
    """
    Simple data structure that maintains a fixed sized random sample of an input stream for computing mean, etc...
    """
    def __init__(self, hz=5, maxlen=10000):
        """
        hz: Frequency at which to collect samples
        maxlen: Number of samples to maintain
        """
        self.hz = hz
        self.period = 1. / hz
        self.q = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.last_sample_time = None
        self.is_full = False
        self.num_samples = 0

    def _add(self, x):
        self.num_samples += 1
        if not self.is_full:
            self.q.append(x)
        else:
            should_replace = self.is_full and random.random() < (1. / self.num_samples ** 0.85)
            if should_replace:
                index = randint(0, self.maxlen-1)
                del self.q[index]
                self.q.append(x)
        if not self.is_full:
            self.is_full = len(self.q) == self.maxlen

    def sample(self, x):
        if self.last_sample_time is None:
            self._add(x)
        else:
            now = time.time()
            delta = now - self.last_sample_time()
            if delta >= self.period:
                self._add(x)

    def mean(self):
        return np.mean(self.q)


if __name__ == '__main__':
    s = Sampler(hz=10**10, maxlen=5)
    for i in range(10000):
        start = time.time()
        # s.sample(randint(0, 100))
        s.sample(i)
        mean = s.mean()
        print(mean, 'took', time.time() - start, 's', s.q)
