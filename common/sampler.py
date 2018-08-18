from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import time
from random import randint

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

from collections import deque
import numpy as np

class Sampler(object):
    """Simple data structure that maintains a fixed sized random sample of an input stream for computing mean, etc..."""
    def __init__(self, hz=5, maxlen=100):
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

    def _add(self, x, index=None):
        if index is None:
            self.q.append(x)
        else:
            self.q.insert(index, x)
        self.is_full = len(self.q) == self.maxlen

    def sample(self, x):
        if self.last_sample_time is None:
            self._add(x)
        else:
            now = time.time()
            delta = now - self.last_sample_time()
            if delta >= self.period:
                if not self.is_full:
                    self._add(x)
                else:
                    index = randint(0, self.maxlen)
                    self._add(x, index)

    def mean(self):
        return running_mean(self.q, len(self.q))[0]


def running_mean(x, N):
    # https://stackoverflow.com/a/27681394/134077
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


if __name__ == '__main__':
    s = Sampler(hz=10**10, maxlen=1000)
    for i in range(1000):
        start = time.time()
        s.sample(randint(0, 100))
        mean = s.mean()
        print(mean, 'took', time.time() - start, 's')
