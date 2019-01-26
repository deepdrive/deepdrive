from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import random
import time
from random import randint

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import math
from enum import Enum


from collections import deque
import numpy as np

import config as c
import logs
log = logs.get_log(__name__)


class SamplerType(Enum):
    ALL_TIME = 1
    RECENT = 2


class Sampler(object):
    """
    Simple data structure that maintains a fixed sized random sample of an input stream for computing mean, etc...
    """
    def __init__(self, hz=5, maxlen=10000, sampler_type=SamplerType.ALL_TIME):
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
        self.type = sampler_type

    def sample(self, x):
        self.num_samples += 1
        if self.type is SamplerType.RECENT or not self.is_full:
            self.q.append(x)
        else:
            if self.type is not SamplerType.ALL_TIME:
                raise RuntimeError('Expected sampling type to be across all time, but was %s' % str(self.type))
            # Once we have filled the buffer, try to keep an equal distribution of old and new samples by
            # decaying the replacement probability.
            should_replace = self.is_full and c.rng.rand() < len(self.q) / self.num_samples
            if should_replace:
                index = randint(0, self.maxlen - 1)
                del self.q[index]
                self.q.append(x)
        if not self.is_full:
            self.is_full = len(self.q) == self.maxlen

    def mean(self):
        return np.mean(self.q)

    def median(self):
        return np.median(self.q)

    def change(self, steps):
        if self.type is SamplerType.ALL_TIME:
            log.warn('Times between samples will grow in all time mode, are you sure you don\'t want recent mode?')
        if len(self.q) < steps:
            return None
        else:
            return self.q[-1] - self.q[-steps]


def main():
    test_all_time()
    test_recent()


def test_recent():
    s = Sampler(maxlen=50, sampler_type=SamplerType.RECENT)
    for i in range(10000):
        time.sleep(s.period)
        # s.sample(randint(0, 100))
        s.sample(i)
        mean = s.mean()
        median = s.median()
        change = s.change(steps=1)
        print(i, 'mean', mean, 'median', median, 'change', change, 'len', len(s.q), s.q)


def test_all_time():
    s = Sampler(maxlen=25, sampler_type=SamplerType.ALL_TIME)
    for i in range(10000):
        time.sleep(s.period)
        # s.sample(randint(0, 100))
        s.sample(i)
        mean = s.mean()
        median = s.median()
        print(i, 'mean', mean, 'median', median, 'len', len(s.q), s.q)


if __name__ == '__main__':
    main()
