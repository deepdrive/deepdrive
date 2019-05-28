from random import randint
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
    Simple data structure that maintains a fixed sized random sample of an
    input stream for computing a running median, etc...

    i.e. reservoir sampling
    """
    def __init__(self, maxlen=10000, sampler_type=SamplerType.ALL_TIME):
        """
        maxlen: Number of samples to maintain
        """
        self.q = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.is_full = False
        self.num_samples = 0
        self.type = sampler_type
        self.max:float = float('-inf')
        self.min:float = float('inf')
        self.total:float = 0
        self._mean:float = 0

    def sample(self, x):
        self.num_samples += 1
        if self.type is SamplerType.RECENT or not self.is_full:
            self.q.append(x)
        else:
            if self.type is not SamplerType.ALL_TIME:
                raise RuntimeError('Expected sampling type to be '
                                   'across all time, but was %s' % str(self.type))
            # Once we have filled the buffer, try to keep an equal distribution of old and new samples by
            # decaying the replacement probability.
            should_replace = self.is_full and self.uniform_random_chance()
            if should_replace:
                index = randint(0, self.maxlen - 1)
                del self.q[index]
                self.q.append(x)
        if not self.is_full:
            self.is_full = len(self.q) == self.maxlen
        self.max = max(self.max, x)
        self.min = min(self.min, x)
        self.total += x

    def uniform_random_chance(self):
        return c.rng.rand() < len(self.q) / self.num_samples

    def mean(self):
        if self.type is SamplerType.ALL_TIME:
            return self.total / self.num_samples
        else:
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
        # s.sample(randint(0, 100))
        s.sample(i)
        mean = s.mean()
        median = s.median()
        change = s.change(steps=1)
    print(i, 'mean', mean, 'median', median, 'change',
          change, 'len', len(s.q), s.q, 'max', s.max, 'min', s.min)


def test_all_time():
    s = Sampler(maxlen=25, sampler_type=SamplerType.ALL_TIME)
    for i in range(10000):
        # s.sample(randint(0, 100))
        s.sample(i)
        mean = s.mean()
        median = s.median()
    print(i, 'mean', mean, 'median', median, 'len', len(s.q), s.q,
          'max', s.max, 'min', s.min)


if __name__ == '__main__':
    main()
