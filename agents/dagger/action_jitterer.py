from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import glob
import os

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
from enum import Enum

import config as c
import logs

log = logs.get_log(__name__)


class JitterState(Enum):
    MAINTAIN = 1
    SWITCH_TO_NONRAND = 2
    SWITCH_TO_RAND = 3


class ActionJitterer(object):
    def __init__(self):
        """
        Jitters state between m random then n non-random steps -
            where m and n are set to different values to increase sample diversity
        """
        self.rand_step = None
        self.nonrand_step = None
        self.rand_total = None
        self.nonrand_total = None
        self.seq_total = None
        self.perform_random_actions = None
        self.reset()

    def advance(self):
        if self.perform_random_actions:
            if self.rand_step < self.rand_total:
                self.rand_step += 1
                ret = JitterState.MAINTAIN
            else:
                # Done with random actions, switch to non-random
                self.perform_random_actions = False
                ret = JitterState.SWITCH_TO_NONRAND
        else:
            if self.nonrand_step < self.nonrand_total:
                self.nonrand_step += 1
                ret = JitterState.MAINTAIN
            else:
                # We are done with the sequence
                self.reset()
                if self.perform_random_actions:
                    ret = JitterState.SWITCH_TO_RAND
                else:
                    ret = JitterState.MAINTAIN  # Skipping random actions this sequence
        return ret

    def reset(self):
        self.rand_step = 0
        self.nonrand_step = 0
        rand = c.rng.rand()
        if rand < 0.50:
            self.rand_total = 0
            self.nonrand_total = 10
        elif rand < 0.67:
            self.rand_total = 4
            self.nonrand_total = 5
        elif rand < 0.85:
            self.rand_total = 8
            self.nonrand_total = 10
        elif rand < 0.95:
            self.rand_total = 12
            self.nonrand_total = 15
        else:
            self.rand_total = 24
            self.nonrand_total = 30
        log.debug('random action total %r, non-random total %r', self.rand_total,
                  self.nonrand_total)
        self.seq_total = self.rand_total + self.nonrand_total
        self.perform_random_actions = self.rand_total != 0


