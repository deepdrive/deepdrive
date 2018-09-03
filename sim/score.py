from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (int, open, round,
                             str)

import time

from util.sampler import Sampler


class Score(object):
    total = 0
    gforce_penalty = 0
    lane_deviation_penalty = 0
    time_penalty = 0
    progress_reward = 0
    speed_reward = 0
    progress = 0
    got_stuck = False
    wrong_way = False

    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        self.episode_time = 0
        self.speed_sampler = Sampler()