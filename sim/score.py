from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from typing import List

import numpy as np
from future.builtins import (int, open, round,
                             str)

import time

from util.sampler import Sampler
import utils


class EpisodeScore(object):
    total = 0
    gforce_penalty = 0
    lane_deviation_penalty = 0
    time_penalty = 0
    progress_reward = 0
    speed_reward = 0
    progress = 0
    prev_progress = 0
    got_stuck = False
    wrong_way = False
    start_time = 0
    end_time = 0
    episode_time = 0

    def __init__(self):
        self.start_time = time.time()
        self.end_time = 0
        self.episode_time = 0
        self.speed_sampler = Sampler()
        self.gforce_sampler = Sampler()

    def serialize(self):
        defaults = utils.obj2dict(EpisodeScore)
        prop_names = defaults.keys()
        ret = {}
        for k in prop_names:
            ret[k] = getattr(self, k, defaults[k])
        return ret


class TotalScore(object):
    median: float
    average: float
    high: float
    low: float
    std: float

    def __init__(self, episode_scores):
        self.median = float(np.median(episode_scores))
        self.average = float(np.mean(episode_scores))
        self.high = float(max(episode_scores))
        self.low = float(min(episode_scores))
        self.std = float(np.std(episode_scores))


def main():
    score = EpisodeScore()
    from utils import obj2dict
    now = time.time()
    ser = obj2dict(score)
    took = time.time() - now
    print('took %f s' % took)
    print(ser)


if __name__ == '__main__':
    main()
