from typing import List

import numpy as np


import time

from util.sampler import Sampler
import utils
import config as c


class EpisodeScore(object):
    total = 0
    gforce_penalty: float = 0
    max_gforce: float = 0
    max_kph: float = 0
    avg_kph: float = 0
    lane_deviation_penalty: float = 0
    time_penalty: float = 0
    progress_reward: float = 0
    speed_reward: float = 0
    progress: float = 0
    prev_progress: float = 0
    got_stuck = False
    wrong_way = False
    start_time: float = 0
    end_time: float = 0
    episode_time: float = 0  # seconds
    num_steps: int = 0
    cm_along_route: float = 0
    route_length_cm: float = 0

    def __init__(self):
        self.start_time = time.time()
        self.end_time = 0
        self.episode_time = 0
        self.cm_per_second_sampler = Sampler()
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
    num_episodes: int = 0
    num_steps: int = 0
    max_gforce: float = 0
    max_kph: float = 0
    avg_kph: float = 0
    trip_speed_kph: float = 0

    def update(self, episode_scores:List[EpisodeScore]):
        totals = [e.total for e in episode_scores]
        self.median = float(np.median(totals))
        self.average = float(np.mean(totals))
        self.high = float(max(totals))
        self.low = float(min(totals))
        self.std = float(np.std(totals))
        self.num_episodes = len(episode_scores)
        cm_per_second_means = \
            [e.cm_per_second_sampler.mean() for e in episode_scores]
        cm_per_second_avg = float(np.mean(cm_per_second_means))
        self.avg_kph = cm_per_second_avg * c.CMPS_TO_KPH
        trip_cm_per_second = float(np.mean(
            [e.cm_along_route / e.episode_time for e in episode_scores]))
        self.trip_speed_kph = trip_cm_per_second * c.CMPS_TO_KPH


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
