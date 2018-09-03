from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (int, open, round,
                             str)
import logs
log = logs.get_log(__name__)

class RewardWeighting(object):
    def __init__(self, progress, gforce, lane_deviation, total_time, speed):
        # Progress and time were used in DeepDrive-v0 (2.0) - keeping for now in case we want to use again
        self.progress_weight = progress
        self.gforce_weight = gforce
        self.lane_deviation_weight = lane_deviation
        self.time_weight = total_time
        self.speed_weight = speed

    @staticmethod
    def combine(progress_reward, gforce_penalty, lane_deviation_penalty, time_penalty, speed):
        return progress_reward \
               - gforce_penalty \
               - lane_deviation_penalty \
               - time_penalty \
               + speed