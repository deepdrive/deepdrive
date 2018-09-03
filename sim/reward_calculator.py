from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from future.builtins import (int, open, round,
                             str)
import logs
log = logs.get_log(__name__)

class RewardCalculator(object):
    @staticmethod
    def clip(reward):
        # time_passed not parameter in order to set hard limits on reward magnitude
        return min(max(reward, -1e2), 1e2)

    @staticmethod
    def get_lane_deviation_penalty(lane_deviation, time_passed):
        lane_deviation_penalty = 0
        if lane_deviation < 0:
            raise ValueError('Lane deviation should be positive')
        if time_passed is not None and lane_deviation > 200:  # Tuned for Canyons spline - change for future maps
            lane_deviation_coeff = 0.1
            lane_deviation_penalty = lane_deviation_coeff * time_passed * lane_deviation ** 2 / 100.
        log.debug('distance_to_center_of_lane %r', lane_deviation)
        lane_deviation_penalty = RewardCalculator.clip(lane_deviation_penalty)
        return lane_deviation_penalty

    @staticmethod
    def get_gforce_penalty(gforces, time_passed):
        log.debug('gforces %r', gforces)
        gforce_penalty = 0
        if gforces < 0:
            raise ValueError('G-Force should be positive')
        if gforces > 0.1:
            # Based on regression model on page 47 - can achieve 92/100 comfort with 0.15 x and y acceleration
            # http://www.diva-portal.org/smash/get/diva2:950643/FULLTEXT01.pdf
            # Perhaps we should combine x, 0.15 and y 0.15,
            time_weighted_gs = time_passed * gforces
            time_weighted_gs = min(time_weighted_gs,
                                   5)  # Don't allow a large frame skip to ruin the approximation
            balance_coeff = 24  # 24 meters of reward every second you do this
            gforce_penalty = time_weighted_gs * balance_coeff
            log.debug('accumulated_gforce %r', time_weighted_gs)
            log.debug('gforce_penalty %r', gforce_penalty)
        gforce_penalty = RewardCalculator.clip(gforce_penalty)
        return gforce_penalty

    @staticmethod
    def get_progress_and_speed_reward(progress, time_passed):
        if not time_passed:
            progress = speed_reward = 0
        else:
            progress = progress / 100.  # cm=>meters
            speed = progress / time_passed
            if speed < -400:
                # Lap completed
                # TODO: Read the lap length on reset and
                log.debug('assuming lap complete, progress zero')
                progress = speed = 0

            # Square speed to outweigh advantage of longer lap time from going slower
            speed_reward = np.sign(speed) * speed ** 2 * time_passed  # i.e. sign(progress) * progress ** 2 / time_passed

        progress_balance_coeff = 1.0
        progress_reward = progress * progress_balance_coeff

        speed_balance_coeff = 0.15
        speed_reward *= speed_balance_coeff

        progress_reward = RewardCalculator.clip(progress_reward)
        speed_reward = RewardCalculator.clip(speed_reward)
        return progress_reward, speed_reward
