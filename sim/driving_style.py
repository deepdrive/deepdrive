from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (int, open, round,
                             str)


from enum import Enum

import logs
log = logs.get_log(__name__)

class DrivingStyle(Enum):
    """Idea: Adjust these weights dynamically to produce a sort of curriculum where speed is learned first,
    then lane, then gforce. Also, need to record unweighted score components in physical units (m, m/s^2, etc...)
    so that scores can be compared across different weightings and environments.

    To adjust dynamically, the reward weighting should be changed per episode, or a horizon based on discount factor,
    in order to achieve a desired reward component balance.

    So say we wanted speed to be the majority of the reward received, i.e. 50%. We would look at the share made up by
    speed in the return for an episode (i.e. trip or lap for driving). If it's 25% of the absolute reward
    (summing positive and abs(negative) rewards), then we double a "curriculum coefficient" or CC for speed. These curriculum
    coefficients then get normalized so the final aggregate reward maintains the same scale as before.

    Then, as speed gets closer to 50% of the reward, the smaller components of the reward will begin to get weighted
    more heavily. If speed becomes more than 50% of the reward, then its CC will shrink and allow learning how to achieve
    other objectives.

    Why do this?
    Optimization should find the best way to squeeze out all the juice from the reward, right? Well, maybe, but
    I'm finding the scale and *order* to be very important in practice. In particular, lane deviation grows like crazy
    once you are out of the lane, regardless of the weight. So if speed is not learned first, our agent just decides
    to not move. Also, g-force penalties counter initial acceleration required to get speed, so we end up needing to
    weight g-force too small or too large with respect to speed over the long term.

    The above curriculum approach aims to fix these things by targeting a certain balance of objectives over the
    long-term, rather than the short-term, while adjusting short-term curriculum weights in order to get there. Yes,
    it does feel like the model should take care of this, but it's only optimized for the expected aggregate reward
    across all the objectives. Perhaps inputting the different components running averages or real-time values to
    a recurrent part of the model would allow it to balance the objectives through SGD rather than the above
    simple linear tweaking.

    (looks like EPG is a nicer formulation of this https://blog.openai.com/evolved-policy-gradients/)
    (Now looks like RUDDER is another principled step in this direction https://arxiv.org/abs/1806.07857)

    - After some experimentation, seems like we may not need this yet. Observation normalization was causing the
    motivating problem by learning too slow. Optimization does find a way. I think distributional RL may be helpful here
    especially if we can get dimensions for all the compoenents of the reward. Also a novelty bonus on
    (observation,action) or (game-state,action) would be helpful most likely to avoid local minima.
    """
    __order__ = 'CRUISING NORMAL LATE EMERGENCY CHASE'
    # TODO: Possibly assign function rather than just weights
    CRUISING   = RewardWeighting(speed=0.5, progress=0.0, gforce=2.00, lane_deviation=1.50, total_time=0.0)
    NORMAL     = RewardWeighting(speed=1.0, progress=0.0, gforce=0.00, lane_deviation=0.10, total_time=0.0)
    LATE       = RewardWeighting(speed=2.0, progress=0.0, gforce=0.50, lane_deviation=0.50, total_time=0.0)
    EMERGENCY  = RewardWeighting(speed=2.0, progress=0.0, gforce=0.75, lane_deviation=0.75, total_time=0.0)
    CHASE      = RewardWeighting(speed=2.0, progress=0.0, gforce=0.00, lane_deviation=0.00, total_time=0.0)
    STEER_ONLY = RewardWeighting(speed=1.0, progress=0.0, gforce=0.00, lane_deviation=0.00, total_time=0.0)


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