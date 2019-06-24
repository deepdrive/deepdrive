from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from itertools import product

from future.builtins import (int, open, round,
                             str)

import math

import numpy as np
import logs
log = logs.get_log(__name__)


class Action(object):
    STEERING_INDEX = 0
    THROTTLE_INDEX = 1
    BRAKE_INDEX = 2
    HANDBRAKE_INDEX = 3
    HAS_CONTROL_INDEX = 4

    STEERING_MIN, STEERING_MAX = -1, 1
    THROTTLE_MIN, THROTTLE_MAX = -1, 1
    BRAKE_MIN, BRAKE_MAX = 0, 1
    HANDBRAKE_MIN, HANDBRAKE_MAX = 0, 1

    def __init__(self, steering=0, throttle=0, brake=0, handbrake=0,
                 has_control=True):
        self.steering = steering
        self.throttle = throttle
        self.brake = brake
        self.handbrake = handbrake
        self.has_control = has_control

    def clip(self):
        self.steering  = min(max(self.steering,  self.STEERING_MIN),  self.STEERING_MAX)
        self.throttle  = min(max(self.throttle,  self.THROTTLE_MIN),  self.THROTTLE_MAX)
        self.brake     = min(max(self.brake,     self.BRAKE_MIN),     self.BRAKE_MAX)
        self.handbrake = min(max(self.handbrake, self.HANDBRAKE_MIN), self.HANDBRAKE_MAX)

    def as_gym(self):
        ret = gym_action(steering=self.steering, throttle=self.throttle,
                         brake=self.brake, handbrake=self.handbrake,
                         has_control=self.has_control)
        return ret

    def serialize(self):
        ret = [self.steering, self.throttle, self.brake, self.handbrake,
               self.has_control]
        return ret


    @classmethod
    def from_gym(cls, action):
        has_control = True
        if len(action) > 4:
            if isinstance(action[4], list):
                has_control = action[4][0]
            else:
                has_control = action[cls.HAS_CONTROL_INDEX]
        handbrake = action[cls.HANDBRAKE_INDEX][0]
        if handbrake <= 0 or math.isnan(handbrake):
            handbrake = 0
        else:
            handbrake = 1
        ret = cls(steering=action[cls.STEERING_INDEX][0],
                  throttle=action[cls.THROTTLE_INDEX][0],
                  brake=action[cls.BRAKE_INDEX][0],
                  handbrake=handbrake, has_control=has_control)
        return ret


def gym_action(steering=0, throttle=0, brake=0, handbrake=0, has_control=True):
    action = [np.array([steering]),
              np.array([throttle]),
              np.array([brake]),
              np.array([handbrake]),
              has_control]
    return action


class DiscreteActions(object):
    def __init__(self, steer, throttle, brake):
        self.steer = steer
        self.throttle = throttle
        self.brake = brake

        self.product = list(product(steer, throttle, brake))

    def get_components(self, idx):
        steer, throttle, brake = self.product[idx]
        return steer, throttle, brake