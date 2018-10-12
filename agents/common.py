from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import glob
import os

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import sys

import logs
import config as c


log = logs.get_log(__name__)


def get_throttle(actual_speed, target_speed):
    # TODO: Use a PID here
    desired_throttle = abs(target_speed / max(actual_speed, 1e-3))
    desired_throttle = min(max(desired_throttle, 0.), 1.)
    return desired_throttle
