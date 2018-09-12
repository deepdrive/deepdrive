from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from future.builtins import (int, open, round,
                             str)

import deepdrive_simulation

import logs
import config as c
log = logs.get_log(__name__)


def sun_speed(speed):
    """Seconds of sun rotation per second of simulation time"""
    assert np.issubdtype(type(speed), np.integer)
    assert 0 < speed <= 10 ** 6
    deepdrive_simulation.set_sun_simulation_speed(int(speed))


def randomize_sun_speed():
    sun_speeds = np.array([[1,        0.5,  'Earth (1 second per second)'],
                           [1000,     0.3,  'Rolling shadows 1'],
                           [2000,     0.11, 'Rolling shadows 2'],
                           [10 ** 4,  0.05, 'Time lapse'],
                           [10 ** 5,  0.02, 'Rave'],
                           [10 ** 6,  0.02, 'Strobe']])

    probs = list(sun_speeds[:, 1].astype(np.float32))
    speeds = sun_speeds[:, 0].astype(np.int)
    rand_sun_speed = c.rng.choice(speeds, 1, probs)[0]
    sun_speed(rand_sun_speed)


def randomize_month():
    deepdrive_simulation.set_date_and_time(month=c.rng.choice(list(range(1, 13))))
