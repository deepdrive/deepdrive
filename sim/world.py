from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from future.builtins import (int, open, round,
                             str)

import deepdrive_simulation

import logs
import config as c
from sim.uepy_client import rpc

log = logs.get_log(__name__)


def sun_speed(speed):
    """Seconds of sun rotation per second of simulation time"""
    if not np.issubdtype(type(speed), np.integer):
        raise ValueError('Expected integer sun speed, got %r', type(speed))
    if not (0 < speed <= 10 ** 6):
        raise ValueError('Sun speed not between zero and 10M, was %r', speed)

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


def randomize_sun_month():
    deepdrive_simulation.set_date_and_time(month=c.rng.choice(list(range(1, 13))))


def reset(enable_traffic=False):
    return rpc('reset', enable_traffic=enable_traffic)


def set_ego_mph(min_mph, max_mph):
    return rpc('set_ego_mph', min_mph=min_mph, max_mph=max_mph)


def get_agent_positions() -> List[list]:
    """
    :return: Positions of non-ego agents
    """
    start = time.time()
    ret = rpc('get_agent_positions')
    t = time.time() - start
    log.debug('get_agent_positions() took %rs' % t)
    return ret


def get_agents() -> List[dict]:
    """
    :return: Detailed list of all agents in the scene
    See: https://gist.github.com/crizCraig/b5f88911cae9dc346bf805498f31ec3f
    """
    start = time.time()
    ret = rpc('get_agents')
    t = time.time() - start
    log.debug('get_agents() took %rs' % t)
    return ret


def get_observation() -> dict:
    """
    This is a combined call to the UEPy server to get all the data needed
    for the step. The UEPy API server can only return one response per frame,
    so at 60FPS latency will be minimum 17ms
    :return: Info appropriate to return every step. 
    """""
    ret = rpc('get_observation')
    return ret
