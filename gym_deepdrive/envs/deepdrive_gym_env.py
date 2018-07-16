from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
import math
import csv
import subprocess
import deepdrive_client
import deepdrive_capture
import os
import random
import sys
import time
from collections import deque, OrderedDict
from multiprocessing import Process, Queue
from subprocess import Popen
import pkg_resources
from distutils.version import LooseVersion as semvar
from itertools import product
from enum import Enum

import arrow
import gym
import numpy as np
from GPUtil import GPUtil
from boto.s3.connection import S3Connection
from gym import spaces
from gym.utils import seeding
try:
    import pyglet
    from pyglet.gl import GLubyte
except:
    pyglet = None

import config as c
import logs
import utils
from utils import obj2dict, download
from dashboard import dashboard_fn, DashboardPub

log = logs.get_log(__name__)
SPEED_LIMIT_KPH = 64.
HEAD_START_TIME = 0


class Score(object):
    total = 0
    gforce_penalty = 0
    lane_deviation_penalty = 0
    time_penalty = 0
    progress_reward = 0
    speed_reward = 0
    progess = 0
    got_stuck = False
    wrong_way = False

    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        self.episode_time = 0


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

    def __init__(self, steering=0, throttle=0, brake=0, handbrake=0, has_control=True):
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
        ret = gym_action(steering=self.steering, throttle=self.throttle, brake=self.brake,
                         handbrake=self.handbrake, has_control=self.has_control)
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


class Camera(object):
    def __init__(self, name, field_of_view, capture_width, capture_height, relative_position, relative_rotation):
        self.name = name
        self.field_of_view = field_of_view
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.relative_position = relative_position
        self.relative_rotation = relative_rotation
        self.connection_id = None


class DiscreteActions(object):
    def __init__(self, steer, throttle, brake):
        self.steer = steer
        self.throttle = throttle
        self.brake = brake

        self.product = list(product(steer, throttle, brake))

    def get_components(self, idx):
        steer, throttle, brake = self.product[idx]
        return steer, throttle, brake


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

    - After some experimentation, seems like we may not need this yet. Observation normalization was causing the
    motivating problem by learning too slow. Optimization does find a way. I think distributional RL may be helpful here
    especially if we can get dimensions for all the compoenents of the reward. Also a novelty bonus on
    (observation,action) or (game-state,action) would be helpful most likely to avoid local minima.
    """
    __order__ = 'CRUISING NORMAL LATE EMERGENCY CHASE'
    # TODO: Possibly assign function rather than just weights
    CRUISING   = RewardWeighting(speed=0.5, progress=0.0, gforce=2.00, lane_deviation=1.50, total_time=0.0)
    NORMAL     = RewardWeighting(speed=1.0, progress=0.0, gforce=0.10, lane_deviation=0.10, total_time=0.0)
    LATE       = RewardWeighting(speed=2.0, progress=0.0, gforce=0.50, lane_deviation=0.50, total_time=0.0)
    EMERGENCY  = RewardWeighting(speed=2.0, progress=0.0, gforce=0.75, lane_deviation=0.75, total_time=0.0)
    CHASE      = RewardWeighting(speed=2.0, progress=0.0, gforce=0.00, lane_deviation=0.00, total_time=0.0)
    STEER_ONLY = RewardWeighting(speed=1.0, progress=0.0, gforce=0.00, lane_deviation=0.00, total_time=0.0)


default_cam = Camera(**c.DEFAULT_CAM)  # TODO: Switch camera dicts to this object


def gym_action(steering=0, throttle=0, brake=0, handbrake=0, has_control=True):
    action = [np.array([steering]),
              np.array([throttle]),
              np.array([brake]),
              np.array([handbrake]),
              has_control]
    return action


# noinspection PyMethodMayBeStatic
class DeepDriveEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.is_discrete = None
        self.is_sync = None
        self.sync_step_time = None
        self.discrete_actions = None
        self.preprocess_with_tensorflow = None
        self.sess = None
        self.prev_observation = None
        self.start_time = time.time()
        self.step_num = 0
        self.prev_step_time = None
        self.display_stats = OrderedDict()
        self.display_stats['g-forces']                = {'total': 0, 'value': 0, 'ymin': 0,      'ymax': 3,    'units': 'g'}
        self.display_stats['gforce penalty']          = {'total': 0, 'value': 0, 'ymin': -500,   'ymax': 0,    'units': ''}
        self.display_stats['lane deviation penalty']  = {'total': 0, 'value': 0, 'ymin': -100,   'ymax': 0,    'units': ''}
        self.display_stats['lap progress']            = {'total': 0, 'value': 0, 'ymin': 0,      'ymax': 100,  'units': '%'}
        self.display_stats['speed reward']            = {'total': 0, 'value': 0, 'ymin': 0,      'ymax': 5000, 'units': ''}
        self.display_stats['episode #']               = {'total': 0, 'value': 0, 'ymin': 0,      'ymax': 5,    'units': ''}
        self.display_stats['time']                    = {'total': 0, 'value': 0, 'ymin': 0,      'ymax': 500,  'units': 's'}
        self.display_stats['episode score']           = {'total': 0, 'value': 0, 'ymin': -500,   'ymax': 4000, 'units': ''}
        self.dashboard_process = None
        self.dashboard_pub = None
        self.should_exit = False
        self.sim_process = None
        self.client_id = None
        self.has_control = None
        self.cameras = None
        self.use_sim_start_command = None
        self.connection_props = None
        self.should_render = None  # type: bool
        self.pyglet_render = False  # type: bool
        self.pyglet_image = None
        self.pyglet_process = None
        self.pyglet_queue = None
        self.ep_time_balance_coeff = 10.  # type: float
        self.previous_action_time = None  # type: time.time
        self.fps = None  # type: int
        self.period = None  # type: float
        self.experiment = None  # type: str
        self.driving_style = None  # type: DrivingStyle`
        self.reset_returns_zero = None  # type: bool
        self.started_driving_wrong_way_time = None  # type: bool
        self.previous_distance_along_route = None  # type: bool

        if not c.REUSE_OPEN_SIM:
            utils.download_sim()

        self.client_version = pkg_resources.get_distribution("deepdrive").version
        # TODO: Check with connection version

        # collision detection  # TODO: Remove in favor of in-game detection
        self.set_forward_progress()

        self.distance_along_route = 0
        self.start_distance_along_route = 0

        # reward
        self.score = Score()

        # laps
        self.lap_number = None
        self.prev_lap_score = 0
        self.total_laps = 0

        # benchmarking - carries over across resets
        self.should_benchmark = False
        self.trial_scores = []

        try:
            self.git_commit = str(utils.run_command('git rev-parse --short HEAD')[0])
        except:
            self.git_commit = 'n/a'
            log.warning('Could not get git commit for associating benchmark results with code state')

        try:
            self.git_diff = utils.run_command('git diff')[0]
        except:
            self.git_diff = None
            log.warning('Could not get git diff for associating benchmark results with code state')

        if c.TENSORFLOW_AVAILABLE:
            import tensorflow as tf
            self.tensorboard_writer = tf.summary.FileWriter(
                os.path.join(c.TENSORFLOW_OUT_DIR, 'env', c.DATE_STR))
        else:
            self.tensorboard_writer = None

    def open_sim(self):
        self._kill_competing_procs()
        if c.REUSE_OPEN_SIM:
            return
        if self.use_sim_start_command:
            log.info('Starting simulator with command %s - this will take a few seconds.',
                     c.SIM_START_COMMAND)

            self.sim_process = Popen(c.SIM_START_COMMAND)

            import win32gui
            import win32process

            def get_hwnds_for_pid(pid):
                def callback(hwnd, _hwnds):
                    if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                        _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                        if found_pid == pid:
                            _hwnds.append(hwnd)
                    return True

                hwnds = []
                win32gui.EnumWindows(callback, hwnds)
                return hwnds

            focused = False
            while not focused:
                time.sleep(1)
                dd_hwnds = get_hwnds_for_pid(self.sim_process.pid)
                if not dd_hwnds:
                    log.info('No windows found, waiting')
                else:
                    try:
                        win32gui.SetForegroundWindow(dd_hwnds[0])
                        focused = True
                    except:
                        log.info('Window not ready, waiting')

            pass
        else:
            log.info('Starting simulator at %s (takes a few seconds the first time).', utils.get_sim_bin_path())
            self.sim_process = Popen([utils.get_sim_bin_path()])

    def close_sim(self):
        log.info('Closing sim')
        if self.sim_process is not None:
            self.sim_process.kill()

    def _kill_competing_procs(self):
        # TODO: Allow for many environments on the same machine by using registry DB for this and sharedmem
        path = utils.get_sim_bin_path()
        if path is None:
            return
        process_name = os.path.basename(utils.get_sim_bin_path())
        if c.IS_WINDOWS:
            cmd = 'taskkill /IM %s /F' % process_name
        elif c.IS_LINUX or c.IS_MAC:
            cmd = 'pkill %s' % process_name
        else:
            raise NotImplementedError('OS not supported')
        utils.run_command(cmd, verbose=False, throw=False, print_errors=False)
        time.sleep(1)  # TODO: Don't rely on time for shared mem to go away, we should have a unique name on startup.

    def set_use_sim_start_command(self, use_sim_start_command):
        self.use_sim_start_command = use_sim_start_command

    def init_benchmarking(self):
        self.should_benchmark = True
        os.makedirs(c.RESULTS_DIR, exist_ok=True)

    def init_pyglet(self, cameras):
        q = Queue(maxsize=1)
        p = Process(target=render_cameras, args=(q, cameras))
        p.start()
        self.pyglet_process = p
        self.pyglet_queue = q

    def start_dashboard(self):
        if utils.is_debugging():
            # TODO: Deal with plot UI not being in the main thread somehow - (move to Unreal HUD)
            log.warning('Dashboard not supported in debug mode')
            return
        elif utils.is_docker():
            # TODO: Move dashboard stats to Unreal / Tensorboard where appropriate
            log.warning('Dashboard not supported in docker')
            return

        p = Process(target=dashboard_fn)
        p.start()
        self.dashboard_process = p
        self.dashboard_pub = DashboardPub()

    def set_tf_session(self, session):
        self.sess = session

    def step(self, action):
        if self.is_discrete:
            steer, throttle, brake = self.discrete_actions.get_components(action)
            dd_action = Action(steering=steer, throttle=throttle, brake=brake)
        else:
            dd_action = Action.from_gym(action)

        send_control_start = time.time()
        self.send_control(dd_action)
        log.debug('send_control took %fs', time.time() - send_control_start)

        obz = self.get_observation()
        if obz and 'is_game_driving' in obz:
            self.has_control = not obz['is_game_driving']
        now = time.time()

        start_reward_stuff = time.time()
        reward, done = self.get_reward(obz, now)
        self.prev_step_time = now

        if self.dashboard_pub is not None:
            start_dashboard_put = time.time()
            self.dashboard_pub.put(OrderedDict({'display_stats': list(self.display_stats.items()), 'should_stop': False}))
            log.debug('dashboard put took %fs', time.time() - start_dashboard_put)

        self.step_num += 1
        log.debug('reward stuff took %fs', time.time() - start_reward_stuff)

        info = self.get_step_info(done)

        self.regulate_fps()

        if self.should_render:
            self.render()

        return obz, reward, done, info

    def get_step_info(self, done):
        info = {}
        info['score'] = info.get('episode', {})
        info['score']['episode_time'] = self.score.episode_time
        if done:
            info = self.report_score(info)
        return info

    def report_score(self, info):
        self.prev_lap_score = self.score.total
        info['episode'] = episode_info = {}
        episode_info['reward'] = self.score.total
        episode_info['length'] = self.step_num
        episode_info['time'] = self.score.episode_time
        if self.should_benchmark:
            self.log_benchmark_trial()
        else:
            log.info('lap %d complete with score of %f', self.total_laps, self.score.total)
        if self.tensorboard_writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary.value.add(tag="score/total", simple_value=self.score.total)
            summary.value.add(tag="score/episode_length", simple_value=self.step_num)
            summary.value.add(tag="score/episode_time", simple_value=self.score.episode_time)
            summary.value.add(tag="score/speed_reward", simple_value=self.score.speed_reward)

            summary.value.add(tag="score/lane_deviation_penalty", simple_value=self.score.lane_deviation_penalty)
            summary.value.add(tag="score/gforce_penalty", simple_value=self.score.gforce_penalty)
            summary.value.add(tag="score/got_stuck", simple_value=self.score.got_stuck)
            summary.value.add(tag="score/wrong_way", simple_value=self.score.wrong_way)
            summary.value.add(tag="score/time_penalty", simple_value=self.score.time_penalty)

            self.tensorboard_writer.add_summary(summary)
            self.tensorboard_writer.flush()
        return info

    def regulate_fps(self):
        now = time.time()
        if self.previous_action_time:
            delta = now - self.previous_action_time
            if delta < self.period:
                time.sleep(delta)
                self.sync_step_time = self.period
            else:
                fps = 1. / delta
                self.sync_step_time = self.period / 2
                if self.step_num > 5 and fps < self.fps / 2:
                    log.warning('Step %r took %rs - target is %rs', self.step_num, delta, 1 / self.fps)
        self.previous_action_time = now

    def compute_lap_statistics(self, obz):
        start_compute_lap_stats = time.time()
        lap_bonus = 0
        done = False
        if obz:
            lap_number = obz.get('lap_number')
            if lap_number is not None and self.lap_number is not None and self.lap_number < lap_number:
                self.total_laps += 1
                done = True  # One lap per episode
                lap_bonus = 10
                self.log_up_time()
            self.lap_number = lap_number
            log.debug('compute lap stats took %fs', time.time() - start_compute_lap_stats)

        return done, lap_bonus

    def get_reward(self, obz, now):
        start_get_reward = time.time()
        done, lap_bonus = self.compute_lap_statistics(obz)
        reward = 0
        if obz:
            if self.is_sync:
                step_time = self.sync_step_time
            elif self.prev_step_time is not None:
                step_time = now - self.prev_step_time
            else:
                step_time = None

            self.score.episode_time += (step_time or 0)

            if self.score.episode_time < HEAD_START_TIME:
                # Give time to get on track after spawn
                reward = 0
            else:
                gforce_penalty = self.get_gforce_penalty(obz, step_time)
                lane_deviation_penalty = self.get_lane_deviation_penalty(obz, step_time)
                time_penalty = self.get_time_penalty(obz, step_time)
                progress_reward, speed = self.get_progress_and_speed_reward(obz, step_time,
                                                                            gforce_penalty, lane_deviation_penalty)
                reward = self.combine_rewards(progress_reward, gforce_penalty, lane_deviation_penalty,
                                              time_penalty, speed)

            self.score.wrong_way = self.driving_wrong_way()
            if self.score.wrong_way:
                log.warn('Going the wrong way, end of episode')
            if self.is_stuck(obz) or self.score.wrong_way:  # TODO: Done if collision, or near collision
                done = True
                reward -= 10

            self.score.total += reward + lap_bonus
            self.display_stats['time']['value'] = self.score.episode_time
            self.display_stats['time']['total'] = self.score.episode_time
            self.display_stats['episode score']['value'] = self.score.total
            self.display_stats['episode score']['total'] = self.score.total

            log.debug('reward %r', reward)
            log.debug('score %r', self.score.total)

        log.debug('get reward took %fs', time.time() - start_get_reward)

        return reward, done

    def log_up_time(self):
        log.info('up for %r' % arrow.get(time.time()).humanize(other=arrow.get(self.start_time), only_distance=True))

    def get_lane_deviation_penalty(self, obz, time_passed):
        lane_deviation_penalty = 0.
        if 'distance_to_center_of_lane' in obz:
            lane_deviation_penalty = DeepDriveRewardCalculator.get_lane_deviation_penalty(
                obz['distance_to_center_of_lane'], time_passed)

        lane_deviation_penalty *= self.driving_style.value.lane_deviation_weight
        self.score.lane_deviation_penalty += lane_deviation_penalty

        self.display_stats['lane deviation penalty']['value'] = -lane_deviation_penalty
        self.display_stats['lane deviation penalty']['total'] = -self.score.lane_deviation_penalty
        return lane_deviation_penalty

    def get_gforce_penalty(self, obz, time_passed):
        gforce_penalty = 0
        if 'acceleration' in obz:
            if time_passed is not None:
                a = obz['acceleration']
                gforces = np.sqrt(a.dot(a)) / 980  # g = 980 cm/s**2
                self.display_stats['g-forces']['value'] = gforces
                self.display_stats['g-forces']['total'] = gforces
                gforce_penalty = DeepDriveRewardCalculator.get_gforce_penalty(gforces, time_passed)

        gforce_penalty *= self.driving_style.value.gforce_weight
        self.score.gforce_penalty += gforce_penalty

        self.display_stats['gforce penalty']['value'] = -self.score.gforce_penalty
        self.display_stats['gforce penalty']['total'] = -self.score.gforce_penalty
        return gforce_penalty

    def get_progress_and_speed_reward(self, obz, time_passed, gforce_penalty, lane_deviation_penalty):
        progress_reward = speed_reward = 0
        if 'distance_along_route' in obz:
            dist = obz['distance_along_route'] - self.start_distance_along_route
            progress = dist - self.distance_along_route
            if self.distance_along_route:
                self.previous_distance_along_route = self.distance_along_route
            self.distance_along_route = dist
            progress_reward, speed_reward = DeepDriveRewardCalculator.get_progress_and_speed_reward(progress, time_passed)

        progress_reward *= self.driving_style.value.progress_weight
        speed_reward *= self.driving_style.value.speed_weight

        if self.score.episode_time < 2:
            # Speed reward is too big at reset due to small offset between origin and spawn, so clip it to
            # avoid incenting resets
            speed_reward = min(max(speed_reward, -1), 1)
            progress_reward = min(max(progress_reward, -1), 1)

        if gforce_penalty > 0 or lane_deviation_penalty > 0:
            # Discourage making up for penalties by going faster
            speed_reward /= 2
            progress_reward /= 2

        self.score.progress_reward += progress_reward
        self.score.speed_reward += speed_reward

        self.score.progress = self.distance_along_route / 2736.7  # TODO Get length of route dynamically

        self.display_stats['lap progress']['total'] = self.score.progress
        self.display_stats['lap progress']['value'] = self.display_stats['lap progress']['total']
        self.display_stats['episode #']['total'] = self.total_laps
        self.display_stats['episode #']['value'] = self.total_laps
        self.display_stats['speed reward']['total'] = self.score.speed_reward
        self.display_stats['speed reward']['value'] = self.score.speed_reward

        return progress_reward, speed_reward

    def get_time_penalty(self, _obz, time_passed):
        time_penalty = time_passed or 0
        time_penalty *= self.driving_style.value.time_weight
        self.score.time_penalty += time_penalty
        return time_penalty

    def combine_rewards(self, progress_reward, gforce_penalty, lane_deviation_penalty, time_penalty, speed):
        return self.driving_style.value.combine(progress_reward, gforce_penalty, lane_deviation_penalty, time_penalty,
                                                speed)

    def is_stuck(self, obz):
        start_is_stuck = time.time()
        # TODO: Get this from the game instead
        ret = False
        if 'TEST_BENCHMARK_WRITE' in os.environ:
            self.score.got_stuck = True
            self.log_benchmark_trial()
            ret = True
        elif obz is None:
            ret = False
        elif obz['speed'] < 100:  # cm/s
            self.steps_crawling += 1
            if obz['throttle'] > 0 and obz['brake'] == 0 and obz['handbrake'] == 0:
                self.steps_crawling_with_throttle_on += 1
            time_crawling = time.time() - self.last_not_stuck_time
            portion_crawling = self.steps_crawling_with_throttle_on / max(1, self.steps_crawling)
            if self.steps_crawling_with_throttle_on > 20 and time_crawling > 2 and portion_crawling > 0.8:
                log.warn('No progress made while throttle on - assuming stuck and ending episode. steps crawling: %r, '
                         'steps crawling with throttle on: %r, time crawling: %r',
                         self.steps_crawling, self.steps_crawling_with_throttle_on, time_crawling)
                self.set_forward_progress()
                if self.should_benchmark:
                    self.score.got_stuck = True
                ret = True
        else:
            self.set_forward_progress()
        log.debug('is stuck took %fs', time.time() - start_is_stuck)

        return ret

    def log_benchmark_trial(self):
        self.score.end_time = time.time()
        log.info('episode time %r', self.score.episode_time)
        self.trial_scores.append(self.score)
        totals = [s.total for s in self.trial_scores]
        median = np.median(totals)
        average = np.mean(totals)
        high = max(totals)
        low = min(totals)
        std = np.std(totals)
        log.info('benchmark lap #%d score: %f - average: %f', len(self.trial_scores), self.score.total, average)
        file_prefix = self.experiment + '_' if self.experiment else ''
        filename = os.path.join(c.RESULTS_DIR, '%s%s.csv' % (file_prefix, c.DATE_STR))
        diff_filename = '%s%s.diff' % (file_prefix, c.DATE_STR)
        diff_filepath = os.path.join(c.RESULTS_DIR, diff_filename)

        if self.git_diff is not None:
            with open(diff_filepath, 'w') as diff_file:
                diff_file.write(self.git_diff)

        with open(filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for i, score in enumerate(self.trial_scores):
                if i == 0:
                    writer.writerow(['episode #', 'score', 'speed reward', 'lane deviation penalty',
                                     'gforce penalty', 'got stuck', 'wrong way', 'start', 'end', 'lap time'])
                writer.writerow([i + 1, score.total,
                                 score.speed_reward, score.lane_deviation_penalty,
                                 score.gforce_penalty, score.got_stuck, score.wrong_way,
                                 str(arrow.get(score.start_time).to('local')),
                                 str(arrow.get(score.end_time).to('local')),
                                 score.episode_time])
            writer.writerow([])
            writer.writerow(['median score', median])
            writer.writerow(['avg score', average])
            writer.writerow(['std', std])
            writer.writerow(['high score', high])
            writer.writerow(['low score', low])
            writer.writerow(['env', self.spec.id])
            writer.writerow(['args', ', '.join(sys.argv[1:])])
            writer.writerow(['git commit', '@' + self.git_commit])
            writer.writerow(['git diff', diff_filename])
            writer.writerow(['experiment name', self.experiment or 'n/a'])
            writer.writerow(['os', sys.platform])
            try:
                gpus = ','.join([gpu.name for gpu in GPUtil.getGPUs()])
            except:
                gpus = 'n/a'
            writer.writerow(['gpus', gpus])

        log.info('median score %r', median)
        log.info('avg score %r', average)
        log.info('std %r', std)
        log.info('high score %r', high)
        log.info('low score %r', low)
        log.info('progress_reward %r', self.score.progress_reward)
        log.info('speed_reward %r', self.score.speed_reward)
        log.info('lane_deviation_penalty %r', self.score.lane_deviation_penalty)
        log.info('time_penalty %r', self.score.time_penalty)
        log.info('gforce_penalty %r', self.score.gforce_penalty)
        log.info('episode_time %r', self.score.episode_time)
        log.info('wrote results to %s', os.path.normpath(filename))

    def release_agent_control(self):
        self.has_control = deepdrive_client.release_agent_control(self.client_id) is not None

    def request_agent_control(self):
        self.has_control = deepdrive_client.request_agent_control(self.client_id) == 1

    # noinspection PyAttributeOutsideInit
    def set_forward_progress(self):
        self.last_not_stuck_time = time.time()
        self.steps_crawling_with_throttle_on = 0
        self.steps_crawling = 0

    def reset(self):
        self.prev_observation = None
        self.reset_agent()
        self.step_num = 0
        self.distance_along_route = 0
        self.start_distance_along_route = 0
        self.prev_step_time = None
        self.score = Score()
        self.start_time = time.time()
        self.started_driving_wrong_way_time = None
        log.info('Reset complete')
        if self.reset_returns_zero:
            # TODO: Always return zero after testing that everything works with dagger agents
            return 0

    def change_viewpoint(self, cameras, use_sim_start_command):
        self.use_sim_start_command = use_sim_start_command
        deepdrive_capture.close()
        deepdrive_client.close(self.client_id)
        self.client_id = 0
        self.close_sim()  # Need to restart process now to change cameras
        self.open_sim()
        self.connect(cameras)

    def __del__(self):
        self.close()

    def close(self):
        if self.dashboard_pub is not None:
            self.dashboard_pub.put({'should_stop': True})
            self.dashboard_pub.close()
        if self.dashboard_process is not None:
            self.dashboard_process.join()
        deepdrive_capture.close()
        deepdrive_client.release_agent_control(self.client_id)
        deepdrive_client.close(self.client_id)
        self.client_id = 0
        if self.sess:
            self.sess.close()
        self.close_sim()

    def render(self, mode='human', close=False):

        # TODO: Implement proper render - this is really only good for one frame - Could use our OpenGLUT viewer (on raw images) for this or PyGame on preprocessed images
        if self.should_render and self.prev_observation is not None:
            if self.pyglet_render and pyglet is not None and self.pyglet_queue is not None:
                self.pyglet_queue.put(self.prev_observation['cameras'])
            else:
                for camera in self.prev_observation['cameras']:
                    utils.show_camera(camera['image'], camera['depth'])

    def seed(self, seed=None):
        self.np_random = seeding.np_random(seed)
        # TODO: Generate random actions with this seed

    def preprocess_observation(self, observation):
        if observation:
            ret = obj2dict(observation, exclude=['cameras'])
            if observation.camera_count > 0 and getattr(observation, 'cameras', None) is not None:
                cameras = observation.cameras
                ret['cameras'] = self.preprocess_cameras(cameras)
            else:
                ret['cameras'] = []
        else:
            ret = None
        return ret

    def preprocess_cameras(self, cameras):
        ret = []
        for camera in cameras:
            image = camera.image_data.reshape(camera.capture_height, camera.capture_width, 3)
            depth = camera.depth_data.reshape(camera.capture_height, camera.capture_width)
            start_preprocess = time.time()
            if self.preprocess_with_tensorflow:
                import tf_utils  # avoid hard requirement on tensorflow
                if self.sess is None:
                    raise Exception('No tensorflow session. Did you call set_tf_session?')
                # This runs ~2x slower (18ms on a gtx 980) than CPU when we are not running a model due to
                # transfer overhead, but we do it anyway to keep training and testing as similar as possible.
                image = tf_utils.preprocess_image(image, self.sess)
                depth = tf_utils.preprocess_depth(depth, self.sess)
            else:
                image = utils.preprocess_image(image)
                depth = utils.preprocess_depth(depth)

            end_preprocess = time.time()
            log.debug('preprocess took %rms', (end_preprocess - start_preprocess) * 1000.)
            camera_out = obj2dict(camera, exclude=['image', 'depth'])
            camera_out['image'] = image
            if self.pyglet_render:
                # Keep copy of image without mean subtraction etc that agent does
                camera_out['image_raw'] = image
            camera_out['depth'] = depth
            ret.append(camera_out)
        return ret

    def get_observation(self):
        start_get_obz = time.time()
        try:
            obz = deepdrive_capture.step()
            end_get_obz = time.time()
            log.debug('get obz took %fs', end_get_obz - start_get_obz)
        except SystemError as e:
            log.error('caught error during step' + str(e))
            ret = None
        else:
            ret = self.preprocess_observation(obz)
        log.debug('completed capture step')
        self.prev_observation = ret
        return ret

    def reset_agent(self):
        deepdrive_client.reset_agent(self.client_id)

    def send_control(self, action):
        if self.has_control != action.has_control:
            self.change_has_control(action.has_control)

        if action.handbrake:
            log.debug('Not expecting any handbraking right now! What\'s happening?! Disabling - hack :D')
            action.handbrake = False

        action.clip()

        if self.is_sync:
            sync_start = time.time()
            seq_number = deepdrive_client.advance_synchronous_stepping(self.client_id, self.sync_step_time,
                                                                       action.steering, action.throttle,
                                                                       action.brake, action.handbrake)
            log.debug('sync step took %fs',  time.time() - sync_start)
        else:
            if c.PPO_RESUME_PATH:
                action.throttle = action.throttle * 0.90  # OmegaHack to deal with sync vs async
            deepdrive_client.set_control_values(self.client_id, steering=action.steering, throttle=action.throttle,
                                                brake=action.brake, handbrake=action.handbrake)

    def set_step_mode(self):
        if self.is_sync:
            ret = deepdrive_client.activate_synchronous_stepping(self.client_id)
            if ret != 1:
                raise RuntimeError('Could not activate synchronous mode - errno %r' % ret)

    def check_version(self):
        self.client_id = self.connection_props['client_id']
        server_version = self.connection_props['server_protocol_version']
        if not server_version:
            log.warn('Server version not reported! Hoping for the best.')
        else:
            server_version = semvar(server_version).version
            # TODO: For dev, store hash of .cpp and .h files on extension build inside VERSION_DEV, then when
            #   connecting, compute same hash and compare. (Need to figure out what to do on dev packaged version as
            #   files may change - maybe ignore as it's uncommon).
            #   Currently, we timestamp the build, and set that as the version in the extension. This is fine unless
            #   you change shared code and build the extension only, then the versions won't change, and you could
            #   see incompatibilities.
            if semvar(self.client_version).version[:2] != server_version[:2]:
                raise RuntimeError(
                    'Server and client major/minor version do not match - server is %s and client is %s' %
                    (server_version, self.client_version))

    def connect(self, cameras=None):
        self._connect_with_retries()

        if cameras is None:
            cameras = [c.DEFAULT_CAM]
        self.cameras = cameras
        if self.client_id and self.client_id > 0:
            for cam in self.cameras:
                cam['cxn_id'] = deepdrive_client.register_camera(self.client_id, cam['field_of_view'],
                                                              cam['capture_width'],
                                                              cam['capture_height'],
                                                              cam['relative_position'],
                                                              cam['relative_rotation'],
                                                              cam['name'])

            shared_mem = deepdrive_client.get_shared_memory(self.client_id)
            self.reset_capture(shared_mem[0], shared_mem[1])
            self._init_observation_space()
        else:
            self.raise_connect_fail()

        if self.pyglet_render:
            self.init_pyglet(cameras)

        self._perform_first_step()
        self.has_control = False

    def _connect_with_retries(self):

        def connect():
            try:
                self.connection_props = deepdrive_client.create('127.0.0.1', 9876)
                if isinstance(self.connection_props, int):
                    raise Exception('You have an old version of the deepdrive client - try uninstalling and reinstalling with pip')
                if not self.connection_props or not self.connection_props['max_capture_resolution']:
                    # Try again
                    return
                self.check_version()

            except deepdrive_client.time_out:
                connect()

        connect()
        cxn_attempts = 0
        max_cxn_attempts = 10
        while not self.connection_props:
            cxn_attempts += 1
            sleep = cxn_attempts + random.random() * 2  # splay to avoid thundering herd
            log.warning('Connection to environment failed, retry (%d/%d) in %d seconds',
                        cxn_attempts, max_cxn_attempts, round(sleep, 0))
            time.sleep(sleep)
            connect()
            if cxn_attempts >= max_cxn_attempts:
                raise RuntimeError('Could not connect to the environment')

    def _perform_first_step(self):
        obz = None
        read_obz_count = 0
        while obz is None:
            if read_obz_count > 10:
                error_msg = 'Failed first step of environment'
                log.error(error_msg)
                raise RuntimeError(error_msg)
            try:
                obz = deepdrive_capture.step()
            except SystemError as e:
                log.error('caught error during step' + str(e))
            time.sleep(0.25  * read_obz_count)
            read_obz_count += 1

    def reset_capture(self, shared_mem_name, shared_mem_size):
        n = 10
        sleep = 0.1
        log.debug('Connecting to deepdrive...')
        while n > 0:
            # TODO: Establish some handshake so we don't hardcode size here and in Unreal project
            if deepdrive_capture.reset(shared_mem_name, shared_mem_size):
                log.debug('Connected to deepdrive shared capture memory')
                return
            n -= 1
            sleep *= 2
            log.debug('Sleeping %r', sleep)
            time.sleep(sleep)
        log.error('Could not connect to deepdrive capture memory at %s', shared_mem_name)
        self.raise_connect_fail()

    @staticmethod
    def raise_connect_fail():
        log.error('Environment connection failed')
        if c.SIM_START_COMMAND:
            raise Exception('Could not connect to environment. You may need to close the Unreal Editor and/or turn off '
                            'saving CPU in background in the Editor preferences (search for CPU).')
        else:
            raise Exception('\n\n\n'
                        '**********************************************************************\n'
                        '**********************************************************************\n'
                        '****                                                              ****\n\n'
                        '|               Could not connect to the environment.                |\n\n'
                        '****                                                              ****\n'
                        '**********************************************************************\n'
                        '**********************************************************************\n\n')

    def init_action_space(self):
        if self.is_discrete:
            num_steer_steps = 30
            steer_step = 1 / ((num_steer_steps - 1) / 2)

            # Fermi estimate of a good discretization
            steer = list(np.arange(-1, 1 + steer_step, steer_step))
            throttle = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.8, 1]
            brake = [0, 0.1, 0.3, 0.7, 1]

            self.discrete_actions = DiscreteActions(steer, throttle, brake)

            learned_space = spaces.Discrete(len(self.discrete_actions.product))
            is_game_driving_space = spaces.Discrete(2)

            # action_space = spaces.Tuple(
            #     (learned_space, is_game_driving_space))

            action_space = learned_space

        else:
            # TODO(PPO)
            # I think what we need to do here is normalize the brake and handbrake space to be between -1 and 1 so
            # that all actions have the same dimension - Then create a single box space. The is_game_driving space
            # can be ignored for now within the ppo agent.
            steering_space = spaces.Box(low=Action.STEERING_MIN, high=Action.STEERING_MAX, shape=(1,), dtype=np.float32)
            throttle_space = spaces.Box(low=Action.THROTTLE_MIN, high=Action.THROTTLE_MAX, shape=(1,), dtype=np.float32)
            brake_space = spaces.Box(low=Action.BRAKE_MIN, high=Action.BRAKE_MAX, shape=(1,), dtype=np.float32)
            handbrake_space = spaces.Box(low=Action.HANDBRAKE_MIN, high=Action.HANDBRAKE_MAX, shape=(1,),
                                         dtype=np.float32)
            is_game_driving_space = spaces.Discrete(2)
            action_space = spaces.Tuple(
                (steering_space, throttle_space, brake_space, handbrake_space, is_game_driving_space))
        self.action_space = action_space
        return action_space

    def _init_observation_space(self):
        if len(self.cameras) > 1:
            log.warning('\n\n\n MULTIPLE CAMERAS OBSERVATION SPACE RETURNS TUPLE - '
                        'YOU MAY WANT TO IMPLEMENT BETTER SUPPORT DEPENDING ON HOW YOUR '
                        'AGENT COMBINES CAMERA VIEWS \n\n\n')

            obz_spaces = []
            for camera in self.cameras:
                obz_spaces.append(spaces.Box(low=0, high=255, shape=(camera['capture_width'], camera['capture_height']),
                                             dtype=np.uint8))
            observation_space = spaces.Tuple(tuple(obz_spaces))
            self.observation_space = observation_space
            return observation_space
        else:
            camera = self.cameras[0]
            self.observation_space = spaces.Box(low=0, high=255, shape=(camera['capture_width'], camera['capture_height']),
                       dtype=np.uint8)

    def change_has_control(self, has_control):
        if has_control:
            self.request_agent_control()
        else:
            self.release_agent_control()

    def driving_wrong_way(self):
        if None in [self.previous_distance_along_route, self.distance_along_route]:
            return False

        if self.distance_along_route < self.previous_distance_along_route:
            now = time.time()
            s = self.started_driving_wrong_way_time
            if s is not None:
                if (now - s) > 2:
                    return True
            else:
                self.started_driving_wrong_way_time = now

        else:
            self.started_driving_wrong_way_time = None
        return False


class DeepDriveRewardCalculator(object):
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
        lane_deviation_penalty = DeepDriveRewardCalculator.clip(lane_deviation_penalty)
        return lane_deviation_penalty

    @staticmethod
    def get_gforce_penalty(gforces, time_passed):
        log.debug('gforces %r', gforces)
        gforce_penalty = 0
        if gforces < 0:
            raise ValueError('G-Force should be positive')
        if gforces > 0.3:
            # Based on regression model on page 47 - can achieve 92/100 comfort with ~0.3 combined x and y acceleration
            # http://www.diva-portal.org/smash/get/diva2:950643/FULLTEXT01.pdf
            time_weighted_gs = time_passed * gforces
            time_weighted_gs = min(time_weighted_gs,
                                   5)  # Don't allow a large frame skip to ruin the approximation
            balance_coeff = 24  # 24 meters of reward every second you do this
            gforce_penalty = time_weighted_gs * balance_coeff
            log.debug('accumulated_gforce %r', time_weighted_gs)
            log.debug('gforce_penalty %r', gforce_penalty)
        gforce_penalty = DeepDriveRewardCalculator.clip(gforce_penalty)
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

        progress_reward = DeepDriveRewardCalculator.clip(progress_reward)
        speed_reward = DeepDriveRewardCalculator.clip(speed_reward)
        return progress_reward, speed_reward


def render_cameras(render_queue, cameras):
    if pyglet is None:
        return
    widths = []
    heights = []
    for camera in cameras:
        widths += [camera['capture_width']]
        heights += [camera['capture_height']]

    width = max(widths) * 2  # image and depths
    height = sum(heights)
    window = pyglet.window.Window(width, height)
    fps_display = pyglet.clock.ClockDisplay()

    @window.event
    def on_draw():
        window.clear()
        cams = render_queue.get(block=True)
        channels = 3
        bytes_per_channel = 1
        for cam_idx, cam in enumerate(cams):
            img_data = np.copy(cam['image_raw'])
            depth_data = np.ascontiguousarray(utils.depth_heatmap(np.copy(cam['depth'])))
            img_data.shape = -1
            depth_data.shape = -1
            img_texture = (GLubyte * img_data.size)(*img_data.astype('uint8'))
            depth_texture = (GLubyte * depth_data.size)(*depth_data.astype('uint8'))
            image = pyglet.image.ImageData(
                cam['capture_width'],
                cam['capture_height'],
                'RGB',
                img_texture,
                pitch= -1 * cam['capture_width'] * channels * bytes_per_channel)
            depth = pyglet.image.ImageData(
                cam['capture_width'],
                cam['capture_height'],
                'RGB',
                depth_texture,
                pitch= -1 * cam['capture_width'] * channels * bytes_per_channel)
            if image is not None:
                image.blit(0, cam_idx * cam['capture_height'])
            if depth is not None:
                depth.blit(cam['capture_width'], cam_idx * cam['capture_height'])
        fps_display.draw()

    while True:
        pyglet.clock.tick()

        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event('on_draw')
            window.flip()


if __name__ == '__main__':
    DeepDriveEnv.get_latest_sim_file()
