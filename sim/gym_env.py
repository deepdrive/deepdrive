from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import platform
import shutil

# noinspection PyUnresolvedReferences
import config.check_bindings


from future.builtins import (int, open, round,
                             str)
import csv
import os
import random
import sys
import time
from collections import OrderedDict
from multiprocessing import Process
from subprocess import Popen
import pkg_resources
from distutils.version import LooseVersion as semvar

import arrow
import gym
import numpy as np
from GPUtil import GPUtil
from gym import spaces
from gym.utils import seeding
import deepdrive_client
import deepdrive_capture
import deepdrive_simulation

import config as c
import logs
import utils
from sim import world
from sim.action import Action, DiscreteActions
from sim.graphics import set_capture_graphics
from sim.reward_calculator import RewardCalculator
from sim.score import Score
from sim.view_mode import ViewModeController
from renderer import renderer_factory, base_renderer
from utils import obj2dict
from dashboard import dashboard_fn, DashboardPub

log = logs.get_log(__name__)
SPEED_LIMIT_KPH = 64.
HEAD_START_TIME = 3
LAP_LENGTH = 2736.7


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
        self.should_render = False  # type: bool
        self.ep_time_balance_coeff = 10.  # type: float
        self.previous_action_time = None  # type: time.time
        self.fps = None  # type: int
        self.period = None  # type: float
        self.experiment = None  # type: str
        self.driving_style = None  # type: DrivingStyle
        self.reset_returns_zero = None  # type: bool
        self.started_driving_wrong_way_time = None  # type: bool
        self.previous_distance_along_route = None  # type: float
        self.renderer = None  # type: base_renderer.Renderer
        self.np_random = None  # type: tuple
        self.last_obz = None  # type: dict
        self.view_mode_controller = None  # type: ViewModeController
        self.enable_traffic = False  # type: bool
        self.ego_mph = None  # type: float
        self.max_steps = None  # type: int

        if not c.REUSE_OPEN_SIM:
            utils.ensure_sim()

        self.client_version = pkg_resources.get_distribution('deepdrive').version

        # collision detection  # TODO: Remove in favor of in-game detection
        self.set_forward_progress()

        self.distance_along_route = 0  # type: float
        self.start_distance_along_route = None  # type: float
        self.previous_distance_along_route = 0  # type: float

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
            cmd = utils.get_sim_bin_path()

            if log.getEffectiveLevel() < 20:  # More verbose than info (i.e. debug)
                cmd += ' -LogCmds="LogPython Verbose, LogSharedMemoryImpl_Linux VeryVerbose, LogDeepDriveAgent VeryVerbose"'

            self.sim_process = Popen(utils.get_sim_bin_path())
            log.info('Starting simulator at %s (takes a few seconds the first time).', cmd)

    def close_sim(self):
        log.info('Closing sim')
        self.connection_props = None
        process_to_kill = self.sim_process
        if process_to_kill is not None:
            utils.kill_process(process_to_kill)

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
        p.daemon = True
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
        if not obz:
            log.debug('Observation not available - step %r', self.step_num)

        self.last_obz = obz
        if self.should_render:
            self.render()

        now = time.time()

        start_reward_stuff = time.time()
        reward, done = self.get_reward(obz, now)
        self.prev_step_time = now

        if self.dashboard_pub is not None:
            start_dashboard_put = time.time()
            self.dashboard_pub.put(OrderedDict({'display_stats': list(self.display_stats.items()),
                                                'should_stop': False}))
            log.debug('dashboard put took %fs', time.time() - start_dashboard_put)

        self.step_num += 1
        log.debug('reward stuff took %fs', time.time() - start_reward_stuff)

        info = self.init_step_info()

        if done:
            self.report_score(info)

        if obz is not None:
            obz['score'] = utils.obj2dict(self.score)

        self.regulate_fps()
        self.view_mode_controller.step(self.client_id)

        return obz, reward, done, info

    def init_step_info(self, done):
        info = {}
        info['score'] = info.get('episode', {})
        info['score']['episode_time'] = self.score.episode_time

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
        log.debug('in regulate_fps')
        if self.previous_action_time:
            delta = now - self.previous_action_time
            fps = 1. / max(delta, 1E-9)
            log.debug('step duration delta actual %f desired %f', delta, self.period)
            if delta < self.period:
                self.sync_step_time = self.period
                if not self.is_sync:
                    sleep_time = max(0., self.period - delta - 0.001)
                    log.debug('regulating fps by sleeping for %f', sleep_time)
                    time.sleep(sleep_time)  # TODO: Set environment capture FPS so that sleep is not needed here.
            else:
                log.debug('step longer than desired')
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
            have_lap_numbers = lap_number is not None and self.lap_number is not None
            lap_via_progress = self.score.progress > 99.9
            if lap_via_progress:
                median_meters_per_sec = self.score.speed_sampler.mean() / 100
                est_travel_cm = median_meters_per_sec * self.score.episode_time * 100  # cm travelled
                took_shortcut = est_travel_cm < (obz['route_length'] * 0.9)
                if took_shortcut:
                    log.warn('Shortcut detected, not scoring lap')
                else:
                    lap_bonus = 10
                self.total_laps += 1
                done = True  # One lap per episode
                log.info('episode finished, lap complete')
                self.log_up_time()
            self.lap_number = lap_number
            log.debug('compute lap stats took %fs', time.time() - start_compute_lap_stats)

        return done, lap_bonus

    def get_reward(self, obz, now):
        start_get_reward = time.time()
        done = False
        lap_done, lap_bonus = self.compute_lap_statistics(obz)
        reward = 0
        gforce_done = False
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
                gforce_penalty, gforce_done = self.get_gforce_penalty(obz, step_time)
                lane_deviation_penalty = self.get_lane_deviation_penalty(obz, step_time)
                time_penalty = self.get_time_penalty(obz, step_time)
                progress_reward, speed = self.get_progress_and_speed_reward(obz, step_time,
                                                                            gforce_penalty, lane_deviation_penalty)
                reward = self.combine_rewards(progress_reward, gforce_penalty, lane_deviation_penalty,
                                              time_penalty, speed)

            self.score.wrong_way = self.driving_wrong_way()
            if self.score.wrong_way:
                log.warn('episode finished, going the wrong way')

            if self.is_stuck(obz) or self.score.wrong_way:  # TODO: Collision or near collision
                done = True
                reward -= 10
                # TODO: Scale cost by collision momentum when speed is returned
                # if obz['last_collision'].time_utc:
                #     reward *= obz['last_collision'].speed

            self.score.speed_sampler.sample(obz['speed'])
            self.score.total += reward + lap_bonus
            self.display_stats['time']['value'] = self.score.episode_time
            self.display_stats['time']['total'] = self.score.episode_time
            self.display_stats['episode score']['value'] = self.score.total
            self.display_stats['episode score']['total'] = self.score.total

            log.debug('reward %r', reward)
            log.debug('score %r', self.score.total)
            log.debug('progress %r', self.score.progress)
            log.debug('throttle %f', obz['throttle'])
            log.debug('steering %f', obz['steering'])
            log.debug('brake %f', obz['brake'])
            log.debug('handbrake %f', obz['handbrake'])

        log.debug('get reward took %fs', time.time() - start_get_reward)

        steps_done = self.step_num == self.max_steps

        if steps_done:
            log.info('Ending episode due to max steps (%r)' % self.max_steps)

        done = done or lap_done or gforce_done or steps_done

        return reward, done

    def log_up_time(self):
        log.info('up for %r' % arrow.get(time.time()).humanize(other=arrow.get(self.start_time), only_distance=True))

    def get_lane_deviation_penalty(self, obz, time_passed):
        lane_deviation_penalty = 0.
        if 'distance_to_center_of_lane' in obz:
            lane_deviation_penalty = RewardCalculator.get_lane_deviation_penalty(
                obz['distance_to_center_of_lane'], time_passed)

        lane_deviation_penalty *= self.driving_style.value.lane_deviation_weight
        self.score.lane_deviation_penalty += lane_deviation_penalty

        self.display_stats['lane deviation penalty']['value'] = -lane_deviation_penalty
        self.display_stats['lane deviation penalty']['total'] = -self.score.lane_deviation_penalty
        return lane_deviation_penalty

    def get_gforce_penalty(self, obz, time_passed):
        gforce_penalty = 0
        done = False

        if 'acceleration' in obz:
            if time_passed is not None:
                a = obz['acceleration']
                gs = np.sqrt(a.dot(a)) / 980  # g = 980 cm/s**2
                sampler = self.score.gforce_sampler
                sampler.sample(gs)
                three_second_avg = self.average_gs(sampler, secs=3)
                if gs > 5 or three_second_avg > 4:
                    # Game over
                    log.warn('G-force limit exceeded, game over. Recent g\'s were: %r',
                             list(reversed(list(sampler.q)[-10:])))
                    done = True

                self.display_stats['g-forces']['value'] = gs
                self.display_stats['g-forces']['total'] = gs
                gforce_penalty = RewardCalculator.get_gforce_penalty(gs, time_passed)

        gforce_penalty *= self.driving_style.value.gforce_weight
        self.score.gforce_penalty += gforce_penalty

        self.display_stats['gforce penalty']['value'] = -self.score.gforce_penalty
        self.display_stats['gforce penalty']['total'] = -self.score.gforce_penalty
        return gforce_penalty, done

    def get_progress_and_speed_reward(self, obz, time_passed, gforce_penalty, lane_deviation_penalty):
        progress_reward = speed_reward = 0
        if 'distance_along_route' in obz:
            if self.start_distance_along_route is None:
                self.start_distance_along_route = obz['distance_along_route']
            if obz['distance_along_route'] < self.start_distance_along_route:
                dist = (obz['route_length'] - self.start_distance_along_route) + obz['distance_along_route']
            else:
                dist = obz['distance_along_route'] - self.start_distance_along_route
            progress = dist - self.distance_along_route
            if self.distance_along_route:
                self.previous_distance_along_route = self.distance_along_route
            self.distance_along_route = dist
            progress_reward, speed_reward = RewardCalculator.get_progress_and_speed_reward(progress, time_passed)
            self.score.prev_progress = self.score.progress
            self.score.progress = 100 * self.distance_along_route / obz['route_length']

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

        self.display_stats['lap progress']['total'] = self.score.progress or 0
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
        if 'TEST_END_OF_EPISODE' in os.environ and self.step_num >= 9:
            log.warn('TEST_END_OF_EPISODE is set triggering end of episode via is_stuck!')
            self.score.got_stuck = True
            self.log_benchmark_trial()
            ret = True
        elif obz is None:
            log.debug('obz is None, not checking if stuck')
            ret = False
        elif obz['speed'] < 100:  # cm/s
            log.debug('speed less than 1m/s, checking if stuck')
            self.steps_crawling += 1
            if obz['throttle'] > 0 and obz['brake'] == 0 and obz['handbrake'] == 0:
                self.steps_crawling_with_throttle_on += 1
                log.debug('crawling detected num steps crawling is %d', self.steps_crawling_with_throttle_on)
            else:
                log.debug('not stuck, throttle %f, brake %f, handbrake %f', obz['throttle'], obz['brake'],
                          obz['handbrake'])

            time_crawling = time.time() - self.last_not_stuck_time

            # This was to detect legitimate stops, but we will have real collision detection before the need to stop
            # portion_crawling = self.steps_crawling_with_throttle_on / max(1, self.steps_crawling)

            if self.steps_crawling_with_throttle_on > 40 and time_crawling > 5:
                log.warn('No progress made while throttle on - assuming stuck and ending episode. steps crawling: %r, '
                         'steps crawling with throttle on: %r, time crawling: %r',
                         self.steps_crawling, self.steps_crawling_with_throttle_on, time_crawling)
                self.set_forward_progress()
                if self.should_benchmark:
                    self.score.got_stuck = True
                ret = True
        else:
            log.debug('speed greater than 1m/s, not stuck')
            self.set_forward_progress()
        log.debug('is stuck took %fs', time.time() - start_is_stuck)

        if ret:
            log.info('episode finished, detected we were stuck')

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
        diff_filename = '%s%s.diff' % (file_prefix, c.DATE_STR)
        diff_filepath = os.path.join(c.RESULTS_DIR, diff_filename)

        if self.git_diff is not None:
            with open(diff_filepath, 'w', encoding='utf-8') as diff_file:
                diff_file.write(self.git_diff)

        self.write_result_csvs(average, diff_filename, high, low, median, std)

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
        log.info('wrote results to %s and %s', os.path.normpath(c.EPISODES_CSV_FILENAME),
                 os.path.normpath(c.SUMMARY_CSV_FILENAME))

    def write_result_csvs(self, average, diff_filename, high, low, median, std):
        with open(c.EPISODES_CSV_FILENAME, 'w', newline='') as csv_file1:
            writer = csv.writer(csv_file1)
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
        with open(c.SUMMARY_CSV_FILENAME, 'w', newline='') as csv_file2:
            writer = csv.writer(csv_file2)
            writer.writerow(['Stat', 'Value'])
            writer.writerow(['median score', median])
            writer.writerow(['avg score', average])
            writer.writerow(['std', std])
            writer.writerow(['high score', high])
            writer.writerow(['low score', low])
            writer.writerow(['env', self.spec.id])
            writer.writerow(['cmd args', ', '.join(sys.argv[1:])])
            writer.writerow(['py args', str(vars(c.PY_ARGS) if c.PY_ARGS else '')])
            writer.writerow(['git commit', '@' + self.git_commit])
            writer.writerow(['git diff', diff_filename])
            writer.writerow(['experiment name', self.experiment or 'n/a'])
            writer.writerow(['run id', c.RUN_ID])

            writer.writerow(['os', self.get_os_version()])

            try:
                gpus = ','.join([gpu.name for gpu in GPUtil.getGPUs()])
            except:
                gpus = 'n/a'
            writer.writerow(['gpus', gpus])

    def get_os_version(self):
        os_version = platform.platform()
        if c.IS_LINUX:
            try:
                os_version = ' '.join(utils.run_command('lsb_release -a')[0].split() + [os_version])
            except:
                log.debug('Could not get os version from lsb_release')
        return os_version

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
        self.reset_agent()
        self.step_num = 0
        self.distance_along_route = 0
        self.previous_distance_along_route = 0
        self.start_distance_along_route = None
        self.prev_step_time = None
        self.score = Score()
        self.start_time = time.time()
        self.started_driving_wrong_way_time = None
        set_capture_graphics(shadow_level=0)
        self.view_mode_controller.reset()
        log.info('Reset complete')
        if self.reset_returns_zero:
            # TODO: Always return zero after testing that everything works with dagger agents
            return 0

    def change_viewpoint(self, cameras, use_sim_start_command):
        # TODO: Delete this method
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
            try:
                self.dashboard_pub.put({'should_stop': True})
                time.sleep(0.25)  # Give time for message to be received
            except Exception as e:
                log.error('Error closing dashboard. %s', e)
                # print(traceback.format_exc())
            self.dashboard_pub.close()
        log.info('Closed dashboard')
        if self.dashboard_process is not None:
            self.dashboard_process.join(timeout=.25)
        if self.is_sync:
            deepdrive_client.deactivate_synchronous_stepping(self.client_id)
        deepdrive_capture.close()
        deepdrive_client.release_agent_control(self.client_id)
        deepdrive_client.close(self.client_id)
        deepdrive_simulation.disconnect()
        self.client_id = 0

        # Keep this open as agents share this session and restart the env for things like changing cameras during recording.
        # if self.sess:
        #     self.sess.close()

        self.close_sim()

    def render(self, mode='human', close=False):
        if self.last_obz:
            self.renderer.render(self.last_obz)

    def seed(self, seed=None):
        self.np_random = seeding.np_random(seed)
        # TODO: Generate random actions with this seed

    def preprocess_observation(self, observation):
        if observation and len(observation.cameras[0].image_data):
            ret = obj2dict(observation, exclude=['cameras'])
            if observation.camera_count > 0 and getattr(observation, 'cameras', None) is not None:
                cameras = observation.cameras
                ret['cameras'] = self.preprocess_cameras(cameras)
            else:
                ret['cameras'] = []
            ret['view_mode'] = self.view_mode_controller.current_mode_name()
            if ret['last_collision']:
                ret['last_collision'] = obj2dict(ret['last_collision'])
        else:
            if observation and len(observation.cameras[0].image_data) == 0:
                log.warn('No camera data received - nulling observation')
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
            if self.should_render:
                # Keep copy of image without mean subtraction etc that agent does
                # through side effect on this mutable sensor data that gets passed around
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
        return ret

    def reset_agent(self):
        world.reset(enable_traffic=self.enable_traffic)
        if self.ego_mph is not None:
            world.set_ego_mph(self.ego_mph, self.ego_mph)

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
        server_version_str = self.connection_props['server_protocol_version']
        if not server_version_str:
            log.warn('Server version not reported. Can not check version compatibility.')
        else:
            server_version = semvar(server_version_str).version
            # TODO: For dev, store hash of .cpp and .h files on extension build inside VERSION_DEV, then when
            #   connecting, compute same hash and compare. (Need to figure out what to do on dev packaged version as
            #   files may change - maybe ignore as it's uncommon).
            #   Currently, we timestamp the build, and set that as the version in the extension. This is fine unless
            #   you change shared code and build the extension only, then the versions won't change, and you could
            #   see incompatibilities.
            if c.MAJOR_MINOR_VERSION != server_version[:2]:
                self.deal_with_server_version_mismatch(server_version_str)
                return False
        return True

    def deal_with_server_version_mismatch(self, server_version_str):
        self.close_sim()
        log.error(
            'Server client version mismatch server@%s client@%s - closed sim' %
            (server_version_str, self.client_version))
        sim_url = utils.get_sim_url()
        if sim_url:
            answer = input('We\'ve found a version of the sim which matches your client. Would you like'
                           ' to download it now? [y/n] ')
            if answer.lower().strip() == 'y':
                backup_dir = os.path.join(
                    c.DEEPDRIVE_DIR, '%s-%s' % (c.SIM_PREFIX, server_version_str))
                log.warn('Backing up old sim to %s', backup_dir)
                shutil.move(c.SIM_PATH, backup_dir)
                utils.ensure_sim()
                self.open_sim()

    def connect(self, cameras=None):
        self._connect_with_retries()

        if cameras is None:
            cameras = [c.DEFAULT_CAM]

        if self.client_id and self.client_id > 0:
            self.register_cameras(cameras)
            shared_mem = deepdrive_client.get_shared_memory(self.client_id)
            self.reset_capture(shared_mem[0], shared_mem[1])
            if not deepdrive_simulation.connect('127.0.0.1', 9009, seed=c.rng.randint(1, 10**9)):
                raise RuntimeError('Could not connect to Deepdrive simulation server')
            self._init_observation_space()
        else:
            log.error('Invalid client_id of "%s". Expected a value greater than zero. Aborting connection',
                      str(self.client_id))
            self.raise_connect_fail()

        if self.should_render:
            self.renderer = renderer_factory(cameras=cameras)

        self._perform_first_step()

    def register_cameras(self, cameras):
        for cam in cameras:
            cam['cxn_id'] = deepdrive_client.register_camera(self.client_id, cam['field_of_view'],
                                                             cam['capture_width'],
                                                             cam['capture_height'],
                                                             cam['relative_position'],
                                                             cam['relative_rotation'],
                                                             cam['name'])

            if 'view_mode' in cam:
                self.view_mode_controller.set_view_mode(cam['view_mode'], cam_id=cam['cxn_id'])

            self.cameras = cameras

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
        if self.connection_props:
            log.info('Connecting to an already open sim')
            self.unregister_cameras()
            for _client_id in range(1, self.connection_props['client_id']):
                try:
                    deepdrive_client.deactivate_synchronous_stepping(_client_id)
                except deepdrive_client.client_doesnt_exist:
                    pass
        cxn_attempts = 0
        max_cxn_attempts = 10
        while not self.connection_props:
            cxn_attempts += 1
            if cxn_attempts == 1:
                log.info('No open sim detected')
                self.open_sim()
            else:
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
        if c.SIM_START_COMMAND or c.REUSE_OPEN_SIM:
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
                if (now - s) > 5:
                    return True
            else:
                self.started_driving_wrong_way_time = now

        else:
            self.started_driving_wrong_way_time = None
        return False

    def change_cameras(self, cameras):
        self.unregister_cameras()
        self.register_cameras(cameras)

    def unregister_cameras(self):
        if not deepdrive_client.unregister_camera(self.client_id, 0):  # 0 => Unregister all
            raise RuntimeError('Not able to unregister cameras')
        self.cameras = None

    def average_gs(self, gforce_sampler, secs):
        total = 0
        steps = secs * self.fps
        if len(gforce_sampler.q) < steps:
            return 0
        for i in range(steps):
            total += gforce_sampler.q[-i]
        avg = total / steps
        return avg



