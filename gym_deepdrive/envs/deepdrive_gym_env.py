import csv
import subprocess
import deepdrive_client
import deepdrive_capture
import os
import queue
import random
import time
from collections import deque, OrderedDict
from multiprocessing import Process, Queue
from subprocess import Popen
import pkg_resources
from distutils.version import LooseVersion as semvar


import arrow
import gym
import numpy as np
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
from dashboard import dashboard_fn

log = logs.get_log(__name__)
SPEED_LIMIT_KPH = 64.


class Score(object):
    total = 0
    gforce_penalty = 0
    lane_deviation_penalty = 0
    progress_reward = 0
    got_stuck = False

    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        self.episode_time = 0


class Action(object):
    def __init__(self, steering=0, throttle=0, brake=0, handbrake=0, has_control=True):
        self.steering = steering
        self.throttle = throttle
        self.brake = brake
        self.handbrake = handbrake
        self.has_control = has_control

    def as_gym(self):
        ret = gym_action(steering=self.steering, throttle=self.throttle, brake=self.brake,
                         handbrake=self.handbrake, has_control=self.has_control)
        return ret

    @classmethod
    def from_gym(cls, action):
        ret = cls(steering=action[0][0], throttle=action[1][0],
                  brake=action[2][0], handbrake=action[3][0], has_control=action[4])
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

    def __init__(self, preprocess_with_tensorflow=False):
        self.action_space = self._init_action_space()
        self.preprocess_with_tensorflow = preprocess_with_tensorflow
        self.sess = None
        self.prev_observation = None
        self.start_time = time.time()
        self.step_num = 0
        self.prev_step_time = None
        self.display_stats = OrderedDict()
        self.display_stats['g-forces']                      = {'total': 0, 'value': 0, 'ymin': 0,     'ymax': 3,    'units': 'g'}
        self.display_stats['gforce penalty']                = {'total': 0, 'value': 0, 'ymin': -500,  'ymax': 0,    'units': ''}
        self.display_stats['lane deviation penalty']        = {'total': 0, 'value': 0, 'ymin': -500,  'ymax': 0,    'units': ''}
        self.display_stats['lap progress']              = {'total': 0, 'value': 0, 'ymin': 0,     'ymax': 100,  'units': '%'}
        self.display_stats['episode #']                     = {'total': 0, 'value': 0, 'ymin': 0,     'ymax': 5,    'units': ''}
        self.display_stats['time']                          = {'total': 0, 'value': 0, 'ymin': 0,     'ymax': 250,  'units': 's'}
        self.display_stats['episode score']                 = {'total': 0, 'value': 0, 'ymin': -500,  'ymax': 2000, 'units': ''}
        self.dashboard_process = None
        self.dashboard_queue = None
        self.should_exit = False
        self.sim_process = None
        self.client_id = None
        self.has_control = None
        self.cameras = None
        self.use_sim_start_command = None
        self.connection_props = None
        self.one_frame_render = False
        self.pyglet_render = False
        self.pyglet_image = None
        self.pyglet_process = None
        self.pyglet_queue = None
        self.ep_time_balance_coeff = 10
        self.previous_action_time = None
        self.fps = None
        self.period = None
        self.experiment = None

        if not c.REUSE_OPEN_SIM:
            if utils.get_sim_bin_path() is None:
                print('\n--------- Simulator not found, downloading ----------')
                if c.IS_LINUX or c.IS_WINDOWS:
                    url = c.BASE_URL + self.get_latest_sim_file()
                    download(url, c.SIM_PATH, warn_existing=False, overwrite=False)
                else:
                    raise NotImplementedError('Sim download not yet implemented for this OS')
            utils.ensure_executable(utils.get_sim_bin_path())

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
        self.done_benchmarking = False
        self.trial_scores = []

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

    @staticmethod
    def get_latest_sim_file():
        if c.IS_WINDOWS:
            os_name = 'windows'
        elif c.IS_LINUX:
            os_name = 'linux'
        else:
            raise RuntimeError('Unexpected OS')
        sim_prefix = 'sim/deepdrive-sim-'
        conn = S3Connection(anon=True)
        bucket = conn.get_bucket('deepdrive')
        deepdrive_version = pkg_resources.get_distribution('deepdrive').version
        major_minor = deepdrive_version[:deepdrive_version.rindex('.')]
        sim_versions = list(bucket.list(sim_prefix + os_name + '-' + major_minor))

        latest_sim_file, path_version = sorted([(x.name, x.name.split('.')[-2])
                                           for x in sim_versions],
                                          key=lambda y: y[1])[-1]
        return '/' + latest_sim_file

    def init_benchmarking(self):
        self.should_benchmark = True
        os.makedirs(c.BENCHMARK_DIR, exist_ok=True)

    def init_pyglet(self, cameras):
        q = Queue(maxsize=1)
        p = Process(target=render_cameras, args=(q, cameras))
        p.start()
        self.pyglet_process = p
        self.pyglet_queue = q

    def start_dashboard(self):
        if utils.is_debugging():
            # TODO: Deal with plot UI not being in the main thread somehow - (move to browser?)
            log.warning('Dashboard not supported in debug mode')
            return
        q = Queue(maxsize=10)
        p = Process(target=dashboard_fn, args=(q,))
        print('DEBUG - after starting dashboard')
        p.start()
        self.dashboard_process = p
        self.dashboard_queue = q

    def set_tf_session(self, session):
        self.sess = session

    def step(self, action):
        dd_action = Action.from_gym(action)
        self.send_control(dd_action)
        obz = self.get_observation()
        if obz and 'is_game_driving' in obz:
            self.has_control = not obz['is_game_driving']
        now = time.time()
        done = False
        reward = self.get_reward(obz, now)
        done = self.compute_lap_statistics(done, obz)
        self.prev_step_time = now

        if self.dashboard_queue is not None:
            self.dashboard_queue.put({'display_stats': self.display_stats, 'should_stop': False})
        if self.is_stuck(obz):  # TODO: derive this from collision, time elapsed, and distance as well
            done = True
            reward -= -10000  # reward is in scale of meters
        info = {}
        self.step_num += 1

        self.regulate_fps()


        return obz, reward, done, info

    def regulate_fps(self):
        now = time.time()
        if self.previous_action_time:
            delta = now - self.previous_action_time
            if delta < self.period:
                time.sleep(delta)
            else:
                fps = 1. / delta
                if self.step_num > 5 and fps < self.fps / 2:
                    log.warning('Step %r took %rs - target is %rs', self.step_num, delta, 1 / self.fps)
        self.previous_action_time = now

    def compute_lap_statistics(self, done, obz):
        if not obz:
            return done
        lap_number = obz.get('lap_number')
        if lap_number is not None and self.lap_number is not None and self.lap_number < lap_number:
            self.total_laps += 1
            log.info('lap %d complete with score of %f', self.total_laps, self.score.total)
            self.prev_lap_score = self.score.total
            if self.should_benchmark:
                self.log_benchmark_trial()
                if len(self.trial_scores) >= 50:
                    self.done_benchmarking = True
            done = True  # One lap per episode
            self.log_up_time()
        self.lap_number = lap_number
        return done

    def get_reward(self, obz, now):
        reward = 0
        if obz:
            if self.prev_step_time is not None:
                step_time = now - self.prev_step_time
            else:
                step_time = None
            now = time.time()
            time_penalty = now - self.score.start_time - self.score.episode_time
            self.score.episode_time = now - self.score.start_time
            if self.score.episode_time < 2.5:
                # Give time to get on track after spawn
                reward = 0
            else:
                progress_reward = self.get_progress_reward(obz, step_time)
                gforce_penalty = self.get_gforce_penalty(obz, step_time)
                lane_deviation_penalty = self.get_lane_deviation_penalty(obz, step_time)
                reward += (progress_reward - gforce_penalty - lane_deviation_penalty - time_penalty)

            self.score.total += reward
            self.display_stats['time']['value'] = self.score.episode_time
            self.display_stats['time']['total'] = self.score.episode_time
            self.display_stats['episode score']['value'] = self.score.total
            self.display_stats['episode score']['total'] = self.score.total

            log.debug('reward %r', reward)
            log.debug('score %r', self.score.total)

        return reward

    def log_up_time(self):
        log.info('up for %r' % arrow.get(time.time()).humanize(other=arrow.get(self.start_time), only_distance=True))

    def get_lane_deviation_penalty(self, obz, time_passed):
        lane_deviation_penalty = 0
        if 'distance_to_center_of_lane' in obz:
            lane_deviation_penalty = DeepDriveRewardCalculator.get_lane_deviation_penalty(
                obz['distance_to_center_of_lane'], time_passed)
        self.score.lane_deviation_penalty += lane_deviation_penalty
        self.display_stats['lane deviation penalty']['value'] = -self.score.lane_deviation_penalty
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
        self.score.gforce_penalty += gforce_penalty
        self.display_stats['gforce penalty']['value'] = -self.score.gforce_penalty
        self.display_stats['gforce penalty']['total'] = -self.score.gforce_penalty
        return gforce_penalty

    def get_progress_reward(self, obz, time_passed):
        progress_reward = 0
        if 'distance_along_route' in obz:
            dist = obz['distance_along_route'] - self.start_distance_along_route
            progress = dist - self.distance_along_route
            self.distance_along_route = dist
            progress_reward = DeepDriveRewardCalculator.get_progress_reward(progress, time_passed)
        self.display_stats['lap progress']['total'] = self.distance_along_route / 2736.7
        self.display_stats['lap progress']['value'] = self.display_stats['lap progress']['total']
        self.display_stats['episode #']['total'] = self.total_laps
        self.display_stats['episode #']['value'] = self.total_laps
        self.score.progress_reward += progress_reward
        return progress_reward

    def is_stuck(self, obz):
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
            time_crawling = time.time() - self.last_forward_progress_time
            portion_crawling = self.steps_crawling_with_throttle_on / max(1, self.steps_crawling)
            if self.steps_crawling_with_throttle_on > 20 and time_crawling > 10 and portion_crawling > 0.8:
                log.warn('No progress made while throttle on - assuming stuck and ending episode. steps crawling: %r, '
                         'steps crawling with throttle on: %r, time crawling: %r',
                         self.steps_crawling, self.steps_crawling_with_throttle_on, time_crawling)
                self.set_forward_progress()
                if self.should_benchmark:
                    self.score.got_stuck = True
                    self.log_benchmark_trial()
                ret = True
        else:
            self.set_forward_progress()
        return ret

    def log_benchmark_trial(self):
        self.score.end_time = time.time()
        self.score.episode_time = self.score.end_time - self.score.start_time
        log.info('episode time %r', self.score.episode_time)
        self.trial_scores.append(self.score)
        totals = [s.total for s in self.trial_scores]
        median = np.median(totals)
        average = np.mean(totals)
        high = max(totals)
        low = min(totals)
        std = np.std(totals)
        log.info('benchmark lap #%d score: %f - average: %f', len(self.trial_scores), self.score.total, average)
        filename = os.path.join(c.BENCHMARK_DIR, '%s_%s.csv' % (self.experiment, c.DATE_STR))
        with open(filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for i, score in enumerate(self.trial_scores):
                if i == 0:
                    writer.writerow(['episode #', 'score', 'progress reward', 'lane deviation penalty',
                                     'gforce penalty', 'got stuck', 'start', 'end', 'lap time'])
                writer.writerow([i + 1, score.total,
                                 score.progress_reward, score.lane_deviation_penalty,
                                 score.gforce_penalty, score.got_stuck, str(arrow.get(score.start_time).to('local')),
                                 str(arrow.get(score.end_time).to('local')),
                                 score.episode_time])
            writer.writerow([])
            writer.writerow(['median score', median])
            writer.writerow(['avg score', average])
            writer.writerow(['std', std])
            writer.writerow(['high score', high])
            writer.writerow(['low score', low])

        log.info('median score %r', median)
        log.info('avg score %r', average)
        log.info('std %r', std)
        log.info('high score %r', high)
        log.info('low score %r', low)
        log.info('progress_reward %r', self.score.progress_reward)
        log.info('lane_deviation_penalty %r', self.score.lane_deviation_penalty)
        log.info('gforce_penalty %r', self.score.gforce_penalty)
        log.info('episode_time %r', self.score.episode_time)
        log.info('wrote results to %s', os.path.normpath(filename))

    def release_agent_control(self):
        self.has_control = deepdrive_client.release_agent_control(self.client_id) is not None

    def request_agent_control(self):
        self.has_control = deepdrive_client.request_agent_control(self.client_id) == 1

    # noinspection PyAttributeOutsideInit
    def set_forward_progress(self):
        self.last_forward_progress_time = time.time()
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
        log.info('Reset complete')

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
        if self.dashboard_queue is not None:
            self.dashboard_queue.put({'should_stop': True})
            self.dashboard_queue.close()
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
        if self.prev_observation is not None:
            if self.one_frame_render:
                for camera in self.prev_observation['cameras']:
                    utils.show_camera(camera['image'], camera['depth'])
            elif self.pyglet_render and pyglet is not None and self.pyglet_queue is not None:
                self.pyglet_queue.put(self.prev_observation['cameras'])

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
        try:
            obz = deepdrive_capture.step()
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
        deepdrive_client.set_control_values(self.client_id, steering=action.steering, throttle=action.throttle,
                                            brake=action.brake, handbrake=action.handbrake)

    def connect(self, cameras=None, render=False):
        def _connect():
            try:
                self.connection_props = deepdrive_client.create('127.0.0.1', 9876)
                if isinstance(self.connection_props, int):
                    raise Exception('You have an old version of the deepdrive client - try uninstalling and reinstalling with pip')
                if not self.connection_props or not self.connection_props['max_capture_resolution']:
                    # Try again
                    return
                self.client_id = self.connection_props['client_id']
                server_version = semvar(self.connection_props['server_protocol_version']).version
                # TODO: For dev, store hash of .cpp and .h files on extension build inside VERSION_DEV, then when
                #   connecting, compute same hash and compare. (Need to figure out what to do on dev packaged version as
                #   files may change - maybe ignore as it's uncommon).
                #   Currently, we timestamp the build, and set that as the version in the extension. This is fine unless
                #   you change shared code and build the extension only, then the versions won't change, and you could
                #   see incompatibilities.

                if semvar(self.client_version).version[:2] != server_version[:2]:
                    raise RuntimeError('Server and client major/minor version do not match - server is %s and client is %s' %
                                       (server_version, self.client_version))

            except deepdrive_client.time_out:
                _connect()

        _connect()
        cxn_attempts = 0
        max_cxn_attempts = 10
        while not self.connection_props:
            cxn_attempts += 1
            sleep = cxn_attempts + random.random() * 2  # splay to avoid thundering herd
            log.warning('Connection to environment failed, retry (%d/%d) in %d seconds',
                        cxn_attempts, max_cxn_attempts, round(sleep, 0))
            time.sleep(sleep)
            _connect()
            if cxn_attempts >= max_cxn_attempts:
                raise RuntimeError('Could not connect to the environment')

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
        if render:
            self.init_pyglet(cameras)

        self._perform_first_step()
        self.has_control = False

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
            time.sleep(0.25)
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

    def _init_action_space(self):
        steering_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        throttle_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        brake_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        handbrake_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        is_game_driving_space = spaces.Discrete(2)
        action_space = spaces.Tuple(
            (steering_space, throttle_space, brake_space, handbrake_space, is_game_driving_space))
        return action_space

    def _init_observation_space(self):
        obz_spaces = []
        for camera in self.cameras:
            obz_spaces.append(spaces.Box(low=0, high=255, shape=(camera['capture_width'], camera['capture_height']),
                                         dtype=np.uint8))
        observation_space = spaces.Tuple(tuple(obz_spaces))
        self.observation_space = observation_space
        return observation_space

    def change_has_control(self, has_control):
        if has_control:
            self.request_agent_control()
        else:
            self.release_agent_control()


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
        if gforces > 0.5:
            # https://www.quora.com/Hyperloop-What-is-a-physically-comfortable-rate-of-acceleration-for-human-beings
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
    def get_progress_reward(progress, time_passed):
        if time_passed is not None:
            step_velocity = progress / time_passed
            if step_velocity < -400 * 100:
                # Lap completed
                # TODO: Read the lap length on reset and
                log.debug('assuming lap complete, progress zero')
                progress = 0
        progress_reward = progress / 100.  # cm=>meters
        balance_coeff = 1.0
        progress_reward *= balance_coeff
        progress_reward = DeepDriveRewardCalculator.clip(progress_reward)
        return progress_reward

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
