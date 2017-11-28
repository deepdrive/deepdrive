import csv
import deepdrive as deepdrive_capture
import deepdrive_control
import platform
import threading
import time
from collections import deque, OrderedDict

import arrow
import gym
from gym import spaces, utils
from gym.utils import seeding

import utils
from config import *
from utils import obj2dict

# TODO: Set log level based on verbosity arg
log = utils.get_log(__name__)

if platform.system() == 'Linux':
    SHARED_CAPTURE_MEM_NAME = '/tmp/deepdrive_shared_memory'  # TODO: Change to deepdrive_capture
    SHARED_CONTROL_MEM_NAME = '/tmp/deepdrive_control'
elif platform.system() == 'Windows':
    SHARED_CAPTURE_MEM_NAME = 'Local\DeepDriveCapture'
    SHARED_CONTROL_MEM_NAME = 'Local\DeepDriveControl'

SHARED_CAPTURE_MEM_SIZE = 157286400
SHARED_CONTROL_MEM_SIZE = 1048580
SPEED_LIMIT_KPH = 64.

log.info('Starting deepdrive, enter "s" or "stop" to stop')


class Score(object):
    total = 0
    gforce_penalty = 0
    speed_reward = 0
    lane_deviation_penalty = 0
    progress_reward = 0
    got_stuck = False

    def __init__(self):
        self.start_time = time.time()
        self.end_time = None


class DeepDriveRewardCalculator(object):
    @staticmethod
    def get_speed_reward(cmps, time_passed):
        """
        Incentivize going quickly while remaining under the speed limit.
        :param cmps: speed in cm / s
        :param time_passed: time passed since previous speed reward (allows running at variable frame rates while 
        still receiving consistent rewards)
        :return: positive or negative real valued speed reward on meter scale
        """
        speed_kph = cmps * 3600. / 100. / 1000.  # cm/s=>kph
        balance_coeff = 2. / 10.
        speed_delta = speed_kph - SPEED_LIMIT_KPH
        if speed_delta > 4:
            # too fast
            speed_reward = -1 * balance_coeff * speed_kph * time_passed * speed_delta ** 2  # squared to outweigh advantage of speeding
        else:
            # incentivize timeliness
            speed_reward = balance_coeff * time_passed * speed_kph

            # No slow penalty as progress already incentivizes this (plus we'll need to stop at some points anyway)
        return speed_reward


class DeepDriveEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cameras, preprocess_with_tensorflow=False):
        self.observation_space = self._init_observation_space(cameras)
        self.action_space = self._init_action_space()
        self.preprocess_with_tensorflow = preprocess_with_tensorflow
        self.sess = None
        self.prev_observation = None
        self.start_time = time.time()
        self.control = deepdrive_control.DeepDriveControl()
        self.step_num = 0
        self.prev_step_time = None
        self.display_stats = OrderedDict()
        self.display_stats['g-forces']                      = {'value': 0, 'ymin': 0,     'ymax': 3,  'units': ''}
        # self.display_stats['lane deviation']                = {'value': 0, 'ymin': 0,     'ymax': 1000,  'units': 'cm'}
        self.display_stats['gforce penalty']                = {'value': 0, 'ymin': 0,     'ymax': 5,   'units': ''}
        self.display_stats['lane deviation penalty']        = {'value': 0, 'ymin': 0,     'ymax': 40,   'units': ''}
        self.display_stats['speed reward']                  = {'value': 0, 'ymin': 0,     'ymax': 5,   'units': ''}
        self.display_stats['progress reward']               = {'value': 0, 'ymin': 0,     'ymax': 5,   'units': ''}
        self.display_stats['reward']                        = {'value': 0, 'ymin': -20,   'ymax': 20,    'units': ''}
        self.display_stats['total score']                   = {'value': 0, 'ymin': -500,  'ymax': 10000, 'units': ''}

        self.reset_capture()
        self.reset_control()

        # collision detection  # TODO: Remove in favor of in-game detection
        self.reset_forward_progress()

        self.distance_along_route = 0
        self.start_distance_along_route = 0

        # reward
        self.score = Score()

        self.lap_number = None
        self.prev_lap_score = 0

        # benchmarking - carries over across resets
        self.should_benchmark = False
        self.done_benchmarking = False
        self.trial_scores = []

    def init_benchmarking(self):
        self.should_benchmark = True
        os.makedirs(BENCHMARK_DIR, exist_ok=True)

    def start_dashboard(self):
        if utils.is_debugging():
            # TODO: Deal with plot UI not being in the main thread somehow - (move to browser?)
            log.error('dashboard not supported in debug mode')
            return
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        display_stats = self.display_stats

        def ui_thread():
            plt.figure(0)
            for i, (stat_name, stat) in enumerate(display_stats.items()):
                stat = display_stats[stat_name]
                stat_label_subplot = plt.subplot2grid((len(display_stats), 3), (i, 0))
                stat_value_subplot = plt.subplot2grid((len(display_stats), 3), (i, 1))
                stat_graph_subplot = plt.subplot2grid((len(display_stats), 3), (i, 2))
                txt_label = stat_label_subplot.text(0.5, 0.5, stat_name, fontsize=12, va="center", ha="center")
                txt_value = stat_value_subplot.text(0.5, 0.5, '', fontsize=12, va="center", ha="center")
                stat['txt_value'] = txt_value
                stat_graph_subplot.set_xlim([0, 200])
                stat_graph_subplot.set_ylim([stat['ymin'], stat['ymax']])
                stat['line'], = stat_graph_subplot.plot([], [])
                stat_label_subplot.axis('off')
                stat_value_subplot.axis('off')
                stat['y_list'] = deque([-1] * 400)
                stat['x_list'] = deque(np.linspace(200, 0, num=400))

            fig = plt.gcf()
            fig.set_size_inches(7.5, len(display_stats) * 1.75)
            fig.canvas.set_window_title('DeepDrive score')
            step = [0]

            def init():
                lines = []
                for stat_name in display_stats:
                    stat = display_stats[stat_name]
                    stat['line'].set_data([], [])
                    lines.append(stat['line'])
                return lines

            def animate(_i):
                step[0] += 1
                lines = []
                for stat_name in display_stats:
                    stat = display_stats[stat_name]
                    val = stat['value']
                    stat['txt_value'].set_text(str(round(val, 2)) + stat['units'])
                    stat['y_list'].pop()
                    stat['y_list'].appendleft(val)
                    stat['line'].set_data(stat['x_list'], stat['y_list'])
                    lines.append(stat['line'])
                plt.draw()
                return lines

            # noinspection PyUnusedLocal
            # TODO: Add blit=True and deal with updating the text if performance becomes unacceptable
            _anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=100)

            plt.show()

        thread = threading.Thread(target=ui_thread)
        thread.daemon = True
        thread.start()

    def set_tf_session(self, session):
        self.sess = session

    def _step(self, action):
        self.send_control(action)
        obz = self.get_observation()
        reward = self.get_reward(obz)
        if self.is_stuck(obz):  # TODO: derive this from collision, time elapsed, and distance as well
            done = True
            reward -= -10000  # reward is in scale of meters
        else:
            done = False
        info = {}
        self.step_num += 1
        return obz, reward, done, info

    def get_reward(self, obz):
        reward = 0

        if obz and time.time() - self.start_time > 6:
            now = time.time()
            if self.prev_step_time is not None:
                time_passed = now - self.prev_step_time
            else:
                time_passed = None

            progress_reward = self.get_progress_reward(obz, time_passed)
            gforce_penalty = self.get_gforce_penalty(obz, time_passed)
            lane_deviation_penalty = self.get_lane_deviation_penalty(obz, time_passed)
            speed_reward = self.get_speed_reward(obz, time_passed)
            reward += (progress_reward + speed_reward - gforce_penalty - lane_deviation_penalty)

            self.score.total += reward
            self.display_stats['reward']['value'] = reward
            self.display_stats['total score']['value'] = self.score.total

            log.debug('reward %r', reward)
            log.debug('score %r', self.score.total)

            lap_number = obz.get('lap_number')
            if lap_number is not None and self.lap_number is not None and self.lap_number < lap_number:
                lap_score = self.score.total - self.prev_lap_score
                log.info('lap %d complete with score of %f, speed reward', lap_number, lap_score)
                self.prev_lap_score = self.score.total
                if self.should_benchmark:
                    self.log_benchmark_trial()
                    if len(self.trial_scores) == 1000:
                        self.done_benchmarking = True
                    self.reset()
                self.log_up_time()
            self.lap_number = lap_number
            self.prev_step_time = now

        return reward

    def log_up_time(self):
        log.info('up for %r' % arrow.get(time.time()).humanize(other=arrow.get(self.start_time), only_distance=True))

    def get_speed_reward(self, obz, time_passed):
        speed_reward = 0
        if 'speed' in obz:
            speed = obz['speed']
            if time_passed is not None:
                speed_reward = DeepDriveRewardCalculator.get_speed_reward(speed, time_passed)
                self.display_stats['speed reward']['value'] = speed_reward
        self.score.speed_reward += speed_reward
        return speed_reward

    def get_lane_deviation_penalty(self, obz, time_passed):
        lane_deviation_penalty = 0
        if 'distance_to_center_of_lane' in obz:
            lane_deviation = obz['distance_to_center_of_lane']
            if time_passed is not None and lane_deviation > 200:  # Tuned for Canyons spline - change for future maps
                lane_deviation_coeff = 0.1
                lane_deviation_penalty =  lane_deviation_coeff * time_passed * lane_deviation ** 2 / 100.
            # self.display_stats['lane deviation']['value'] = lane_deviation
            log.debug('distance_to_center_of_lane %r', lane_deviation)
        self.display_stats['lane deviation penalty']['value'] = lane_deviation_penalty
        self.score.lane_deviation_penalty += lane_deviation_penalty
        return lane_deviation_penalty

    def get_gforce_penalty(self, obz, time_passed):
        gforce_penalty = 0
        if 'acceleration' in obz:
            if time_passed is not None:
                a = obz['acceleration']
                gforces = np.sqrt(a.dot(a)) / 980  # g = 980 cm/s**2
                self.display_stats['g-forces']['value'] = gforces
                log.debug('gforces %r', gforces)

                if gforces > 0.5:
                    # https://www.quora.com/Hyperloop-What-is-a-physically-comfortable-rate-of-acceleration-for-human-beings
                    time_weighted_gs = time_passed * gforces
                    time_weighted_gs = min(time_weighted_gs,
                                           5)  # Don't allow a large frame skip to ruin the approximation
                    balance_coeff = 24  # 24 meters of reward every second you do this
                    gforce_penalty = time_weighted_gs * balance_coeff
                    log.debug('accumulated_gforce %r', time_weighted_gs)
                    log.debug('gforce_penalty %r', gforce_penalty)

        self.display_stats['gforce penalty']['value'] = gforce_penalty
        self.score.gforce_penalty += gforce_penalty
        return gforce_penalty

    def get_progress_reward(self, obz, time_passed):
        progress_reward = 0
        if 'distance_along_route' in obz:
            dist = obz['distance_along_route'] - self.start_distance_along_route
            progress = dist - self.distance_along_route
            self.distance_along_route = dist
            if time_passed is not None:
                step_velocity = progress / time_passed
                if step_velocity < -400 * 100:
                    # Lap completed
                    # TODO: Read the lap length on reset and
                    log.info('assuming lap complete')
                    progress = 0
            progress_reward = progress / 100.  # cm=>meters
            balance_coeff = 1.0
            progress_reward *= balance_coeff
        self.display_stats['progress reward']['value'] = progress_reward
        self.score.progress_reward += progress_reward
        return progress_reward

    def is_stuck(self, obz):
        # TODO: Get this from the game instead

        if 'TEST_BENCHMARK_WRITE' in os.environ:
            self.score.got_stuck = True
            self.log_benchmark_trial()
            return True

        if obz is None:
            return False
        if obz['speed'] < 100:  # cm/s
            self.steps_crawling += 1
            if obz['throttle'] > 0:
                self.steps_crawling_with_throttle_on += 1
            if time.time() - self.last_forward_progress_time > 1 and \
                    (self.steps_crawling_with_throttle_on / float(self.steps_crawling)) > 0.8:
                self.reset_forward_progress()
                if self.should_benchmark:
                    self.score.got_stuck = True
                    self.log_benchmark_trial()
                return True
        else:
            self.reset_forward_progress()
        return False

    def log_benchmark_trial(self):
        self.score.end_time = time.time()
        self.trial_scores.append(self.score)
        totals = [s.total for s in self.trial_scores]
        median = np.median(totals)
        average = np.mean(totals)
        high = max(totals)
        low = min(totals)
        std = np.std(totals)
        log.info('benchmark lap #%d score: %f - high score: %f', len(self.trial_scores), self.score.total, median)
        filename = os.path.join(BENCHMARK_DIR, DATE_STR + '.csv')
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i, score in enumerate(self.trial_scores):
                if i == 0:
                    writer.writerow(['lap #', 'total score', 'speed reward', 'progress reward', 'lane deviation penalty',
                                     'gforce penalty', 'got stuck', 'start', 'end', 'lap time'])
                writer.writerow([i + 1, score.total, score.speed_reward,
                                 score.progress_reward, score.lane_deviation_penalty,
                                 score.gforce_penalty, score.got_stuck, str(arrow.get(score.start_time).to('local')),
                                 str(arrow.get(score.end_time).to('local')),
                                 score.end_time - score.start_time])
            writer.writerow([])
            writer.writerow(['median score', median])
            writer.writerow(['avg score', average])
            writer.writerow(['std', std])
            writer.writerow(['high score', high])
            writer.writerow(['low score', low])
        log.info('wrote results to %r', filename)

    def get_action_array(self, steering=0, throttle=0, brake=0, handbrake=0, is_game_driving=False, should_reset=False):
        log.debug('steering %f', steering)
        action = [np.array([steering]),
                  np.array([throttle]),
                  np.array([brake]),
                  np.array([handbrake]),
                  is_game_driving,
                  should_reset]
        return action

    def reset_forward_progress(self):
        self.last_forward_progress_time = time.time()
        self.steps_crawling_with_throttle_on = 0
        self.steps_crawling = 0

    def _reset(self):
        self.prev_observation = None
        done = False
        i = 0
        while not done:
            self.send_control(self.get_action_array(should_reset=True, is_game_driving=False))
            obz = self.get_observation()
            if obz and obz['distance_along_route'] < 20 * 100:
                self.start_distance_along_route = obz['distance_along_route']
                log.info('Finished resetting')
                done = True
            log.info('Waiting for reset...')  # TODO: Add a transactional comms channel using sockets as shared mem is bad for this type of signalling
            time.sleep(1)
            i += 1
            if i > 5:
                log.error('Unable to reset, try restarting the sim.')
                raise Exception('Unable to reset, try restarting the sim.')
        self.send_control(self.get_action_array(is_game_driving=True))
        self.step_num = 0
        self.distance_along_route = 0
        self.start_distance_along_route = 0
        self.prev_step_time = None
        self.lap_number = None
        self.prev_lap_score = 0
        self.score = Score()
        self.start_time = time.time()

    def _close(self):
        log.debug('closing connection to deepdrive')
        deepdrive_capture.close()
        deepdrive_control.close()
        if self.sess:
            self.sess.close()

    def _render(self, mode='human', close=False):
        # TODO: Implement proper render - this is really only good for one frame - Could use our OpenGLUT viewer (on raw images) for this or PyGame on preprocessed images
        # if self._previous_observation is not None:
        #     for camera in self._previous_observation['cameras']:
        #         utils.show_camera(camera['image'], camera['depth'])
        pass

    def _seed(self, seed=None):
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
        for camera in cameras[:1]:  # Ignore all but the first camera  TODO: Don't hardcode this!
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

    def send_control(self, action):
        self.control.steering = action[0][0]
        self.control.throttle = action[1][0]
        self.control.brake = action[2][0]
        self.control.handbrake = action[3][0]
        self.control.is_game_driving = action[4]
        self.control.should_reset = action[5]
        deepdrive_control.send_control(self.control)

    def reset_capture(self):
        # TODO: Establish some handshake so we don't hardcode size here and in Unreal project
        if deepdrive_capture.reset(SHARED_CAPTURE_MEM_NAME, SHARED_CAPTURE_MEM_SIZE):
            log.info('Connected to deepdrive shared capture memory')
        else:
            log.error('Could not connect to deepdrive capture memory at %s', SHARED_CAPTURE_MEM_NAME)
            self.raise_connect_fail()

    @staticmethod
    def raise_connect_fail():
        raise Exception('\n\n\n'
                        '**********************************************************************\n'
                        '**********************************************************************\n'
                        '****                                                              ****\n\n'
                        '|   Could not connect to the environment. Is the simulator running?  |\n\n'
                        '****                                                              ****\n'
                        '**********************************************************************\n'
                        '**********************************************************************\n\n')

    def reset_control(self):
        # TODO: Establish some handshake so we don't hardcode size here and in Unreal project
        if deepdrive_control.reset(SHARED_CONTROL_MEM_NAME, SHARED_CONTROL_MEM_SIZE):
            log.info('Connected to deepdrive shared control memory')
        else:
            log.error('Could not connect to deepdrive control memory at %s', SHARED_CONTROL_MEM_NAME)
            self.raise_connect_fail()

    def _init_action_space(self):
        steering_space = spaces.Box(low=-1, high=1, shape=1)
        throttle_space = spaces.Box(low=-1, high=1, shape=1)
        brake_space = spaces.Box(low=0, high=1, shape=1)
        handbrake_space = spaces.Box(low=0, high=1, shape=1)
        is_game_driving_space = spaces.Discrete(2)
        action_space = spaces.Tuple(
            (steering_space, throttle_space, brake_space, handbrake_space, is_game_driving_space))
        return action_space

    def _init_observation_space(self, cameras):
        obz_spaces = []
        for camera in cameras:
            obz_spaces.append(spaces.Box(low=0, high=255, shape=camera['img_shape']))
        observation_space = spaces.Tuple(tuple(obz_spaces))
        return observation_space
