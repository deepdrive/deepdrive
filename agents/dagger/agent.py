import copy
import json
import os
import time
from datetime import datetime
import math
import glob

import tensorflow as tf
import numpy as np
from simple_pid import PID

import config as c
import sim
import utils
from agents.common import get_throttle
from agents.dagger import net
from agents.dagger.action_jitterer import ActionJitterer, JitterState
from sim.driving_style import DrivingStyle
from sim.action import Action
from sim import world
from agents.dagger.net import AlexNet, MobileNetV2
from utils import save_hdf5, download
import logs

log = logs.get_log(__name__)


TEST_SAVE_IMAGE = False

TARGET_MPH = 25
TARGET_MPS = TARGET_MPH / 2.237
TARGET_MPS_TEST = 75 * TARGET_MPS


class Agent(object):
    def __init__(self, tf_session, should_jitter_actions=True,
                 should_record=False, net_path=None, use_frozen_net=False, random_action_count=0,
                 non_random_action_count=5, path_follower=False, recording_dir=c.RECORDING_DIR,
                 output_last_hidden=False,
                 net_name=net.ALEXNET_NAME, driving_style=DrivingStyle.NORMAL):
        np.random.seed(c.RNG_SEED)
        self.previous_action = None
        self.previous_net_out = None
        self.step = 0
        self.net_name = net_name
        self.driving_style = driving_style

        # State for toggling random actions
        self.should_jitter_actions = should_jitter_actions
        self.episode_action_count = 0
        self.recorded_obz_count = 0
        self.num_saved_observations = 0
        self.path_follower_mode = path_follower
        self.recording_dir = recording_dir
        self.output_last_hidden = output_last_hidden

        # Recording state
        self.should_record = should_record
        self.sess_dir = c.HDF5_SESSION_DIR
        self.obz_recording = []
        self.skipped_first_corrective_action = False

        if should_jitter_actions:
            log.info('Mixing in random actions to increase data diversity (these are not recorded).')
        if should_record:
            log.info('Recording driving data to %s', self.sess_dir)

        # Net
        self.sess = tf_session
        self.use_frozen_net = use_frozen_net
        if net_path is not None:
            if net_name == net.ALEXNET_NAME:
                input_shape = net.ALEXNET_IMAGE_SHAPE
            elif net_name == net.MOBILENET_V2_NAME:
                input_shape = net.MOBILENET_V2_IMAGE_SHAPE
            else:
                raise NotImplementedError(net_name + ' not recognized')
            self.load_net(net_path, use_frozen_net, input_shape)
        else:
            self.net = None
            self.sess = None

        self.throttle_pid = PID(0.3, 0.05, 0.4)
        self.throttle_pid.output_limits = (-1, 1)

        self.jitterer = ActionJitterer()

    def act(self, obz, reward, done, episode_time=None):
        net_out = None
        if obz:
            try:
                log.debug('steering %r', obz['steering'])
                log.debug('throttle %r', obz['throttle'])
            except TypeError as e:
                log.error('Could not parse this observation %r', obz)
                raise
            obz = self.preprocess_obz(obz)

        if self.should_jitter_actions:
            if episode_time is None or episode_time < 10:
                # Hold off a bit at start of episode
                action = Action(has_control=False)
            else:
                if okay_to_jitter_actions(obz):
                    action = self.jitter_action(obz)
                else:
                    action = Action(has_control=False)

        elif self.net is not None:
            if not obz or not obz['cameras']:
                net_out = None
            else:
                image = obz['cameras'][0]['image']
                net_out = self.get_net_out(image)
            if net_out is not None and self.output_last_hidden:
                y_hat = net_out[0]
            else:
                y_hat = net_out
            action = self.get_next_action(obz, y_hat)
        else:
            action = Action(has_control=(not self.path_follower_mode))
        self.previous_action = action
        self.step += 1
        log.debug('obz_exists? %r should_record? %r', obz is not None, self.should_record)
        if self.should_record_obz(obz):
            log.debug('Recording frame')
            self.obz_recording.append(obz)
            if TEST_SAVE_IMAGE:
                utils.save_camera(obz['cameras'][0]['image'], obz['cameras'][0]['depth'],
                                  self.sess_dir, 'screenshot_' + str(self.step).zfill(10))
                input('continue?')
            self.recorded_obz_count += 1
            if self.recorded_obz_count % 100 == 0:
                log.info('%d recorded observations', self.recorded_obz_count)

        else:
            log.debug('Not recording frame.')

        self.maybe_save()
        action = action.as_gym()
        return action, net_out

    def should_record_obz(self, obz):
        if not obz:
            return False
        if not self.should_record:
            return False
        if not self.should_jitter_actions:
            return self.should_record
        else:
            is_game_driving = self.get_is_game_driving(obz)
            safe_action = is_game_driving and not self.jitterer.perform_random_actions
            if safe_action:
                # TODO: Fix race condition in sim and remove skipped_first_corrective_action guard
                if self.skipped_first_corrective_action:
                    return True
                else:
                    self.skipped_first_corrective_action = True
                    return False
            else:
                self.skipped_first_corrective_action = False
                return False

    def get_is_game_driving(self, obz):
        if not obz:
            log.warn('Observation not set, assuming game not driving to prevent recording bad actions')
            return False
        return obz['is_game_driving'] == 1

    def get_next_action(self, obz, net_out):
        log.debug('getting next action')
        if net_out is None:
            log.debug('net out is None')
            return self.previous_action or Action()

        net_out = net_out[0]  # We currently only have one environment

        desired_spin, desired_direction, desired_speed, desired_speed_change, desired_steering, desired_throttle = \
            net_out

        desired_spin = desired_spin * c.SPIN_NORMALIZATION_FACTOR
        desired_speed = desired_speed * c.SPEED_NORMALIZATION_FACTOR
        desired_speed_change = desired_speed_change * c.SPEED_NORMALIZATION_FACTOR

        log.debug('desired_steering %f', desired_steering)
        log.debug('desired_throttle %f', desired_throttle)

        log.debug('desired_direction %f', desired_direction)
        log.debug('desired_speed %f', desired_speed)
        log.debug('desired_speed_change %f', desired_speed_change)
        log.debug('desired_throttle %f', desired_throttle)
        log.debug('desired_spin %f', desired_spin)

        actual_speed = obz['speed']
        log.debug('actual_speed %f', actual_speed)
        log.debug('desired_speed %f', desired_speed)

        # max_meters_per_sec = 8

        if isinstance(self.net, MobileNetV2):
            # target_speed = desired_speed
            # desired_throttle = abs(target_speed / max(actual_speed, 1e-3))

            # if actual_speed > 0.8 * (max_meters_per_sec * 100):
            #     desired_speed *= 0.8

            # TODO: Support different driving styles

            # desired_throttle = get_throttle(actual_speed, desired_speed * 0.48)

            pid_throttle = self.get_target_throttle(obz)

            desired_throttle = pid_throttle  # min(max(desired_throttle, 0.), pid_throttle)

            # if self.previous_net_out:
            #     desired_throttle = 0.2 * self.previous_action.throttle + 0.8 * desired_throttle
            # else:
            #     desired_throttle = desired_throttle * 0.95
            #
            # desired_throttle *= 0.95
        else:
            # AlexNet

            if self.driving_style == DrivingStyle.CRUISING:
                target_speed = 8 * 100
            elif self.driving_style == DrivingStyle.NORMAL:
                target_speed = 9 * 100
            elif self.driving_style == DrivingStyle.LATE:
                target_speed = 10 * 100
            else:
                raise NotImplementedError('Driving style not supported')

            # AlexNet overfit on speed, plus it's nice to be able to change it,
            # so we just ignore output speed of net
            desired_throttle = get_throttle(actual_speed, target_speed)
        log.debug('actual_speed %r' % actual_speed)

        # log.info('desired_steering %f', desired_steering)
        # log.info('desired_throttle %f', desired_throttle)
        if self.previous_action:
            smoothed_steering = 0.2 * self.previous_action.steering + 0.5 * desired_steering
        else:
            smoothed_steering = desired_steering * 0.7

        # desired_throttle = desired_throttle * 1.1

        action = Action(smoothed_steering, desired_throttle)
        return action

    def maybe_save(self):
        # TODO: Move recording to env
        if (
            self.should_record and self.recorded_obz_count % c.FRAMES_PER_HDF5_FILE == 0 and
            self.recorded_obz_count != 0 and self.num_saved_observations < self.recorded_obz_count
           ):
            self.save_recordings()

    def save_recordings(self):
        filename = os.path.join(self.sess_dir, '%s.hdf5' %
                                str(self.recorded_obz_count // c.FRAMES_PER_HDF5_FILE).zfill(10))
        save_hdf5(self.obz_recording, filename=filename, background=False)
        log.info('Flushing output data')
        self.obz_recording = []
        self.num_saved_observations = self.recorded_obz_count

    def jitter_action(self, obz):
        """Reduce sampling error by randomly exploring space around non-random agent's trajectory with occasional
            random actions"""
        state = self.jitterer.advance()
        if state is JitterState.SWITCH_TO_RAND:
            steering = np.random.uniform(-0.5, 0.5, 1)[0]  # Going too large here gets us stuck
            throttle = self.get_target_throttle(obz) * 0.5  # Slow down a bit so we don't crash before recovering
            has_control = True
            log.debug('random steering %f', steering)
        elif state is JitterState.SWITCH_TO_NONRAND:
            has_control = False
            steering = throttle = 0  # Has no effect
        elif state is JitterState.MAINTAIN:
            if self.previous_action is None:
                has_control = False
                steering = throttle = 0  # Has no effect
                log.warning('Previous action none')
            else:
                steering = self.previous_action.steering
                throttle = self.get_target_throttle(obz)
                has_control = self.previous_action.has_control
        else:
            raise ValueError('Unexpected action jitter state')
        action = Action(steering, throttle, has_control=has_control)
        if not has_control:
            # TODO: Move setpoint to env
            world.set_ego_mph(TARGET_MPH, TARGET_MPH)
        return action

    def get_target_throttle(self, obz):
        if obz and 'speed' in obz:
            actual_speed = obz['speed']
        else:
            actual_speed = TARGET_MPS

        pid = self.throttle_pid
        target_cmps = TARGET_MPS * 100
        if pid.setpoint != target_cmps:
            pid.setpoint = target_cmps
        throttle = pid(actual_speed)
        if not pid.auto_mode:
            pid.auto_mode = True
        if throttle is None:
            log.warn('PID output None, setting throttle to 0.')
            throttle = 0.
        throttle = min(max(throttle, 0.), 1.)
        return throttle

    def load_net(self, net_path, is_frozen=False, image_shape=None):
        """
        Frozen nets can be generated with something like

        `python freeze_graph.py --input_graph="C:\tmp\deepdrive\tensorflow_random_action\train\graph.pbtxt" --input_checkpoint="C:\tmp\deepdrive\tensorflow_random_action\train\model.ckpt-273141" --output_graph="C:\tmp\deepdrive\tensorflow_random_action\frozen_graph.pb" --output_node_names="model/add_2"`

        where model/add_2 is the auto-generated name for self.net.p
        """
        if image_shape is None:
            raise RuntimeError('Image shape not defined')
        if is_frozen:
            # TODO: Get frozen nets working

            # We load the protobuf file from the disk and parse it to retrieve the
            # unserialized graph_def
            with tf.gfile.GFile(net_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # Then, we can use again a convenient built-in function to import a graph_def into the
            # current default Graph
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name="prefix",
                    op_dict=None,
                    producer_op_list=None
                )
            self.net = graph

        else:
            if self.net_name == net.MOBILENET_V2_NAME:
                self.net = MobileNetV2(is_training=False)
            else:
                self.net = AlexNet(is_training=False)
            saver = tf.train.Saver()
            saver.restore(self.sess, net_path)

    def close(self):
        if self.sess is not None:
            self.sess.close()

    def save_unsaved_observations(self):
        if self.should_record and self.num_saved_observations < self.recorded_obz_count:
            self.save_recordings()

    def get_net_out(self, image):
        begin = time.time()
        if self.use_frozen_net:
            out_var = 'prefix/model/add_2'
        else:
            out_var = self.net.out
        if self.output_last_hidden:
            out_var = [out_var, self.net.last_hidden]

        image = image.reshape(1, *self.net.input_image_shape)
        net_out = self.sess.run(out_var, feed_dict={
            self.net.input: image, })

        # print(net_out)
        end = time.time()
        log.debug('inference time %s', end - begin)
        return net_out

    # noinspection PyMethodMayBeStatic
    def preprocess_obz(self, obz):
        for camera in obz['cameras']:
            prepro_start = time.time()
            image = camera['image']
            image = image.astype(np.float32)
            image -= c.MEAN_PIXEL
            if isinstance(self.net, MobileNetV2):
                resize_start = time.time()
                image = utils.resize_images(self.net.input_image_shape, [image], always=True)[0]
                log.debug('resize took %fs', time.time() - resize_start)
            camera['image'] = image
            log.debug('prepro took %fs',  time.time() - prepro_start)
        return obz

    def reset(self):
        self.jitterer.reset()
        self.throttle_pid.auto_mode = False


def create_artifacts_inventory(gist_url: str, hdf5_dir: str, episodes_file: str, summary_file: str,
                               mp4_file: str):

    # TODO: Add list of artifacts results file with:
    filename = os.path.join(c.RESULTS_DIR, 'artifact-inventory.json')
    with open(filename, 'w') as out_file:
        observations: list = glob.glob(hdf5_dir + '/*.hdf5')
        data = {'artifacts': {
            'mp4': mp4_file,
            'gist': gist_url,
            'performance_summary': summary_file,
            'episodes': episodes_file,
            'observations': observations,
        }}
        json.dump(data, out_file, indent=2)
    log.info('Wrote artifacts inventory to %s' % filename)

    # TODO: Upload to YouTube on pull request

    # TODO: Save a description file with the episode score summary, gist link, and s3 link


def run(experiment, env_id='Deepdrive-v0', should_record=False, net_path=None, should_benchmark=True,
        run_baseline_agent=False, run_mnet2_baseline_agent=False, run_ppo_baseline_agent=False, camera_rigs=None,
        should_rotate_sim_types=False, should_jitter_actions=False, render=False,
        path_follower=False, fps=c.DEFAULT_FPS, net_name=net.ALEXNET_NAME, driving_style=DrivingStyle.NORMAL,
        is_sync=False, is_remote=False, recording_dir=c.RECORDING_DIR, randomize_view_mode=False,
        randomize_sun_speed=False, randomize_shadow_level=False, randomize_month=False, enable_traffic=True,
        view_mode_period=None, max_steps=None, max_episodes=1000):

    if should_record:
        path_follower = True
        randomize_sun_speed = True
        randomize_month = True

    agent, env, should_rotate_camera_rigs, start_env = \
        setup(experiment, camera_rigs, driving_style, net_name, net_path, path_follower, recording_dir,
              run_baseline_agent,
              run_mnet2_baseline_agent, run_ppo_baseline_agent, should_record,
              should_jitter_actions, env_id, render, fps, should_benchmark, is_remote, is_sync,
              enable_traffic, view_mode_period, max_steps)

    reward = 0
    episode_done = False

    def close():
        env.close()
        agent.close()

    session_done = False
    episode = 0

    # if enable_traffic:
    #     world.enable_traffic_next_reset()
    # else:
    #     world.disable_traffic_next_reset()
    # time.sleep(5)
    # env.reset()  # TODO: Remove once traffic bug is fixed
    # time.sleep(5)
    # env.reset()  # TODO: Remove once traffic bug is fixed

    try:
        while not session_done:
            if episode_done:
                obz = env.reset()
                episode_done = False
            else:
                obz = None

            domain_randomization(env, randomize_month, randomize_shadow_level,
                                 randomize_sun_speed, randomize_view_mode)

            if max_episodes is not None and episode >= (max_episodes - 1):
                session_done = True

            while not episode_done:

                act_start = time.time()
                action, net_out = agent.act(obz, reward, episode_done, env.score.episode_time)
                log.debug('act took %fs',  time.time() - act_start)

                env_step_start = time.time()
                obz, reward, episode_done, _ = env.step(action)
                log.debug('env step took %fs', time.time() - env_step_start)

                if agent.recorded_obz_count > c.MAX_RECORDED_OBSERVATIONS:
                    session_done = True

            if session_done:
                log.info('Session done')
                if c.PY_ARGS.eval_only:
                    # Keep an even number of observations in recorded datasets
                    agent.save_unsaved_observations()
                else:
                    log.info('Discarding %d observations to keep even number of frames in recorded datasets. '
                             'Pass --eval-only to save all observations.')
                if agent.recorded_obz_count > 0:
                    mp4_file = utils.hdf5_to_mp4()
                    gist_url = utils.upload_to_gist('deepdrive-results-' + c.DATE_STR,
                                                    [c.SUMMARY_CSV_FILENAME, c.EPISODES_CSV_FILENAME])
                    create_artifacts_inventory(gist_url=gist_url, hdf5_dir=c.HDF5_SESSION_DIR,
                                               episodes_file=c.EPISODES_CSV_FILENAME, summary_file=c.SUMMARY_CSV_FILENAME,
                                               mp4_file=mp4_file)
            else:
                log.info('Episode done')
                episode += 1

                agent.reset()
                if should_rotate_camera_rigs:
                    # TODO: Allow changing viewpoint as remote client
                    # TODO: Add this to domain_randomization()
                    cameras = copy.deepcopy(camera_rigs[episode % len(camera_rigs)])
                    randomize_cameras(cameras)
                    env.unwrapped.change_cameras(cameras)

    except KeyboardInterrupt:
        log.info('keyboard interrupt detected in agent, closing')
    finally:
        close()


def domain_randomization(env, randomize_month, randomize_shadow_level, randomize_sun_speed, randomize_view_mode):
    if randomize_view_mode:
        set_random_view_mode(env)
    # if randomize_sun_speed:
    #     world.randomize_sun_speed()
    # if randomize_shadow_level:
    #     graphics.randomize_shadow_level()
    # if randomize_month:
    #     world.randomize_sun_month()
    pass


def set_random_view_mode(env):
    env.unwrapped.view_mode_controller.set_random()


def setup(experiment, camera_rigs, driving_style, net_name, net_path, path_follower, recording_dir, run_baseline_agent,
          run_mnet2_baseline_agent, run_ppo_baseline_agent, should_record, should_jitter_actions,
          env_id, render, fps, should_benchmark, is_remote, is_sync, enable_traffic, view_mode_period, max_steps):
    if run_baseline_agent:
        net_path = ensure_mnet2_baseline_weights(net_path)
    elif run_mnet2_baseline_agent:
        net_path = ensure_mnet2_baseline_weights(net_path)
    elif run_ppo_baseline_agent:
        net_path = ensure_ppo_baseline_weights(net_path)

    # The following will work with 4GB vram
    if net_name == net.ALEXNET_NAME:
        per_process_gpu_memory_fraction = 0.8
    else:
        per_process_gpu_memory_fraction = 0.4
    tf_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction,
            # leave room for the game,
            # NOTE: debugging python, i.e. with PyCharm can cause OOM errors, where running will not
            allow_growth=True
        ),
    )
    sess = tf.Session(config=tf_config)
    if camera_rigs:
        cameras = camera_rigs[0]
    else:
        cameras = None
    if should_record and camera_rigs is not None and len(camera_rigs) >= 1:
        should_rotate_camera_rigs = True
        log.info('Rotating cameras while recording to encourage camera robustness')
    else:
        should_rotate_camera_rigs = False
    if should_rotate_camera_rigs:
        randomize_cameras(cameras)
    use_sim_start_command_first_lap = c.SIM_START_COMMAND is not None

    def start_env():
        return sim.start(experiment=experiment, env_id=env_id, should_benchmark=should_benchmark,
                         cameras=cameras,
                         use_sim_start_command=use_sim_start_command_first_lap, render=render, fps=fps,
                         driving_style=driving_style, is_sync=is_sync, reset_returns_zero=False,
                         is_remote_client=is_remote, enable_traffic=enable_traffic, view_mode_period=view_mode_period,
                         max_steps=max_steps)

    env = start_env()
    agent = Agent(sess,
                  should_jitter_actions=should_jitter_actions,
                  should_record=should_record, net_path=net_path, random_action_count=4, non_random_action_count=5,
                  path_follower=path_follower, net_name=net_name, driving_style=driving_style,
                  recording_dir=recording_dir)
    if net_path:
        log.info('Running tensorflow agent checkpoint: %s', net_path)
    return agent, env, should_rotate_camera_rigs, start_env


def okay_to_jitter_actions(obz):
    if obz is None:
        return False
    else:
        dnext = obz['distance_to_next_agent']
        dprev = obz['distance_to_prev_agent']
        if dnext == -1.0 and dprev == -1.0:
            return True
        if dnext < (100 * 100) or dprev < (50 * 100):
            log.info('Not okay to act randomly passing %r distance next %r distance prev %r',
                     obz['is_passing'], obz['distance_to_next_agent'], obz['distance_to_prev_agent'])
            return False
        else:
            return True


def randomize_cameras(cameras):
    for cam in cameras:
        # Add some randomness to the position (less than meter), rotation (less than degree), fov (less than degree),
        # capture height (1%), and capture width (1%)
        for i in range(len(cam['relative_rotation'])):
            cam['relative_rotation'][i] += math.radians(np.random.random())
        for i in range(len(cam['relative_position'])):
            if i == 0 or i == 2:
                # x - forward or z - up
                cam['relative_position'][i] += (np.random.random() * 100)
            elif i == 1:
                # y - right
                cam['relative_position'][i] += (np.random.random() * 100 - 50)

        cam['field_of_view'] += (np.random.random() - 0.5)
        cam['capture_height'] += round(np.random.random() * 0.01 * cam['capture_height'])
        cam['capture_width'] += round(np.random.random() * 0.01 * cam['capture_width'])


def random_use_sim_start_command(should_rotate_sim_types):
    use_sim_start_command = should_rotate_sim_types and np.random.random() < 0.5
    return use_sim_start_command


def _ensure_baseline_weights(net_path, version, weights_dir, url):
    if net_path is not None:
        raise ValueError('Net path should not be set when running the baseline agent as it has its own weights.')
    net_path = os.path.join(weights_dir, version)
    if not glob.glob(net_path + '*'):
        print('\n--------- Baseline weights not found, downloading ----------')
        download(url + '?cache_bust=' + version, c.WEIGHTS_DIR,
                 warn_existing=False, overwrite=True)
    return net_path


def ensure_alexnet_baseline_weights(net_path):
    return _ensure_baseline_weights(net_path, c.ALEXNET_BASELINE_WEIGHTS_VERSION, c.ALEXNET_BASELINE_WEIGHTS_DIR,
                                    c.ALEXNET_BASELINE_WEIGHTS_URL)


def ensure_mnet2_baseline_weights(net_path):
    return _ensure_baseline_weights(net_path, c.MNET2_BASELINE_WEIGHTS_VERSION, c.MNET2_BASELINE_WEIGHTS_DIR,
                                    c.MNET2_BASELINE_WEIGHTS_URL)

def ensure_ppo_baseline_weights(net_path):
    return _ensure_baseline_weights(net_path, c.PPO_BASELINE_WEIGHTS_VERSION, c.PPO_BASELINE_WEIGHTS_DIR,
                                    c.PPO_BASELINE_WEIGHTS_URL)


if __name__ == '__main__':
    ensure_mnet2_baseline_weights(None)
