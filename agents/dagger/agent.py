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
from agents.common import get_throttle
from agents.dagger import net
from agents.dagger.action_jitterer import ActionJitterer, JitterState
from sim.driving_style import DrivingStyle
from sim.action import Action
from sim import world
from agents.dagger.net import AlexNet, MobileNetV2
from config import MOBILENET_V2_IMAGE_SHAPE
from sim.sim_args import SimArgs
from util.download import download
import logs

log = logs.get_log(__name__)

TARGET_MPH = 25
TARGET_MPS = TARGET_MPH / 2.237
TARGET_MPS_TEST = 75 * TARGET_MPS


class Agent(object):
    def __init__(self, tf_session, should_jitter_actions=True,
                 net_path=None, use_frozen_net=False, path_follower=False,
                 output_last_hidden=False, net_name=c.ALEXNET_NAME,
                 driving_style: DrivingStyle = DrivingStyle.NORMAL):
        np.random.seed(c.RNG_SEED)
        self.previous_action = None
        self.previous_net_out = None
        self.step = 0
        self.net_name = net_name
        self.driving_style = driving_style

        # State for toggling random actions
        self.should_jitter_actions = should_jitter_actions
        self.episode_action_count = 0
        self.path_follower_mode = path_follower
        self.output_last_hidden = output_last_hidden

        if should_jitter_actions:
            log.info('Mixing in random actions to increase data diversity '
                     '(these are not recorded).')

        # Net
        self.sess = tf_session
        self.use_frozen_net = use_frozen_net
        if net_path is not None:
            if net_name == c.ALEXNET_NAME:
                input_shape = c.ALEXNET_IMAGE_SHAPE
            elif net_name == c.MOBILENET_V2_NAME:
                input_shape = c.MOBILENET_V2_IMAGE_SHAPE
            else:
                raise NotImplementedError(net_name + ' not recognized')
            self.load_net(net_path, use_frozen_net, input_shape)
        else:
            self.net = None
            self.sess = None

        self.throttle_pid = PID(0.3, 0.05, 0.4)
        self.throttle_pid.output_limits = (-1, 1)

        self.jitterer = ActionJitterer()

    def act(self, obz, reward, done):
        net_out = None
        if obz:
            try:
                log.debug('steering %r', obz['steering'])
                log.debug('throttle %r', obz['throttle'])
            except TypeError as e:
                log.error('Could not parse this observation %r', obz)
                raise

        if self.should_jitter_actions:
            episode_time = obz.get('episode_return', {}).get('episode_time',
                                                    None) if obz else None
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
        action = action.as_gym()
        return action, net_out

    def get_next_action(self, obz, net_out):
        log.debug('getting next action')
        if net_out is None:
            log.debug('net out is None')
            return self.previous_action or Action()

        net_out = net_out[0]  # We currently only have one environment

        (desired_spin,
         desired_direction,
         desired_speed,
         desired_speed_change,
         desired_steering,
         desired_throttle) = net_out

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

            # TODO(post v3): Support different driving styles

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

    def jitter_action(self, obz):
        """
        Reduce sampling error by randomly exploring space around non-random agent's
        trajectory with occasional random actions
        """
        state = self.jitterer.advance()
        if state is JitterState.SWITCH_TO_RAND:
            # Going too large here gets us stuck
            steering = np.random.uniform(-0.5, 0.5, 1)[0]

            # Slow down a bit so we don't crash before recovering
            throttle = self.get_target_throttle(obz) * 0.5

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
            # TODO(post v3): Move setpoint to env
            # We should not be calling single purpose
            # UEPy API methods this frequently, i.e. every step as the server
            # can only respond to one message per frame.
            # This should be moved to a combined call in step instead.
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
            # TODO(post v3): Get frozen nets working

            # We load the protobuf file from the disk and parse it to
            # retrieve the unserialized graph_def
            with tf.gfile.GFile(net_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # Then, we can use again a convenient built-in function to 
            # import a graph_def into the current default Graph
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
            if self.net_name == c.MOBILENET_V2_NAME:
                self.net = MobileNetV2(is_training=False)
            else:
                self.net = AlexNet(is_training=False)
            saver = tf.train.Saver()
            saver.restore(self.sess, net_path)

    def close(self):
        if self.sess is not None:
            self.sess.close()

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

    def reset(self):
        self.jitterer.reset()
        self.throttle_pid.auto_mode = False


def run(sim_args: SimArgs,
        net_path=None,
        run_baseline_agent=False,
        run_mnet2_baseline_agent=False,
        run_ppo_baseline_agent=False,
        camera_rigs=None,
        should_jitter_actions=False,
        path_follower=False,
        net_name=c.ALEXNET_NAME,
        max_episodes=1000,
        agent_name=None,

        # Placeholder for rotating between Unreal Editor and packaged game
        should_rotate_sim_types=False, ):
    """
    Run inference for agents
    """
    sim_args.max_episodes = max_episodes
    if sim_args.should_record:
        path_follower = True

    if agent_name == c.DAGGER_MNET2:
        image_resize_dims = MOBILENET_V2_IMAGE_SHAPE
    else:
        image_resize_dims = None

    agent, env, should_rotate_camera_rigs, start_env = \
        setup(sim_args=sim_args,
              camera_rigs=camera_rigs,
              net_name=net_name,
              net_path=net_path,
              path_follower=path_follower,
              run_baseline_agent=run_baseline_agent,
              run_mnet2_baseline_agent=run_mnet2_baseline_agent,
              run_ppo_baseline_agent=run_ppo_baseline_agent,
              should_jitter_actions=should_jitter_actions,
              image_resize_dims=image_resize_dims,
              agent_name=agent_name)

    reward = 0
    episode_done = False

    def close():
        env.close()
        agent.close()

    session_done = False
    episode = 0

    try:
        while not session_done:
            if episode_done:
                # TODO: Don't use reset return value. Should always be 0.
                obz = env.reset()
                episode_done = False
            else:
                obz = None
            if max_episodes is not None and episode >= (max_episodes - 1):
                session_done = True
            episode_done, should_close = run_episode(
                agent, env, episode_done, obz, reward)

            if session_done:
                log.info('Session done')
            elif should_close:
                session_done = True
                log.info('Server directed to close')
            else:
                log.info('Episode done')
                episode += 1
                agent.reset()
                if should_rotate_camera_rigs:
                    rotate_cameras(camera_rigs, env, episode)

    except KeyboardInterrupt:
        log.info('keyboard interrupt detected in agent, closing')
    except Exception as e:
        raise e
    finally:
        close()


def rotate_cameras(camera_rigs, env, episode):
    # TODO: Add this to domain_randomization()
    cameras = copy.deepcopy(
        camera_rigs[episode % len(camera_rigs)])
    randomize_cameras(cameras)
    env.change_cameras(cameras)


def run_episode(agent, env, episode_done, obz, reward):
    should_close = False
    while not episode_done:
        act_start = time.time()
        action, net_out = agent.act(obz, reward, episode_done)
        log.debug('act took %fs', time.time() - act_start)

        env_step_start = time.time()
        obz, reward, episode_done, info = env.step(action)
        if 'should_close' in info:
            should_close = info['should_close']
        log.debug('env step took %fs', time.time() - env_step_start)
    return episode_done, should_close


def setup(sim_args, camera_rigs, net_name, net_path,
          path_follower, run_baseline_agent,
          run_mnet2_baseline_agent,
          run_ppo_baseline_agent, should_jitter_actions, image_resize_dims,
          agent_name):
    if net_path and (run_baseline_agent or
                     run_mnet2_baseline_agent or
                     run_ppo_baseline_agent):
        raise RuntimeError('Running a baseline agent does not require a '
                           'net_path. Please remove net_path or baseline '
                           'argument')
    if not net_path:
        if agent_name == c.DAGGER:
            log.info('Running alexnet dagger agent')
            run_baseline_agent = True
        elif agent_name == c.DAGGER_MNET2:
            log.info('Running mnet2 dagger agent')
            run_mnet2_baseline_agent = True
        elif agent_name == c.BOOTSTRAPPED_PPO2:
            log.info('Running baseline ppo2 agent')
            run_ppo_baseline_agent = True

    if run_baseline_agent:
        net_path = ensure_alexnet_baseline_weights(net_path)
    elif run_mnet2_baseline_agent:
        net_path = ensure_mnet2_baseline_weights(net_path)
    elif run_ppo_baseline_agent:
        net_path = ensure_ppo_baseline_weights(net_path)

    sess = config_tensorflow_memory(net_name)

    # TODO: Allow rotating and randomizing cameras rigs in env
    if camera_rigs:
        cameras = camera_rigs[0]
    else:
        cameras = None
    if sim_args.should_record and camera_rigs is not None and \
            len(camera_rigs) >= 1:
        should_rotate_camera_rigs = True
        log.info(
            'Rotating cameras while recording to encourage visual robustness')
    else:
        should_rotate_camera_rigs = False
    if should_rotate_camera_rigs:
        randomize_cameras(cameras)

    use_sim_start_command_first_lap = c.SIM_START_COMMAND is not None

    def start_env():
        sim_args.cameras = cameras
        sim_args.use_sim_start_command = use_sim_start_command_first_lap
        sim_args.image_resize_dims = image_resize_dims
        return sim.start(**(sim_args.to_dict()))

    env = start_env()

    try:
        agent = Agent(sess, should_jitter_actions=should_jitter_actions,
                      net_path=net_path, path_follower=path_follower,
                      net_name=net_name,
                      driving_style=DrivingStyle.from_str(sim_args.driving_style))
    except Exception as e:
        env.close()
        raise e

    if net_path:
        log.info('Running tensorflow agent checkpoint: %s', net_path)
    return agent, env, should_rotate_camera_rigs, start_env


def config_tensorflow_memory(net_name):
    """
    Configure TensorFlow so that we leave GPU memory for the game to run
    on cards with 4GB of VRAM

    NOTE: Debugging python, i.e. with PyCharm can cause OOM errors,
    where running will not

    :param net_name: Name of the Neural Network we're using
    :return: Tensorflow Session object
    """

    if net_name == c.ALEXNET_NAME:
        per_process_gpu_memory_fraction = 0.8
    else:
        per_process_gpu_memory_fraction = 0.4
    tf_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction,
            allow_growth=True
        ),
    )
    sess = tf.Session(config=tf_config)
    return sess


def okay_to_jitter_actions(obz):
    if obz is None:
        return False
    else:
        dnext = obz['distance_to_next_agent']
        dprev = obz['distance_to_prev_agent']
        if dnext == -1.0 and dprev == -1.0:
            return True
        if dnext < (100 * 100) or dprev < (50 * 100):
            log.info(
                'Not okay to act randomly passing %r distance next %r distance prev %r',
                obz['is_passing'], obz['distance_to_next_agent'],
                obz['distance_to_prev_agent'])
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
        cam['capture_height'] += round(np.random.random() *
                                       0.01 * cam['capture_height'])
        cam['capture_width'] += round(np.random.random() *
                                      0.01 * cam['capture_width'])


def random_use_sim_start_command(should_rotate_sim_types):
    use_sim_start_command = should_rotate_sim_types and np.random.random() < 0.5
    return use_sim_start_command


def _ensure_baseline_weights(net_path, version, weights_dir, url):
    if net_path is not None:
        raise ValueError('Net path should not be set when running the '
                         'baseline agent as it has its own weights.')
    net_path = os.path.join(weights_dir, version)
    if not glob.glob(net_path + '*'):
        print('\n--------- Baseline weights not found, downloading ----------')
        download(url + '?cache_bust=' + version, c.WEIGHTS_DIR,
                 warn_existing=False, overwrite=True)
    return net_path


def ensure_alexnet_baseline_weights(net_path):
    return _ensure_baseline_weights(net_path,
                                    c.ALEXNET_BASELINE_WEIGHTS_VERSION,
                                    c.ALEXNET_BASELINE_WEIGHTS_DIR,
                                    c.ALEXNET_BASELINE_WEIGHTS_URL)


def ensure_mnet2_baseline_weights(net_path):
    return _ensure_baseline_weights(net_path, c.MNET2_BASELINE_WEIGHTS_VERSION,
                                    c.MNET2_BASELINE_WEIGHTS_DIR,
                                    c.MNET2_BASELINE_WEIGHTS_URL)


def ensure_ppo_baseline_weights(net_path):
    return _ensure_baseline_weights(net_path, c.PPO_BASELINE_WEIGHTS_VERSION,
                                    c.PPO_BASELINE_WEIGHTS_DIR,
                                    c.PPO_BASELINE_WEIGHTS_URL)


if __name__ == '__main__':
    ensure_mnet2_baseline_weights(None)
