import os
import time
from datetime import datetime

import gym
import tensorflow as tf
import numpy as np

import config as c
import deepdrive_env
from gym_deepdrive.envs.deepdrive_gym_env import Action
from tensorflow_agent.net import Net
from utils import save_hdf5, download
import logs

gym.undo_logger_setup()
log = logs.get_log(__name__)


class Agent(object):
    def __init__(self, action_space, tf_session, env, fps=8, should_toggle_random_actions=True, should_record=False,
                 net_path=None, use_frozen_net=False, random_action_count=0, non_random_action_count=5):
        np.random.seed(c.RNG_SEED)
        self.action_space = action_space
        self.previous_action_time = None
        self.fps = fps
        self.period = 1. / fps
        self.previous_action = None
        self.step = 0
        self.env = env

        # State for toggling random actions
        self.should_toggle_random_actions = should_toggle_random_actions
        self.random_action_count = random_action_count
        self.non_random_action_count = non_random_action_count
        self.semirandom_sequence_step = 0
        self.action_count = 0
        self.recorded_obz_count = 0
        self.performing_random_actions = False

        # Recording state
        self.should_record = should_record
        self.sess_dir = os.path.join(c.RECORDINGS_DIR, datetime.now().strftime(c.DIR_DATE_FORMAT))
        self.obz_recording = []

        if should_toggle_random_actions:
            log.info('Mixing in random actions to increase data diversity (these are not recorded).')
        if should_record:
            log.info('Recording driving data to %s', self.sess_dir)

        # Net
        self.sess = tf_session
        self.use_frozen_net = use_frozen_net
        if net_path is not None:
            self.load_net(net_path, use_frozen_net)
        else:
            self.net = None
            self.net_input_placeholder = None
            self.sess = None

    def act(self, obz, reward, done):
        now = time.time()
        if self.previous_action_time:
            delta = now - self.previous_action_time
            if delta < self.period:
                time.sleep(delta)
            else:
                fps = 1. / delta
                if self.step > 5 and fps < self.fps / 2:
                    log.warning('Low FPS of %r, target is %r, step %r', fps, self.fps, self.step)

        if obz is not None:
            log.debug('steering %r', obz['steering'])
            log.debug('throttle %r', obz['throttle'])
            obz = self.preprocess_obz(obz)

        if self.should_toggle_random_actions:
            action = self.toggle_random_action()
            self.action_count += 1
        elif self.net is not None:
            if obz is None or not obz['cameras']:
                y = None
            else:
                image = obz['cameras'][0]['image']
                y = self.get_net_out(image)
            action = self.get_next_action(obz, y)
        else:
            action = Action()

        self.previous_action_time = now
        self.previous_action = action
        self.step += 1

        if obz and obz['is_game_driving'] == 1 and self.should_record:
            self.obz_recording.append(obz)
            # utils.save_camera(obz['cameras'][0]['image'], obz['cameras'][0]['depth'],
            #                   os.path.join(self.sess_dir, str(self.total_obz).zfill(10)))
            self.recorded_obz_count += 1

        self.maybe_save()

        action = Action.as_gym(action)
        return action

    def get_next_action(self, obz, y):
        log.debug('getting next action')
        if y is None:
            log.debug('net out is None')
            return self.previous_action  # or Action(has_control=False)

        desired_spin, desired_direction, desired_speed, desired_speed_change, desired_steering, desired_throttle = y[0]

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

        if actual_speed > desired_speed or actual_speed > 25 * 100:
            # Compensate for bad turning ability at high speed
            log.debug('limiting throttle')
            desired_throttle = desired_throttle * 0.3 - self.previous_action[1][0] * 0.3
            desired_throttle = max(desired_throttle, 0.0)
        elif actual_speed < 0.7 * desired_speed or actual_speed < 25 * 100:
            log.debug('boosting throttle')
            desired_throttle = desired_throttle * 1.25 + self.previous_action[1][0] * 0.5
            desired_throttle = min(desired_throttle, 1.25)
        log.debug('desired_steering %f', desired_steering)
        log.debug('desired_throttle %f', desired_throttle)
        smoothed_steering = 0.2 * self.previous_action[0][0] + 0.5 * desired_steering
        # desired_throttle = desired_throttle * 1.1
        action = Action(smoothed_steering, desired_throttle)
        return action

    def maybe_save(self):
        if (
            self.should_record and self.recorded_obz_count % c.FRAMES_PER_HDF5_FILE == 0 and
            self.recorded_obz_count != 0
           ):
            filename = os.path.join(self.sess_dir, '%s.hdf5' %
                                    str(self.recorded_obz_count // c.FRAMES_PER_HDF5_FILE).zfill(10))
            save_hdf5(self.obz_recording, filename=filename)
            log.info('Flushing output data')
            self.obz_recording = []

    def toggle_random_action(self):
        """Reduce sampling error by diversifying experience"""
        if self.performing_random_actions:
            if self.action_count < self.random_action_count and self.previous_action is not None:
                action = self.previous_action
            else:
                # switch to non-random
                action = Action(has_control=False)
                self.action_count = 0
                self.performing_random_actions = False
        else:
            if self.action_count < self.non_random_action_count and self.previous_action is not None:
                action = self.previous_action
            else:
                # switch to random
                steering = np.random.uniform(-0.5, 0.5, 1)[0]  # Going too large here gets us stuck
                log.debug('random steering %f', steering)
                throttle = 0.65  # TODO: Make throttle random to get better variation here
                action = Action(steering, throttle)
                self.action_count = 0
                self.performing_random_actions = True
        return action

    def load_net(self, net_path, is_frozen=False):
        '''
        Frozen nets can be generated with something like 
        
        `python freeze_graph.py --input_graph="C:\tmp\deepdrive\tensorflow_random_action\train\graph.pbtxt" --input_checkpoint="C:\tmp\deepdrive\tensorflow_random_action\train\model.ckpt-273141" --output_graph="C:\tmp\deepdrive\tensorflow_random_action\frozen_graph.pb" --output_node_names="model/add_2"`
        
        where model/add_2 is the auto-generated name for self.net.p 
        '''
        self.net_input_placeholder = tf.placeholder(tf.float32, (None,) + c.IMAGE_SHAPE)
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
            with tf.variable_scope("model") as _vs:
                self.net = Net(self.net_input_placeholder, c.NUM_TARGETS, is_training=False)
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
            out_var = self.net.p
        net_out = self.sess.run(out_var, feed_dict={
            self.net_input_placeholder: image.reshape(1, *image.shape),})
        # print(net_out)
        end = time.time()
        log.debug('inference time %s', end - begin)
        return net_out

    def preprocess_obz(self, obz):
        for camera in obz['cameras']:
            image = camera['image']
            image = image.astype(np.float32)
            image -= c.MEAN_PIXEL
            camera['image'] = image
        return obz

    def perform_semirandom_action(self):
        if self.semirandom_sequence_step == (self.random_action_count + self.non_random_action_count):
            self.semirandom_sequence_step = 0
            rand = c.RNG.random()
            if 0 <= rand < 0.67:
                self.random_action_count = 0
                self.non_random_action_count = 10
            elif 0.67 <= rand < 0.85:
                self.random_action_count = 4
                self.non_random_action_count = 5
            elif 0.85 <= rand < 0.95:
                self.random_action_count = 8
                self.non_random_action_count = 10
            else:
                self.random_action_count = 12
                self.non_random_action_count = 15
            log.debug('random actions at %r, non-random %r', self.random_action_count, self.non_random_action_count)

        else:
            self.semirandom_sequence_step += 1


def run(env_id='DeepDrivePreproTensorflow-v0', should_record=False, net_path=None, should_benchmark=False,
        run_baseline_agent=False):
    if run_baseline_agent:
        net_path = ensure_baseline_weights(net_path)
        if c.IS_SIM_DEV:
            log.warning('****\n Baseline agent performs poorly in the Unreal editor as it was not trained '
                        'there.\n\n****')
    reward = 0
    done = False
    render = False
    episode_count = 1
    tf_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.8,
            # leave room for the game,
            # NOTE: debugging python, i.e. with PyCharm can cause OOM errors, where running will not
            allow_growth=True
        ),
    )

    sess = tf.Session(config=tf_config)
    gym_env = deepdrive_env.start(env_id, should_benchmark=True)
    dd_env = gym_env.env

    # Perform random actions to reduce sampling error in the recorded dataset
    agent = Agent(gym_env.action_space, sess, env=gym_env.env, should_toggle_random_actions=should_record,
                  should_record=should_record, net_path=net_path, random_action_count=4, non_random_action_count=5)

    if net_path:
        log.info('Running tensorflow agent checkpoint: %s', net_path)

    def close():
        gym_env.close()
        agent.close()

    for episode in range(episode_count):
        if episode == 0 or done:
            obz = gym_env.reset()
        else:
            obz = None
        try:
            while True:
                action = agent.act(obz, reward, done)
                obz, reward, done, _ = gym_env.step(action)
                if render:
                    gym_env.render()
                if done:
                    gym_env.reset()
                if should_record:
                    agent.perform_semirandom_action()
                if agent.recorded_obz_count > c.MAX_RECORDED_OBSERVATIONS:
                    break
                if should_benchmark and dd_env.done_benchmarking:
                    break
        except KeyboardInterrupt:
            log.info('keyboard interrupt detected, closing')
            close()
    close()


def ensure_baseline_weights(net_path):
    if net_path is not None:
        raise ValueError('Net path should not be set when running the baseline agent as it has its own weights.')
    if not os.path.exists(c.BASELINE_WEIGHTS_DIR):
        print('\n--------- Baseline weights not found, downloading ----------')
        download(c.BASELINE_WEIGHTS_URL, c.WEIGHTS_DIR,
                 warn_existing=False, overwrite=False)
    net_path = os.path.join(c.BASELINE_WEIGHTS_DIR, c.BASELINE_WEIGHTS_VERSION)
    return net_path
