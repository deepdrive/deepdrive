from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from future.builtins import (super)

import gym
import tensorflow as tf
import numpy as np
from gym import spaces

import config as c
import sim
from agents.common import get_throttle
from agents.dagger.agent import Agent
from config import MOBILENET_V2_NAME, MOBILENET_V2_IMAGE_SHAPE
from sim.driving_style import DrivingStyle
from sim.action import Action
from util.experience_buffer import ExperienceBuffer
from vendor.openai.baselines.ppo2.run_deepdrive import train


class BootstrapRLGymEnv(gym.Wrapper):
    """
    Bootstrap is probably a bad name here due to its overloaded use in RL
    where bootstrapping historically refers to learning with value based or
    TDD methods.
    """
    def __init__(self, env, dagger_agent, driving_style=DrivingStyle.STEER_ONLY):
        """
        Normalize the brake and
        handbrake space to be between -1 and 1 so
        that all actions have the same dimension -
        Then create a single box space. The is_game_driving space
        can be ignored for now within the ppo agent.
        """
        super(BootstrapRLGymEnv, self).__init__(env)

        self.denormalizers = None
        self.combine_action_spaces()

        self.dagger_agent = dagger_agent
        self.driving_style = driving_style
        self.previous_obz = None
        self.experience_buffer = ExperienceBuffer()

        self.simple_test = c.SIMPLE_PPO

        # One thing we need to do here is to make each action a bi-modal guassian to avoid averaging 50/50 decisions
        # i.e. half the time we veer left, half the time veer right - but on average this is go straight and can run us
        # into an obstacle. right now the DiagGaussianPd is just adding up errors which would not be the right
        # thing to do for a bi-modal guassian. also, DiagGaussianPd assumes steering and throttle are
        # independent which is not the case (steering at higher speeds causes more acceleration a = v**2/r),
        # so that may be a problem as well.

        if self.simple_test:
            shape = (5,)
        else:
            # TODO(post v3): Add prior 200ms, 500ms, 1s and 2s mobilenet activations, along with speed, acceleration, and other stats we get from obz
            speed_length = 1
            acceleration_length = 3  # x,y,z
            previous_output_length = 3  # steer,throttle,handbrake

            # obz_length = dagger_agent.net.num_last_hidden + dagger_agent.net.num_targets
            #
            # shape = (obz_length * self.experience_buffer.fade_length,)

            shape = (dagger_agent.net.num_last_hidden + dagger_agent.net.num_targets,)

        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            # shape=(c.ALEXNET_FC7 + c.NUM_TARGETS,),
                                            shape=shape,
                                            dtype=np.float32)

    def step(self, action):

        # Denormalize the action into the original high and low for the space
        action = [[denorm(action[i])] for i, denorm in
                  enumerate(self.denormalizers)]

        if self.driving_style == DrivingStyle.STEER_ONLY and self.previous_obz is not None:
            # Simplifying by only controlling steering. Otherwise, we need to shape rewards so that initial acceleration
            # is not disincentivized by gforce penalty.
            action[Action.THROTTLE_INDEX] = [get_throttle(
                actual_speed=self.previous_obz['speed'],
                target_speed=(8 * 100))]
            action[Action.BRAKE_INDEX] = [0]
            action[Action.HANDBRAKE_INDEX] = [0]

        obz, reward, done, info = self.env.step(action)
        if 'episode_return' in info and 'episode_time' in info['episode_return']:
            self.experience_buffer.maybe_add(obz, info['episode_return']['episode_time'])
        self.previous_obz = obz
        action, net_out = self.dagger_agent.act(obz, reward, done)
        if net_out is None:
            obz = None
        else:
            if self.simple_test:
                obz = np.array([np.squeeze(a) for a in action])
            else:
                obz = np.concatenate((np.squeeze(net_out[0]), np.squeeze(net_out[1])))
        return obz, reward, done, info

    def reset(self):
        return self.env.reset()

    def combine_action_spaces(self):
        """
        Normalize the brake and handbrake space to be between -1 and 1 so
        that all actions have the same dimension -
        Then create a single box space. The is_game_driving space
        can be ignored for now within the ppo agent.
        """
        ac_space = self.action_space
        if isinstance(ac_space, gym.spaces.Tuple):
            self.denormalizers = []
            box_spaces = [s for s in ac_space.spaces if
                          isinstance(s, gym.spaces.Box)]
            total_dims = 0
            for i, space in enumerate(box_spaces):
                if len(space.shape) > 1 or space.shape[0] > 1:
                    raise NotImplementedError(
                        'Multi-dimensional box spaces not yet supported - need to flatten / separate')
                else:
                    total_dims += 1
                self.denormalizers.append(
                    self.get_denormalizer(space.high[0], space.low[0]))
            self.action_space = gym.spaces.Box(-1, 1, shape=(total_dims,))

    @staticmethod
    def get_denormalizer(high, low):
        def denormalizer(x):
            ret = (x + 1) * (high - low) / 2 + low
            return ret
        return denormalizer


def run(env_id, bootstrap_net_path,
        resume_dir=None, experiment=None, camera_rigs=None, render=False, fps=c.DEFAULT_FPS,
        should_record=False, is_discrete=False, agent_name=MOBILENET_V2_NAME, is_sync=True,
        driving_style=DrivingStyle.NORMAL, is_remote_client=False, eval_only=False,
        sim_args=None):
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.4,
            # leave room for the game,
            # NOTE: debugging python, i.e. with PyCharm can cause OOM errors, where running will not
            allow_growth=True
        ),
    )

    g_1 = tf.Graph()
    with g_1.as_default():
        sess_1 = tf.Session(config=tf_config)

        with sess_1.as_default():
            sim_args_for_rl = dict(
                experiment=experiment, env_id=env_id, cameras=camera_rigs,
                render=render, fps=fps,
                is_sync=is_sync, driving_style=driving_style,
                is_remote_client=is_remote_client, should_record=should_record,
                image_resize_dims=MOBILENET_V2_IMAGE_SHAPE,
                should_normalize_image=True)

            sim_args_dict = sim_args.to_dict()
            sim_args_dict.update(sim_args_for_rl)

            dagger_gym_env = sim.start(**sim_args_dict)
            dagger_agent = Agent(
                sess_1, should_jitter_actions=False,
                net_path=bootstrap_net_path, output_last_hidden=True,
                net_name=MOBILENET_V2_NAME)

    g_2 = tf.Graph()
    with g_2.as_default():
        sess_2 = tf.Session(config=tf_config)

        with sess_2.as_default():

            # Wrap step so we get the pretrained layer activations rather than pixels for our observation
            bootstrap_gym_env = BootstrapRLGymEnv(dagger_gym_env, dagger_agent, driving_style)

            if c.SIMPLE_PPO:
                minibatch_steps = 16
                mlp_width = 5
            else:
                minibatch_steps = 80
                mlp_width = 64
            train(bootstrap_gym_env, seed=c.RNG_SEED, sess=sess_2, is_discrete=is_discrete,
                  minibatch_steps=minibatch_steps, mlp_width=mlp_width, eval_only=eval_only)
    #
    # action = deepdrive.action()
    # while not done:
    #     observation, reward, done, info = gym_env.step(action)


