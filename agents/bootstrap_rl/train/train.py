from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

from collections import deque

import gym
import tensorflow as tf
import numpy as np
from gym import spaces

import deepdrive
import config as c
from agents.common import get_throttle
from agents.dagger.agent import Agent
from agents.dagger.net import MOBILENET_V2_NAME
from gym_deepdrive.envs.deepdrive_gym_env import Action, DrivingStyle
from gym_deepdrive.experience_buffer import ExperienceBuffer
from vendor.openai.baselines.ppo2.run_deepdrive import train


class BootstrapRLGymEnv(gym.Wrapper):
    """Bootstrap is probably a bad name here due to its overloaded use in RL where bootstrapping historically refers
    to learning with value based or TDD methods."""
    def __init__(self, env, dagger_agent):
        super(BootstrapRLGymEnv, self).__init__(env)
        self.dagger_agent = dagger_agent
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
            # TODO: Add prior 200ms, 500ms, 1s and 2s mobilenet activations, along with speed, acceleration, and other stats we get from obz
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
        if self.env.unwrapped.driving_style == DrivingStyle.STEER_ONLY and self.previous_obz is not None:
            # Simplifying by only controlling steering. Otherwise, we need to shape rewards so that initial acceleration
            # is not disincentivized by gforce penalty.
            action[Action.THROTTLE_INDEX] = get_throttle(actual_speed=self.previous_obz['speed'],
                                                         target_speed=(8 * 100))
        obz, reward, done, info = self.env.step(action)
        print(obz)
        input('continue?')
        self.experience_buffer.maybe_add(obz, self.env.unwrapped.score.episode_time)
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


def run(env_id, bootstrap_net_path,
        resume_dir=None, experiment=None, camera_rigs=None, render=False, fps=c.DEFAULT_FPS,
        should_record=False, is_discrete=False, agent_name=MOBILENET_V2_NAME, is_sync=True,
        driving_style=DrivingStyle.NORMAL):
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
            dagger_gym_env = deepdrive.start(experiment=experiment, env_id=env_id, cameras=camera_rigs, render=render, fps=fps,
                                             combine_box_action_spaces=True, is_sync=is_sync,
                                             driving_style=driving_style)

            dagger_agent = Agent(sess_1, env=dagger_gym_env.env,
                                 should_record_recovery_from_random_actions=False, should_record=should_record,
                                 net_path=bootstrap_net_path, output_last_hidden=True, net_name=MOBILENET_V2_NAME)

    g_2 = tf.Graph()
    with g_2.as_default():
        sess_2 = tf.Session(config=tf_config)

        with sess_2.as_default():

            # Wrap step so we get the pretrained layer activations rather than pixels for our observation
            bootstrap_gym_env = BootstrapRLGymEnv(dagger_gym_env, dagger_agent)

            if c.SIMPLE_PPO:
                minibatch_steps = 16
                mlp_width = 5
            else:
                minibatch_steps = 80
                mlp_width = 64
            train(bootstrap_gym_env, seed=c.RNG_SEED, sess=sess_2, is_discrete=is_discrete,
                  minibatch_steps=minibatch_steps, mlp_width=mlp_width)
    #
    # action = deepdrive.action()
    # while not done:
    #     observation, reward, done, info = gym_env.step(action)


