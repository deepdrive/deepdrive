#!/usr/bin/env python3

import os

import config as c

from vendor.openai.baselines import bench, logger

from vendor.openai.baselines.common.cmd_util import continuous_mountain_car_arg_parser


def train(env, seed, sess=None, is_discrete=True, minibatch_steps=None, mlp_width=None):
    from vendor.openai.baselines.common.misc_util import set_global_seeds
    from vendor.openai.baselines.common.vec_env.vec_normalize import VecNormalize
    from vendor.openai.baselines.ppo2 import ppo2
    from vendor.openai.baselines.ppo2.policies import CnnPolicy, LstmPolicyFlat, MlpPolicy
    import gym
    import tensorflow as tf
    from vendor.openai.baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1

    logger.configure()

    if sess is None:
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=ncpu,
                                inter_op_parallelism_threads=ncpu)
        tf.Session(config=config).__enter__()

    env = DummyVecEnv(envs=[env])

    if c.SIMPLE_PPO:
        env = VecNormalize(env, ob=False)
    else:
        env = VecNormalize(env)

    set_global_seeds(seed)
    if is_discrete:
        policy = LstmPolicyFlat
    else:
        # continuous
        policy = MlpPolicy


    # TODO: Stack 8 (1 second) input frames as is done for atari environments

    # TODO: Simplify by just outputting right, left, straight discrete actions, no bootstrap, Enduro setup. manual throttle - reward center of lane only

    ppo2.learn(policy=policy,
               env=env,
               nsteps=minibatch_steps,
               nminibatches=1,  # Sweet spot is between 16 and 64 for continuous mountain car @55fps
               lam=0.95,
               gamma=0.99,
               save_interval=1,
               noptepochs=3,
               log_interval=1,
               ent_coef=0.0,
               lr=lambda f: f * 2.5e-3,
               cliprange=lambda f: f * 0.1,
               total_timesteps=int(1e5),
               mlp_width=mlp_width)

