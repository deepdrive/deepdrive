#!/usr/bin/env python3

import os

from vendor.openai.baselines import bench, logger

from vendor.openai.baselines.common.cmd_util import continuous_mountain_car_arg_parser


def train(env_id, num_timesteps, seed):
    from vendor.openai.baselines.common.misc_util import set_global_seeds
    from vendor.openai.baselines.common.vec_env import VecNormalize
    from vendor.openai.baselines.ppo2 import ppo2
    from vendor.openai.baselines.ppo2 import MlpPolicy, LstmPolicyFlat
    import gym
    import tensorflow as tf
    from vendor.openai.baselines.common.vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        pstep = env.step

        def step(action):
            observation, reward, done, info = pstep(action)
            env.render()
            return observation, reward, done, info
        env.step = step
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    if 'LSTM_FLAT' in os.environ:
        policy = LstmPolicyFlat
    else:
        policy = MlpPolicy
    ppo2.learn(policy=policy,
               env=env,
               nsteps=256,
               nminibatches=32,  # Sweet spot is between 16 and 64
               lam=0.95,
               gamma=0.99,
               noptepochs=10,
               log_interval=1,
               ent_coef=0.0,
               lr=lambda f: f * 2.5e-4,
               cliprange=lambda f: f * 0.1,
               total_timesteps=num_timesteps)


def main():
    args = continuous_mountain_car_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
