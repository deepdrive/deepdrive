import gym
import tensorflow as tf
import numpy as np
from gym import spaces

import deepdrive
import config as c
from agents.dagger.agent import Agent
from agents.dagger.net import MOBILENET_V2_NAME
from vendor.openai.baselines.ppo2.run_deepdrive import train


class BootstrapRLGymEnv(gym.Wrapper):
    def __init__(self, env, dagger_agent):
        super(BootstrapRLGymEnv, self).__init__(env)
        self.dagger_agent = dagger_agent

        # One thing we need to do here is to make each action a bi-modal guassian to avoid averaging 50/50 decisions
        # i.e. half the time we veer left, half the time veer right - but on average this is go straight and can run us
        # into an obstacle. right now the DiagGaussianPd is just adding up errors which would not be the right
        # thing to do for a bi-modal guassian. also, DiagGaussianPd assumes steering and throttle are
        # independent which is not the case (steering at higher speeds causes more acceleration a = v**2/r),
        # so that may be a problem as well.

        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            # shape=(c.ALEXNET_FC7 + c.NUM_TARGETS,),
                                            shape=(dagger_agent.net.num_last_hidden,),
                                            dtype=np.float32)

    def step(self, action):
        obz, reward, done, info = self.env.step(action)
        action, net_out = self.dagger_agent.act(obz, reward, done)
        if net_out is None:
            obz = None
        else:
            obz = np.concatenate((net_out[0][0], net_out[1][0]))
        return obz, reward, done, info

    def reset(self):
        return self.env.reset()


def run(env_id, bootstrap_net_path,
        resume_dir=None, experiment=None, cameras=None, render=False, fps=c.DEFAULT_FPS,
        should_record=False, is_discrete=False, agent_name=MOBILENET_V2_NAME):
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.8,
            # leave room for the game,
            # NOTE: debugging python, i.e. with PyCharm can cause OOM errors, where running will not
            allow_growth=True
        ),
    )

    sess = tf.Session(config=tf_config)
    with sess.as_default():
        dagger_gym_env = deepdrive.start(experiment, env_id, cameras=cameras, render=render, fps=fps,
                                         combine_box_action_spaces=True, is_sync=True, sync_step_time=0.125)

        dagger_agent = Agent(dagger_gym_env.action_space, sess, env=dagger_gym_env.env,
                             should_record_recovery_from_random_actions=False, should_record=should_record,
                             net_path=bootstrap_net_path, output_last_hidden=True, net_name=MOBILENET_V2_NAME)

        # Wrap step so we get the pretrained layer activations rather than pixels for our observation
        bootstrap_gym_env = BootstrapRLGymEnv(dagger_gym_env, dagger_agent)

        train(bootstrap_gym_env, num_timesteps=int(10e6), seed=c.RNG_SEED, sess=sess, is_discrete=is_discrete)
    #
    # action = deepdrive.action()
    # while not done:
    #     observation, reward, done, info = gym_env.step(action)


