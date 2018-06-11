# noinspection PyUnresolvedReferences
import gym_deepdrive  # forward registers gym environment
import gym
import logs

import config as c
import random_name

# noinspection PyUnresolvedReferences
from gym_deepdrive.envs.deepdrive_gym_env import gym_action as action, DrivingStyle
from vendor.openai.baselines.common.continuous_action_wrapper import CombineBoxSpaceWrapper

log = logs.get_log(__name__)


def start(experiment_name=None, env='Deepdrive-v0', sess=None, start_dashboard=True, should_benchmark=True,
          cameras=None, use_sim_start_command=False, render=False, fps=c.DEFAULT_FPS, combine_box_action_spaces=False,
          is_discrete=False, preprocess_with_tensorflow=False, is_sync=False,
          driving_style=DrivingStyle.NORMAL):
    env = gym.make(env)
    env.seed(c.RNG_SEED)

    if experiment_name is None:
        experiment_name = ''

    raw_env = env.unwrapped

    # This becomes our constructor - to facilitate using Gym API without registering combinations of params
    raw_env.is_discrete = is_discrete
    raw_env.preprocess_with_tensorflow = preprocess_with_tensorflow
    raw_env.is_sync = is_sync
    raw_env.init_action_space()
    raw_env.fps = fps
    raw_env.experiment = experiment_name.replace(' ', '_')
    raw_env.period = raw_env.sync_step_time = 1. / fps
    raw_env.driving_style = driving_style
    raw_env.should_render = render
    raw_env.set_use_sim_start_command(use_sim_start_command)
    raw_env.open_sim()
    if use_sim_start_command:
        input('Press any key when the game has loaded')  # TODO: Find a better way to do this. Waiting for the hwnd and focusing does not work in windows.
    raw_env.connect(cameras)
    raw_env.set_step_mode()
    if combine_box_action_spaces:
        env = CombineBoxSpaceWrapper(env)
    env = gym.wrappers.Monitor(env, directory=c.GYM_DIR, force=True)
    if sess:
        raw_env.set_tf_session(sess)
    if start_dashboard:
        raw_env.start_dashboard()
    if should_benchmark:
        log.info('Benchmarking enabled - will save results to %s', c.BENCHMARK_DIR)
        raw_env.init_benchmarking()
    env.reset()
    return env
