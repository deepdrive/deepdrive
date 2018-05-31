# noinspection PyUnresolvedReferences
import gym_deepdrive  # forward registers gym environment
import gym
import logs

import config as c
import random_name

# noinspection PyUnresolvedReferences
from gym_deepdrive.envs.deepdrive_gym_env import gym_action as action
from vendor.openai.baselines.common.continuous_action_wrapper import CombineBoxSpaceWrapper

log = logs.get_log(__name__)


def start(experiment_name=None, env='DeepDrive-v0', sess=None, start_dashboard=True, should_benchmark=True,
          cameras=None, use_sim_start_command=False, render=False, fps=c.DEFAULT_FPS, combine_box_action_spaces=False,
          is_discrete=False, preprocess_with_tensorflow=False, is_sync=False, sync_step_time=0.125):
    env = gym.make(env)
    env.seed(c.RNG_SEED)

    if experiment_name is None:
        experiment_name = ''

    dd_env = env.unwrapped
    dd_env.is_discrete = is_discrete
    dd_env.preprocess_with_tensorflow = preprocess_with_tensorflow
    dd_env.is_sync = is_sync
    dd_env.sync_step_time = sync_step_time
    dd_env.init_action_space()
    dd_env.fps = fps
    dd_env.experiment = experiment_name.replace(' ', '_')
    dd_env.period = 1. / fps
    dd_env.set_use_sim_start_command(use_sim_start_command)
    dd_env.open_sim()
    if use_sim_start_command:
        input('Press any key when the game has loaded')  # TODO: Find a better way to do this. Waiting for the hwnd and focusing does not work in windows.
    dd_env.connect(cameras, render)
    dd_env.set_step_mode()
    if combine_box_action_spaces:
        env = CombineBoxSpaceWrapper(env)
    env = gym.wrappers.Monitor(env, directory=c.GYM_DIR, force=True)
    if sess:
        dd_env.set_tf_session(sess)
    if start_dashboard:
        dd_env.start_dashboard()
    if should_benchmark:
        log.info('Benchmarking enabled - will save results to %s', c.BENCHMARK_DIR)
        dd_env.init_benchmarking()
    env.reset()
    return env
