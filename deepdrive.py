# noinspection PyUnresolvedReferences
import gym_deepdrive  # forward registers gym environment
import gym
import logs

import config as c
import random_name

# noinspection PyUnresolvedReferences
from gym_deepdrive.envs.deepdrive_gym_env import gym_action as action
log = logs.get_log(__name__)


def start(experiment_name=None, env='DeepDrive-v0', sess=None, start_dashboard=True, should_benchmark=True,
          cameras=None, use_sim_start_command=False, render=False, fps=c.DEFAULT_FPS):
    env = gym.make(env)
    env = gym.wrappers.Monitor(env, directory=c.GYM_DIR, force=True)
    env.seed(0)

    if experiment_name is None:
        default_ex_name = random_name.generate()
        experiment_name = input('Experiment name? (or press enter for random default "%s"): ' % default_ex_name) or default_ex_name

    dd_env = env.env
    dd_env.fps = fps
    dd_env.experiment = experiment_name.replace(' ', '_')
    dd_env.period = 1. / fps
    dd_env.set_use_sim_start_command(use_sim_start_command)
    dd_env.open_sim()
    if use_sim_start_command:
        input('Press any key when the game has loaded')  # TODO: Find a better way to do this. Waiting for the hwnd and focusing does not work in windows.
    dd_env.connect(cameras, render)
    if sess:
        dd_env.set_tf_session(sess)
    if start_dashboard:
        dd_env.start_dashboard()
    if should_benchmark:
        log.info('Benchmarking enabled - will save results to %s', c.BENCHMARK_DIR)
        dd_env.init_benchmarking()
    env.reset()
    return env
