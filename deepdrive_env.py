import gym_deepdrive  # forward registers gym enviornment
import gym
import logs

import config as c
from gym_deepdrive.envs.deepdrive_gym_env import gym_action as action
gym.undo_logger_setup()
log = logs.get_log(__name__)


def start(env='DeepDrive-v0', sess=None, start_dashboard=True, should_benchmark=True):
    env = gym.make(env)
    env = gym.wrappers.Monitor(env, directory=c.GYM_DIR, force=True)
    env.seed(0)
    dd_env = env.env
    dd_env.connect()
    if sess:
        dd_env.set_tf_session(sess)
    if start_dashboard:
        dd_env.start_dashboard()
    if should_benchmark:
        log.info('Benchmarking enabled - will save results to %s', c.BENCHMARK_DIR)
        dd_env.init_benchmarking()
    return env
