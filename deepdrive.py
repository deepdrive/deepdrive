from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

# noinspection PyUnresolvedReferences
import gym_deepdrive  # forward registers gym environment
import gym
import logs

import config as c
from api import client
from api.client import Client
from utils import remotable

# noinspection PyUnresolvedReferences
from gym_deepdrive.envs.deepdrive_gym_env import gym_action as action, DrivingStyle, ViewMode
from vendor.openai.baselines.common.continuous_action_wrapper import CombineBoxSpaceWrapper

log = logs.get_log(__name__)

def start(**kwargs):
    all_kwargs = dict(experiment_name=None, env_id='Deepdrive-v0', sess=None, start_dashboard=True,
                      should_benchmark=True, cameras=None, use_sim_start_command=False, render=False,
                      fps=c.DEFAULT_FPS, combine_box_action_spaces=False, is_discrete=False,
                      preprocess_with_tensorflow=False, is_sync=False, driving_style=DrivingStyle.NORMAL,
                      reset_returns_zero=True, is_remote_client=False)

    unexpected_args = set(kwargs) - set(all_kwargs)

    if unexpected_args:
        raise RuntimeError('Received unexpected args in run: ' + str(unexpected_args))

    all_kwargs.update(kwargs)
    kwargs = all_kwargs

    if kwargs['is_remote_client']:
        if not isinstance(kwargs['driving_style'], str):
            kwargs['driving_style'] = kwargs['driving_style'].name
        env = Client(**kwargs)
    else:
        if isinstance(kwargs['driving_style'], str):
            kwargs['driving_style'] = DrivingStyle.__members__[kwargs['driving_style']]

        env = gym.make(kwargs['env_id'])
        env.seed(c.RNG_SEED)

        if kwargs['experiment_name'] is None:
            kwargs['experiment_name'] = ''

        raw_env = env.unwrapped

        # This becomes our constructor - to facilitate using Gym API without registering combinations of params for the
        # wide variety of different environments we want.
        raw_env.is_discrete = kwargs['is_discrete']
        raw_env.preprocess_with_tensorflow = kwargs['preprocess_with_tensorflow']
        raw_env.is_sync = kwargs['is_sync']
        raw_env.reset_returns_zero = kwargs['reset_returns_zero']
        raw_env.init_action_space()
        raw_env.fps = kwargs['fps']
        raw_env.experiment = kwargs['experiment_name'].replace(' ', '_')
        raw_env.period = raw_env.sync_step_time = 1. / kwargs['fps']
        raw_env.driving_style = kwargs['driving_style']
        raw_env.should_render = kwargs['render']
        raw_env.set_use_sim_start_command(kwargs['use_sim_start_command'])
        raw_env.open_sim()
        if kwargs['use_sim_start_command']:
            # TODO: Find a better way to do this. Waiting for the hwnd and focusing does not work in windows.
            input('Press any key when the game has loaded')
        raw_env.connect(kwargs['cameras'])
        raw_env.set_step_mode()
        raw_env.set_view_mode(ViewMode.REFLECTIVITY)
        if kwargs['combine_box_action_spaces']:
            env = CombineBoxSpaceWrapper(env)
        if kwargs['sess']:
            raw_env.set_tf_session(kwargs['sess'])
        # if kwargs['start_dashboard']:
        #     raw_env.start_dashboard()
        if kwargs['should_benchmark']:
            log.info('Benchmarking enabled - will save results to %s', c.RESULTS_DIR)
            raw_env.init_benchmarking()

        env.reset()
    return env
