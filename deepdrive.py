from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (dict, input,
                             str)

# noinspection PyUnresolvedReferences
import sim  # forward registers gym environment
import gym
import logs

import config as c
from api.client import Client

# noinspection PyUnresolvedReferences
from sim.action import gym_action as action
from sim.driving_style import DrivingStyle
from sim.view_mode import ViewMode
from sim import world
from vendor.openai.baselines.common.continuous_action_wrapper import CombineBoxSpaceWrapper

log = logs.get_log(__name__)

def start(**kwargs):
    all_kwargs = dict(experiment_name=None, env_id='Deepdrive-v0', sess=None, start_dashboard=True,
                      should_benchmark=True, cameras=None, use_sim_start_command=False, render=False,
                      fps=c.DEFAULT_FPS, combine_box_action_spaces=False, is_discrete=False,
                      preprocess_with_tensorflow=False, is_sync=False, driving_style=DrivingStyle.NORMAL,
                      reset_returns_zero=True, is_remote_client=False, enable_traffic=True)

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

        deepdrive_env = env.unwrapped

        # This becomes our constructor - to facilitate using Gym API without registering combinations of params for the
        # wide variety of configurations we want.
        deepdrive_env.is_discrete = kwargs['is_discrete']
        deepdrive_env.preprocess_with_tensorflow = kwargs['preprocess_with_tensorflow']
        deepdrive_env.is_sync = kwargs['is_sync']
        deepdrive_env.reset_returns_zero = kwargs['reset_returns_zero']
        deepdrive_env.init_action_space()
        deepdrive_env.fps = kwargs['fps']
        deepdrive_env.experiment = kwargs['experiment_name'].replace(' ', '_')
        deepdrive_env.period = deepdrive_env.sync_step_time = 1. / kwargs['fps']
        deepdrive_env.driving_style = kwargs['driving_style']
        deepdrive_env.should_render = kwargs['render']
        deepdrive_env.set_use_sim_start_command(kwargs['use_sim_start_command'])
        deepdrive_env.open_sim()
        if kwargs['use_sim_start_command']:
            # TODO: Find a better way to do this. Waiting for the hwnd and focusing does not work in windows.
            input('Press any key when the game has loaded')
        deepdrive_env.connect(kwargs['cameras'])
        deepdrive_env.set_step_mode()

        if kwargs['combine_box_action_spaces']:
            env = CombineBoxSpaceWrapper(env)
        if kwargs['sess']:
            deepdrive_env.set_tf_session(kwargs['sess'])
        if kwargs['start_dashboard']:
            deepdrive_env.start_dashboard()
        if kwargs['should_benchmark']:
            log.info('Benchmarking enabled - will save results to %s', c.RESULTS_DIR)
            deepdrive_env.init_benchmarking()

        if kwargs['enable_traffic']:
            world.enable_traffic_next_reset()
        else:
            world.disable_traffic_next_reset()

        env.reset()
    return env
