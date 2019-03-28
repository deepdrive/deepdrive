from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (dict, input,
                             str)

from gym.envs.registration import register

# noinspection PyUnresolvedReferences
import gym
import logs

import config as c
from api.client import Client

# noinspection PyUnresolvedReferences
from sim.action import gym_action as action
from sim.driving_style import DrivingStyle
from sim.view_mode import ViewMode, ViewModeController
from sim import world
from vendor.openai.baselines.common.continuous_action_wrapper import CombineBoxSpaceWrapper

log = logs.get_log(__name__)


# Use start() to parameterize environment. Parameterizing here leads to combinitorial splosion.
register(
    id='Deepdrive-v0',
    entry_point='sim.gym_env:DeepDriveEnv',
    kwargs=dict(),
)


def start(**kwargs):
    all_kwargs = dict(experiment=None, env_id='Deepdrive-v0', sess=None, start_dashboard=True,
                      should_benchmark=True, cameras=None, use_sim_start_command=False, render=False,
                      fps=c.DEFAULT_FPS, combine_box_action_spaces=False, is_discrete=False,
                      preprocess_with_tensorflow=False, is_sync=False, driving_style=DrivingStyle.NORMAL,
                      reset_returns_zero=True, is_remote_client=False, enable_traffic=False, ego_mph=None,
                      view_mode_period=None, max_steps=None)

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

        if kwargs['experiment'] is None:
            kwargs['experiment'] = ''

        _env = env.unwrapped

        # This becomes our constructor - to facilitate using Gym API without registering combinations of params for the
        # wide variety of configurations we want.
        _env.is_discrete = kwargs['is_discrete']
        _env.preprocess_with_tensorflow = kwargs['preprocess_with_tensorflow']
        _env.is_sync = kwargs['is_sync']
        _env.reset_returns_zero = kwargs['reset_returns_zero']
        _env.init_action_space()
        _env.fps = kwargs['fps']
        _env.experiment = kwargs['experiment'].replace(' ', '_')
        _env.period = _env.sync_step_time = 1. / kwargs['fps']
        _env.driving_style = kwargs['driving_style']
        _env.should_render = kwargs['render']
        _env.enable_traffic = kwargs['enable_traffic']
        _env.ego_mph = kwargs['ego_mph']
        _env.view_mode_controller = ViewModeController(period=kwargs['view_mode_period'])
        _env.max_steps = kwargs['max_steps']
        _env.set_use_sim_start_command(kwargs['use_sim_start_command'])
        if kwargs['use_sim_start_command']:
            # TODO: Find a better way to do this. Waiting for the hwnd and focusing does not work in windows.
            input('Press any key when the game has loaded')
        _env.connect(kwargs['cameras'])
        _env.set_step_mode()

        if kwargs['combine_box_action_spaces']:
            env = CombineBoxSpaceWrapper(env)
        if kwargs['sess']:
            _env.set_tf_session(kwargs['sess'])
        if kwargs['start_dashboard']:
            _env.start_dashboard()
        if kwargs['should_benchmark']:
            log.info('Benchmarking enabled - will save results to %s', c.RESULTS_DIR)
            _env.init_benchmarking()

        # Monkey patch methods we want to expose to the remote client,
        # so they can be called the same way locally.
        env.change_cameras = env.unwrapped.change_cameras

        env.reset()
    return env



