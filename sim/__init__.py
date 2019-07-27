from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (dict, input, str)

from box import Box
from gym.envs.registration import register

# noinspection PyUnresolvedReferences
import gym
import logs

import config as c
from deepdrive_api.client import Client

# noinspection PyUnresolvedReferences
from sim.action import gym_action as action
from sim.driving_style import DrivingStyle
from sim import driving_style
from sim.sim_args import SimArgs
from sim.view_mode import ViewMode, ViewModeController
from sim import world
from recorder.recorder import Recorder

log = logs.get_log(__name__)


def start(**kwargs):
    """
    Deepdrive gym environment factory.

    This configures and creates a gym environment from parameters passed through main.py

    The parameters are either used to start the environment in process,
    or marshalled and passed to a remote instance of the environment,
    in which case this method is called again on the remote server
    with the same args, minus is_remote_client.

    :param kwargs: Deepdrive gym env configuration
    :return: environment object that implements the gym API (i.e. step, reset, close, render)
    """
    args = SimArgs(**kwargs)
    if args.is_remote_client:
        if isinstance(args.driving_style, DrivingStyle):
            args.driving_style = args.driving_style.as_string()
        args.client_main_args = c.MAIN_ARGS
        env = Client(**(args.to_dict()))
    else:
        env = start_local_env(args)
    return env


def start_local_env(args:SimArgs):
    """
    Acts as a constructor / factory for a local gym environment
    and starts the associated Unreal process
    :param args:
    :return: gym environment
    """
    if isinstance(args.driving_style, str):
        args.driving_style = DrivingStyle.from_str(args.driving_style)
    env = gym.make(args.env_id)
    env.seed(c.RNG_SEED)
    if args.experiment is None:
        args.experiment = ''
    _env = env.unwrapped
    _env.is_discrete = args.is_discrete
    _env.preprocess_with_tensorflow = args.preprocess_with_tensorflow
    _env.is_sync = args.is_sync
    _env.reset_returns_zero = args.reset_returns_zero
    _env.init_action_space()
    _env.target_fps = args.fps
    _env.experiment = args.experiment.replace(' ', '_')
    _env.sim_step_time = args.sim_step_time
    _env.period = 1. / args.fps
    _env.driving_style = args.driving_style
    _env.should_render = args.render
    _env.enable_traffic = args.enable_traffic
    _env.ego_mph = args.ego_mph
    _env.view_mode_controller = ViewModeController(period=args.view_mode_period)
    _env.max_steps = args.max_steps
    _env.max_episodes = args.max_episodes
    _env.set_use_sim_start_command(args.use_sim_start_command)
    _env.image_resize_dims = args.image_resize_dims
    add_recorder(_env, args)
    _env.should_normalize_image = args.should_normalize_image
    _env.randomize_sun_speed = args.randomize_sun_speed
    _env.randomize_shadow_level = args.randomize_shadow_level
    _env.randomize_month = args.randomize_month
    _env.is_botleague = args.is_botleague

    connect_to_unreal(_env, args)
    _env.set_step_mode()
    if args.sess:
        _env.set_tf_session(args.sess)
    if args.start_dashboard:
        _env.start_dashboard()
    log.info('Will save results to %s', c.RESULTS_DIR)
    _env.init_benchmarking()

    monkey_patch_env_api(env)

    env.reset()
    return env


def add_recorder(_env, args: SimArgs):
    if args.is_remote_client:
        main_args = args.client_main_args
    else:
        main_args = c.MAIN_ARGS
    _env.recorder = Recorder(args.recording_dir,
                             should_record=args.should_record,
                             eval_only=args.eval_only,
                             should_upload_gist=args.upload_gist,
                             public=args.public,
                             main_args=main_args,
                             is_botleague=args.is_botleague)


def monkey_patch_env_api(env):
    """
    Monkey patch methods we want in the env API (c.f. api/client.py)
    :param env: gym environment to patch
    """
    env.change_cameras = env.unwrapped.change_cameras


def connect_to_unreal(_env, args):
    """
    Start or connect to an existing instance of the Deepdrive
    simulator running in Unreal Engine
    """
    if args.use_sim_start_command:
        # TODO: Find a better way to do this. Waiting for the hwnd and focusing does not work in windows.
        input('Press any key when the game has loaded')
    _env.connect(args.cameras)


# Use start() to parameterize environment.
# Parameterizing here leads to combinitorial splosion.
register(
    id='Deepdrive-v0',
    entry_point='sim.gym_env:DeepDriveEnv',
    kwargs=dict(),
)
