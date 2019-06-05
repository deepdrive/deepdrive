from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import json
import sys

import h5py  # Needs to be imported before tensorflow to avoid seg faults

import glob
import logging
import os
import traceback

# noinspection PyUnresolvedReferences
import config.check_bindings

from config import camera_config, MOBILENET_V2_IMAGE_SHAPE
import config as c
from sim import DrivingStyle, SimArgs
from util.ensure_sim import get_sim_path
import sim
import logs
from util.args import Args

log = logs.get_log(__name__)


def add_standard_args(args:Args):
    args.add(
        '-e', '--env-id', nargs='?', default='Deepdrive-v0',
        help='Select the environment to run')
    args.add(
        '-r', '--record', action='store_true', default=False,
        help='Records game driving, including recovering from random actions')
    args.add(
        '--discrete-actions', action='store_true', default=False,
        help='Use discrete, rather than continuous actions')
    args.add(
        '--recording-dir', nargs='?', default=c.RECORDING_DIR,
        help='Where to store and read recorded environment data from')
    args.add(
        '--render', action='store_true', default=False,
        help='Show the cameras as seen your agents in Python')
    args.add(
        '--sync', action='store_true', default=False,
        help='Use synchronous stepping mode where the simulation advances only '
             'when calling step')
    args.add(
        '--sim-step-time', type=float,
        default=c.DEFAULT_SIM_STEP_TIME,
        help='Time to pause sim in synchronous stepping mode')
    args.add(
        '--enable-traffic', action='store_true', default=False,
        help='Enable traffic within the simulator')
    args.add(
        '--randomize-sun-speed', action='store_true', default=False,
        help='Whether to randomize the virtual speed of the earth\'s orbit '
             'around the sun')
    args.add(
        '--randomize-view-mode', action='store_true', default=False,
        help='Whether to randomize view mode on episode reset')
    args.add(
        '--randomize-shadow-level', action='store_true', default=False,
        help='Whether to randomize virtual position of Earth around Sun via '
             'month')
    args.add(
        '--randomize-month', action='store_true', default=False,
        help='Whether to randomize shadow quality render levels')
    args.add(
        '--path-follower', action='store_true', default=False,
        help='Whether to let the in-game path follower drive')
    args.add(
        '--eval-only', action='store_true', default=False,
        help='Whether to just run evaluation, i.e. disable gradient updates', )
    args.add(
        '--driving-style', nargs='?',
        default=DrivingStyle.NORMAL.as_string(),
        help='Speed vs comfort prioritization, i.e. ' +
             ', '.join([level.name.lower() for level in
                        DrivingStyle]))
    args.add(
        '--remote', action='store_true', default=False,
        help='Use API to connect to a remote environment')
    args.add(
        '-v', '--verbose',
        help='Increase output verbosity', action='store_true')
    args.add(
        '--camera-rigs', nargs='?', default=None,
        help='Name of camera rigs to use')
    args.add(
        '--experiment', nargs='?', default=None,
        help='Name of your experiment')
    args.add(
        '--fps', type=int, default=c.DEFAULT_FPS,
        help='Frames or steps per second')
    args.add(
        '--ego-mph', type=float, default=25,
        help='Ego (i.e. main) agent vehicle miles per hour')
    args.add(
        '--view-mode-period', type=int, default=None,
        help='Number of steps between view mode switches')
    args.add(
        '--max-steps', type=int, default=None,
        help='Max number of steps to run per episode')
    args.add(
        '--max-episodes', type=int, default=None,
        help='Maximum number of episodes')
    args.add(
        '--server', action='store_true', default=False,
        help='Run as an API server', )
    args.add(
        '--botleague', action='store_true', default=False,
        help='This is a botleague server', )
    args.add(
        '--upload-gist', action='store_true', default=False,
        help='Upload a private gist with driving performance'
             'stats csv files', )
    args.add(
        '--public', action='store_true', default=False,
        help='Results will be made public, i.e. artifacts like '
             'https://gist.github.com/deepdrive-results/cce0a164498c17269ce2adea2a88ec95', )

    args.add(
        '--image-resize-dims', nargs='?',
        default=json.dumps(MOBILENET_V2_IMAGE_SHAPE),
        help='Resize the image coming from the cameras. This was added as '
             'we trained MNET (224x224) on old AlexNet data (227x227), and'
             'wanted to test using the same transformation.')


def add_agent_args(args):
    args.add_agent_arg(
        '--baseline', action='store_true', default=False,
        help='Runs pretrained alexnet-based imitation learning based agent')
    args.add_agent_arg(
        '--mnet2-baseline', action='store_true', default=False,
        help='Runs pretrained mnet2-based imitation learning based agent')
    args.add_agent_arg(
        '--ppo-baseline', action='store_true', default=False,
        help='Runs pretrained ppo-based imitation learning based agent')
    args.add_agent_arg(
        '-t', '--train', action='store_true', default=False,
        help='Trains tensorflow agent on stored driving data')
    args.add_agent_arg(
        '--hdf5-2-tfrecord', action='store_true', default=False,
        help='Converts all recorded hdf5 files to a tfrecord dataset')
    args.add_agent_arg(
        '--use-latest-model', action='store_true', default=False,
        help='Use most recently trained model')
    args.add_agent_arg(
        '--jitter-actions', action='store_true', default=False,
        help='Whether to occasionally perform random actions and record recovery from them')
    args.add_agent_arg(
        '--overfit', action='store_true', default=False,
        help='Whether or not to overfit to a small test set during training to sanity check '
             'convergability')
    args.add_agent_arg(
        '--net-path', nargs='?', default=None,
        help='Path to the tensorflow checkpoint you want to test drive. '
             'i.e. /home/a/DeepDrive/tensorflow/2018-01-01__11-11-11AM_train/model.ckpt-98331')
    args.add_agent_arg(
        '--net-type', nargs='?', default=None,
        help='Your model type - i.e. AlexNet or MobileNetV2')
    args.add_agent_arg(
        '--resume-train', nargs='?', default=None,
        help='Name of the tensorflow training session you want '
             'to resume within %s, '
             'i.e. 2018-01-01__11-11-11AM_train' %
             c.TENSORFLOW_OUT_DIR)
    args.add_agent_arg(
        '--tf-debug', action='store_true', default=False,
        help='Run a tf_debug session')
    args.add_agent_arg(
        '--freeze-pretrained', action='store_true',
        default=False,
        help='Freeze pretrained layers during training')
    args.add_agent_arg(
        '--train-args-collection', nargs='?', default=None,
        help='Name of the set of training args to use')
    args.add_agent_arg(
        '--agent', nargs='?', default=c.DAGGER_MNET2,
        help='Agent type (%s, %s, %s)' % (c.DAGGER,
                                          c.DAGGER_MNET2,
                                          c.BOOTSTRAPPED_PPO2))


def get_args():
    args = Args()
    add_standard_args(args)
    add_agent_args(args)
    all_args = args.all.parse_args()
    agent_args = args.agent.parse_known_args()
    return all_args, agent_args


def main():
    args, agent_args = get_args()
    c.MAIN_ARGS = vars(args)  # For documenting runs
    if args.verbose:
        logs.set_level(logging.DEBUG)

    if args.public:
        if not c.PUBLIC:
            answer = input('Please confirm you want to make the results '
                           'of this evaluation public? ')
            args.public = answer.lower() in ['y', 'yes']
            if not args.public:
                print('Answer was not "y" or "yes", not making public')

    if args.hdf5_2_tfrecord:
        from agents.dagger.train import hdf5_to_tfrecord
        hdf5_to_tfrecord.encode(hdf5_path=args.recording_dir,
                                experiment=args.experiment)
        return
    elif args.server:
        from deepdrive_api import server
        sim_args = None
        if args.botleague:
            log.info('Starting a botleague evaluation')
        if len(sys.argv) > 2:
            # More than just --server was passed,
            # so sim will be configured purely on the server side,
            # vs purely from the client.
            sim_args = get_sim_args_from_command_args(args)
        server.start(sim, get_sim_path(), sim_args=sim_args)
        return
    else:
        camera_rigs = get_camera_rigs(args)
        driving_style = DrivingStyle.from_str(args.driving_style)
        if args.path_follower:
            run_path_follower(args, driving_style, camera_rigs)
        else:
            run_tf_based_models(args, camera_rigs, driving_style)


def run_tf_based_models(args, camera_rigs, driving_style):
    from install import check_tensorflow_gpu
    if not check_tensorflow_gpu():
        raise RuntimeError('Tensorflow not installed, cannot run or '
                           'trained tensorflow agents')
    configure_net_args(args)
    if args.train or args.agent == c.BOOTSTRAPPED_PPO2:
        # Training and running are more tightly linked in RL, so we
        # call train_agent even for eval_only.
        train_agent(args, driving_style)
    else:
        run_agent(args, camera_rigs)


def get_camera_rigs(args):
    if args.camera_rigs:
        camera_rigs = camera_config.rigs[args.camera_rigs]
    else:
        camera_rigs = camera_config.rigs['baseline_rigs']
    return camera_rigs


def configure_net_args(args):
    if args.use_latest_model:
        if args.net_path:
            raise ValueError('--use-latest-model and '
                             '--net-path cannot both be set')
        if args.train:
            args.resume_train = get_latest_model()
        else:
            args.net_path = get_latest_model()
    elif args.net_path:
        if args.net_path.startswith('https://'):
            url = str(args.net_path)
            import utils
            args.net_path = utils.download_weights(url)
        if os.path.isdir(args.net_path):
            args.net_path = get_latest_model_from_path(args.net_path)

    from agents.dagger import net
    if args.net_type is None:
        args.net_type = c.MOBILENET_V2_NAME
    if args.mnet2_baseline:
        args.net_type = c.MOBILENET_V2_NAME

    if args.ppo_baseline and not args.agent:
        # args.agent / agent_name are use in training, but
        # training and running are more tightly linked in RL, so we
        # want to make sure agent is set here even for just running a baseline.
        args.agent = c.BOOTSTRAPPED_PPO2


def run_agent(args, camera_rigs):
    """
    Here we run an agent alongside an open simulator and either just benchmark
    it's performance, as with agents trained offline (i.e. the current dagger
    mnet and alexnet agents), or train an online agent (i.e. the PP02 agent).

    :param args Command line args that configure agent and sim
    :param camera_rigs: A collection of camera configs to cycle through, with
    one rig used for the duration of an episode
    """
    from agents.dagger import agent
    sim_args = get_sim_args_from_command_args(args)
    agent.run(sim_args,
              net_path=args.net_path,
              run_baseline_agent=args.baseline,
              run_mnet2_baseline_agent=args.mnet2_baseline,
              run_ppo_baseline_agent=args.ppo_baseline,
              camera_rigs=camera_rigs,
              should_jitter_actions=args.jitter_actions,
              net_name=args.net_type,
              max_episodes=args.max_episodes,
              agent_name=args.agent,)


def get_sim_args_from_command_args(args):
    sim_args = SimArgs(
        experiment=args.experiment,
        env_id=args.env_id,
        should_record=args.record,
        render=args.render,
        # cameras will be set in agent
        fps=args.fps,
        is_sync=args.sync,
        driving_style=args.driving_style,
        is_remote_client=args.remote,
        recording_dir=args.recording_dir,
        enable_traffic=args.enable_traffic,
        view_mode_period=args.view_mode_period,
        max_steps=args.max_steps,
        max_episodes=args.max_episodes,
        eval_only=args.eval_only,
        upload_gist=args.upload_gist, public=args.public,
        sim_step_time=args.sim_step_time,
        randomize_view_mode=args.randomize_view_mode,
        randomize_sun_speed=args.randomize_sun_speed,
        randomize_shadow_level=args.randomize_shadow_level,
        randomize_month=args.randomize_month,
        image_resize_dims=tuple(json.loads(args.image_resize_dims)),
    )
    return sim_args


def run_path_follower(args, driving_style, camera_rigs):
    """
    Runs the C++ PID-based path follower agent which uses a reference
    spline in the center of the lane, and speed annotations on tight turns
    to drive.
    Refer to https://github.com/deepdrive/deepdrive-sim/tree/b21e0a0bf8cec60538425fa41b5fc5ee28142556/Plugins/DeepDrivePlugin/Source/DeepDrivePlugin/Private/Simulation/Agent
    """
    done = False
    episode_count = 1
    gym_env = None
    try:
        cams = camera_rigs
        if isinstance(camera_rigs[0], list):
            cams = cams[0]
        gym_env = sim.start(experiment=args.experiment, env_id=args.env_id,
                            fps=args.fps,
                            driving_style=driving_style,
                            is_remote_client=args.remote,
                            render=args.render, cameras=cams,
                            enable_traffic=args.enable_traffic,
                            ego_mph=args.ego_mph)
        log.info('Path follower drive mode')
        for episode in range(episode_count):
            if done:
                gym_env.reset()
            while True:
                action = sim.action(has_control=False)
                obz, reward, done, _ = gym_env.step(action)
                if done:
                    gym_env.reset()
    except KeyboardInterrupt:
        log.info('keyboard interrupt detected, closing')
    except Exception as e:
        log.error('Error running agent. %s', e)
        print(traceback.format_exc())
    else:
        log.info('Last episode complete, closing')
    finally:
        if gym_env:
            gym_env.close()


def train_agent(args, driving_style):
    from agents.dagger.agent import ensure_mnet2_baseline_weights
    if args.agent == c.DAGGER or args.agent == c.DAGGER_MNET2:
        train_dagger(args)
    elif args.agent == c.BOOTSTRAPPED_PPO2:
        from agents.bootstrap_rl.train import train
        net_path = args.net_path
        if not net_path:
            log.info('Bootstrapping from baseline agent')
            net_path = ensure_mnet2_baseline_weights(args.net_path)
        if not args.sync and not args.eval_only:
            args.sync = True
            log.warning('Detected training RL in async mode which '
                        'can cause unequal time deltas. '
                        'Switching to synchronous mode. '
                        'Use --sync to avoid this.')

        train.run(args.env_id, resume_dir=args.resume_train,
                  bootstrap_net_path=net_path, agent_name=args.agent,
                  render=args.render, camera_rigs=[c.DEFAULT_CAM],
                  is_sync=args.sync, driving_style=driving_style,
                  is_remote_client=args.remote, eval_only=args.eval_only)
    else:
        raise Exception('Agent type not recognized')


def train_dagger(args):
    """
    Run the first iteration of DAgger where our policy is random.
    """
    from agents.dagger.train import train
    train.run(resume_dir=args.resume_train, data_dir=args.recording_dir,
              agent_name=args.agent,
              overfit=args.overfit, eval_only=args.eval_only,
              tf_debug=args.tf_debug,
              freeze_pretrained=args.freeze_pretrained,
              train_args_collection_name=args.train_args_collection)


def get_latest_model():
    # TODO: Get best performing model from n latest
    return get_latest_model_from_path('%s/*' % c.TENSORFLOW_OUT_DIR)


def get_latest_model_from_path(model_dir):
    model = max(glob.glob(
        '%s/model.ckpt-*.meta' % model_dir),
        key=os.path.getmtime)
    if not model:
        raise RuntimeError('No tensorflow models found in %s' % model_dir)
    prefix = model[:-len('.meta')]
    log.info('Latest model is %s', prefix)
    return prefix


if __name__ == '__main__':
    main()
