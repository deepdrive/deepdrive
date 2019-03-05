import argparse
import glob
import logging
import os
import traceback

# noinspection PyUnresolvedReferences
import config.check_bindings

from config import camera_config
import config as c
from agents.dagger import net
from agents.dagger.agent import ensure_mnet2_baseline_weights
from agents.dagger.train import hdf5_to_tfrecord
from sim.driving_style import DrivingStyle
import sim
import logs
log = logs.get_log(__name__)


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env-id', nargs='?', default='Deepdrive-v0', help='Select the environment to run')
    parser.add_argument('-r', '--record', action='store_true', default=False,
                        help='Records game driving, including recovering from random actions')
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='Runs pretrained alexnet-based imitation learning based agent')
    parser.add_argument('--mnet2-baseline', action='store_true', default=False,
                        help='Runs pretrained mnet2-based imitation learning based agent')
    parser.add_argument('--ppo-baseline', action='store_true', default=False,
                        help='Runs pretrained ppo-based imitation learning based agent')
    parser.add_argument('-t', '--train', action='store_true', default=False,
                        help='Trains tensorflow agent on stored driving data')
    parser.add_argument('--hdf5-2-tfrecord', action='store_true', default=False,
                        help='Converts all recorded hdf5 files to a tfrecord dataset')
    parser.add_argument('--discrete-actions', action='store_true', default=False,
                        help='Trains tensorflow agent on stored driving data')
    parser.add_argument('--use-latest-model', action='store_true', default=False,
                        help='Use most recently trained model')
    parser.add_argument('--recording-dir', nargs='?', default=c.RECORDING_DIR, help='Where to store and read recorded '
                                                                                    'environment data from')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Show the cameras as seen your agents in Python')
    parser.add_argument('--sync', action='store_true', default=False,
                        help='Use synchronous stepping mode where the simulation advances only when calling step')
    parser.add_argument('--enable-traffic', action='store_true', default=False,
                        help='Enable traffic within the simulator')
    parser.add_argument('--jitter-actions', action='store_true', default=False,
                        help='Whether to occasionally perform random actions and record recovery from them')
    parser.add_argument('--randomize-sun-speed', action='store_true', default=False,
                        help='Whether to randomize the virtual speed of the earth\'s orbit around the sun')
    parser.add_argument('--randomize-view-mode', action='store_true', default=False,
                        help='Whether to randomize view mode')
    parser.add_argument('--randomize-shadow-level', action='store_true', default=False,
                        help='Whether to randomize virtual position of Earth around Sun via month')
    parser.add_argument('--randomize-month', action='store_true', default=False,
                        help='Whether to randomize shadow quality render levels')
    parser.add_argument('--path-follower', action='store_true', default=False,
                        help='Whether to let the in-game path follower drive')
    parser.add_argument('--overfit', action='store_true', default=False,
                        help='Whether or not to overfit to a small test set during training to sanity check '
                             'convergability')
    parser.add_argument('--eval-only', help='Whether to just run evaluation phase of training', action='store_true',
                        default=False)
    parser.add_argument('--net-path', nargs='?', default=None,
                        help='Path to the tensorflow checkpoint you want to test drive. '
                             'i.e. /home/a/DeepDrive/tensorflow/2018-01-01__11-11-11AM_train/model.ckpt-98331')
    parser.add_argument('--net-type', nargs='?', default=net.MOBILENET_V2_NAME,
                        help='Your model type - i.e. AlexNet or MobileNetV2')
    parser.add_argument('--driving-style', nargs='?', default=DrivingStyle.NORMAL.name.lower(),
                        help='Speed vs comfort prioritization, i.e. ' +
                             ', '.join([level.name.lower() for level in DrivingStyle]))
    parser.add_argument('--resume-train', nargs='?', default=None,
                        help='Name of the tensorflow training session you want to resume within %s, '
                             'i.e. 2018-01-01__11-11-11AM_train' % c.TENSORFLOW_OUT_DIR)
    parser.add_argument('--tf-debug', action='store_true', default=False, help='Run a tf_debug session')
    parser.add_argument('--freeze-pretrained', action='store_true', default=False, help='Freeze pretrained layers '
                                                                                        'during training')
    parser.add_argument('--is-remote-client', action='store_true', default=False,
                        help='Use API to connect to a remote environment')
    parser.add_argument('-v', '--verbose', help='Increase output verbosity',
                        action='store_true')
    parser.add_argument('--camera-rigs', nargs='?', default=None, help='Name of camera rigs to use')
    parser.add_argument('--train-args-collection', nargs='?', default=None, help='Name of the set of training args to '
                                                                                'use')
    parser.add_argument('--experiment', nargs='?', default=None, help='Name of your experiment')
    parser.add_argument('--fps', type=int, default=c.DEFAULT_FPS, help='Frames / steps per second')
    parser.add_argument('--ego-mph', type=float, default=25, help='Ego (i.e. main) agent vehicle miles per hour')
    parser.add_argument('--agent', nargs='?', default=c.DAGGER_MNET2,
                        help='Agent type (%s, %s, %s)' % (c.DAGGER,
                                                          c.DAGGER_MNET2,
                                                          c.BOOTSTRAPPED_PPO2))

    args = c.PY_ARGS = parser.parse_args()
    if args.verbose:
        logs.set_level(logging.DEBUG)

    if args.hdf5_2_tfrecord:
        hdf5_to_tfrecord.encode(hdf5_path=args.recording_dir)
        return

    if args.camera_rigs:
        camera_rigs = camera_config.rigs[args.camera_rigs]
    else:
        camera_rigs = camera_config.rigs['baseline_rigs']

    if args.use_latest_model:
        if args.net_path:
            raise ValueError('--use-latest-model and --net-path cannot both be set')
        if args.train:
            args.resume_train = get_latest_model()
        else:
            args.net_path = get_latest_model()
    elif args.net_path and os.path.isdir(args.net_path):
        args.net_path = get_latest_model_from_path(args.net_path)

    if args.mnet2_baseline:
        args.net_type = net.MOBILENET_V2_NAME

    driving_style = DrivingStyle[args.driving_style.upper()]

    if args.train:
        train_agent(args, driving_style)
    elif args.path_follower:
        run_path_follower(args, driving_style, camera_rigs)
    else:
        # TODO: Run PPO agent here, not with c.TEST_PPO
        run_agent(args, camera_rigs, driving_style)


def run_agent(args, camera_rigs, driving_style):
    from agents.dagger import agent
    agent.run(args.experiment,
              should_record=args.record, net_path=args.net_path, env_id=args.env_id,
              run_baseline_agent=args.baseline, run_mnet2_baseline_agent=args.mnet2_baseline,
              run_ppo_baseline_agent=args.ppo_baseline, render=args.render, camera_rigs=camera_rigs,
              should_jitter_actions=args.jitter_actions, fps=args.fps,
              net_name=args.net_type, is_sync=args.sync, driving_style=driving_style,
              is_remote=args.is_remote_client, recording_dir=args.recording_dir,
              randomize_view_mode=args.randomize_view_mode, randomize_sun_speed=args.randomize_sun_speed,
              randomize_shadow_level=args.randomize_shadow_level, randomize_month=args.randomize_month,
              enable_traffic=args.enable_traffic)


def run_path_follower(args, driving_style, camera_rigs):
    done = False
    episode_count = 1
    gym_env = None
    try:
        cams = camera_rigs
        if isinstance(camera_rigs[0], list):
            cams = cams[0]
        gym_env = sim.start(experiment=args.experiment, env_id=args.env_id, fps=args.fps,
                                  driving_style=driving_style, is_remote_client=args.is_remote_client,
                                  render=args.render, cameras=cams, enable_traffic=args.enable_traffic,
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
    # TODO: Add experiment name here as well, and integrate it into Tensorflow runs, recording names, model checkpoints, etc...
    if args.agent == 'dagger' or args.agent == 'dagger_mobilenet_v2':
        '''
        Really it's just the first iteration of DAgger where our policy is random.
        This seems to be sufficient for exploring the types of mistakes our AI makes and labeling
        corrections to those mistakes. This does a better job at handling edge cases that
        the agent would not encounter acting under its own policy during training.
        In this way, we come a little closer to reinforcement learning, as we explore randomly and cover
        a larger number of possibilities.
        '''
        from agents.dagger.train import train
        train.run(resume_dir=args.resume_train, data_dir=args.recording_dir, agent_name=args.agent,
                  overfit=args.overfit, eval_only=args.eval_only, tf_debug=args.tf_debug,
                  freeze_pretrained=args.freeze_pretrained, train_args_collection_name=args.train_args_collection)
    elif args.agent == 'bootstrapped_ppo2':
        from agents.bootstrap_rl.train import train
        net_path = args.net_path
        if not net_path:
            log.info('Bootstrapping from baseline agent')
            net_path = ensure_mnet2_baseline_weights(args.net_path)
        if not args.sync and not args.eval_only:
            args.sync = True
            log.warning('Detected training RL in async mode which can cause unequal time deltas. '
                        'Switching to synchronous mode. Use --sync to avoid this.')

        train.run(args.env_id, resume_dir=args.resume_train, bootstrap_net_path=net_path, agent_name=args.agent,
                  render=args.render, camera_rigs=[c.DEFAULT_CAM], is_sync=args.sync, driving_style=driving_style,
                  is_remote_client=args.is_remote_client, eval_only=args.eval_only)
    else:
        raise Exception('Agent type not recognized')


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

