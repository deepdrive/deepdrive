import argparse
import glob
import logging
import os
import time

import tensorflow as tf

import camera_config
import config as c
import deepdrive
import logs
from agents.dagger import net
from agents.dagger.agent import ensure_mnet2_baseline_weights
from gym_deepdrive.envs.deepdrive_gym_env import DrivingStyle


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
    parser.add_argument('--discrete-actions', action='store_true', default=False,
                        help='Trains tensorflow agent on stored driving data')
    parser.add_argument('--use-last-model', action='store_true', default=False,
                        help='Run the most recently trained model')
    parser.add_argument('--recording-dir', nargs='?', default=c.RECORDING_DIR, help='Where to store and read recorded '
                                                                                    'environment data from')
    parser.add_argument('--render', action='store_true', default=False,
                        help='SLOW: render of camera data in Python - Use Unreal for real time camera rendering')
    parser.add_argument('--sync', action='store_true', default=False,
                        help='Use synchronous stepping mode where the simulation advances only when calling step')
    parser.add_argument('--record-recovery-from-random-actions', action='store_true', default=False,
                        help='Whether to occasionally perform random actions and record recovery from them')
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
    parser.add_argument('--net-type', nargs='?', default=net.ALEXNET_NAME,
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
    parser.add_argument('-n', '--experiment-name', nargs='?', default=None, help='Name of your experiment')
    parser.add_argument('--fps', type=int, default=c.DEFAULT_FPS, help='Frames / steps per second')
    parser.add_argument('--agent', nargs='?', default=c.DAGGER_MNET2,
                        help='Agent type (%s, %s, %s)' % (c.DAGGER,
                                                          c.DAGGER_MNET2,
                                                          c.BOOTSTRAPPED_PPO2))

    args = parser.parse_args()
    if args.verbose:
        logs.set_level(logging.DEBUG)

    if args.camera_rigs:
        camera_rigs = camera_config.rigs[args.camera_rigs]
    else:
        camera_rigs = camera_config.rigs['baseline_rigs']

    if args.use_last_model:
        if args.train:
            args.resume_train = get_latest_model()
        else:
            args.net_path = get_latest_model()

    driving_style = DrivingStyle[args.driving_style.upper()]

    if args.train:
        train_agent(args, driving_style)
    elif args.path_follower:
        run_path_follower(args, driving_style, camera_rigs)
    else:
        # TODO: Run PPO agent here, not with c.TEST_PPO
        run_trained_agent(args, camera_rigs, driving_style)


def run_trained_agent(args, camera_rigs, driving_style):
    from agents.dagger import agent
    agent.run(args.experiment_name,
              should_record=args.record, net_path=args.net_path, env_id=args.env_id,
              run_baseline_agent=args.baseline, run_mnet2_baseline_agent=args.mnet2_baseline,
              run_ppo_baseline_agent=args.ppo_baseline, render=args.render, camera_rigs=camera_rigs,
              should_record_recovery_from_random_actions=args.record_recovery_from_random_actions, fps=args.fps,
              net_name=args.net_type, is_sync=args.sync, driving_style=driving_style,
              is_remote=args.is_remote_client)


def run_path_follower(args, driving_style, camera_rigs):
    done = False
    episode_count = 1
    gym_env = None
    try:
        cams = camera_rigs
        if isinstance(camera_rigs[0], list):
            cams = cams[0]
        gym_env = deepdrive.start(experiment_name=args.experiment_name, env_id=args.env_id, fps=args.fps,
                                  driving_style=driving_style, is_remote_client=args.is_remote_client,
                                  render=args.render, cameras=cams)
        log.info('Path follower drive mode')
        for episode in range(episode_count):
            if done:
                gym_env.reset()
            while True:
                action = deepdrive.action(has_control=False)
                obz, reward, done, _ = gym_env.step(action)
                if done:
                    gym_env.reset()
    except KeyboardInterrupt:
        log.info('keyboard interrupt detected, closing')
    except Exception as e:
        log.error('Error running agent. %s', e)
        if gym_env:
            gym_env.close()
        raise e
    if gym_env:
        gym_env.close()
    log.info('Last episode complete, closing')


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
                  freeze_pretrained=args.freeze_pretrained)
    elif args.agent == 'bootstrapped_ppo2':
        from agents.bootstrap_rl.train import train
        net_path = args.net_path
        if not net_path:
            log.info('Bootstrapping from baseline agent')
            net_path = ensure_mnet2_baseline_weights(args.net_path)
        train.run(args.env_id, resume_dir=args.resume_train, bootstrap_net_path=net_path, agent_name=args.agent,
                  render=args.render, camera_rigs=[c.DEFAULT_CAM], is_sync=args.sync, driving_style=driving_style,
                  is_remote_client=args.is_remote_client, eval_only=args.eval_only)
    else:
        raise Exception('Agent type not recognized')


def get_latest_model():
    train_dirs = glob.glob('%s/*_train' % c.TENSORFLOW_OUT_DIR)
    latest_subdir = max(train_dirs, key=os.path.getmtime)
    if not latest_subdir:
        raise RuntimeError('Can not get latest model, no models found in % s' % c.TENSORFLOW_OUT_DIR)
    latest_model = max(glob.glob('%s/model.ckpt-*.meta' % latest_subdir), key=os.path.getmtime)
    if not latest_model:
        raise RuntimeError('Can not get latest model, no models found in %s' % latest_subdir)
    latest_prefix = latest_model[:-len('.meta')]
    return latest_prefix


log = logs.get_log(__name__)

if __name__ == '__main__':
    main()

