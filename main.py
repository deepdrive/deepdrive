import argparse
import logging

import camera_config
import config as c
import logs
import deepdrive_env


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env-id', nargs='?', default='DeepDrive-v0', help='Select the environment to run')
    parser.add_argument('-r', '--record', action='store_true', default=False,
                        help='Records game driving, including recovering from random actions')
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='Runs pretrained imitation learning based agent')
    parser.add_argument('--manual', action='store_true', default=False,
                        help='Allows driving manually within the simulator')
    parser.add_argument('-t', '--train', action='store_true', default=False,
                        help='Trains tensorflow agent on stored driving data')
    parser.add_argument('--render', action='store_true', default=False,
                        help='SLOW: render of camera data in Python - Use Unreal for real time camera rendering')
    parser.add_argument('--record-recovery-from-random-actions', action='store_true', default=False,
                        help='Whether to occasionally perform random actions and record recovery from them')
    parser.add_argument('--let-game-drive', action='store_true', default=False,
                        help='Whether to let the in-game path follower drive')
    parser.add_argument('-n', '--net-path', nargs='?', default=None,
                        help='Path to the tensorflow checkpoint you want to test drive. '
                             'i.e. /home/a/DeepDrive/tensorflow/2018-01-01__11-11-11AM_train/model.ckpt-98331')
    parser.add_argument('-c', '--resume-train', nargs='?', default=None,
                        help='Path to the tensorflow training session you want to resume, '
                             'i.e. /home/a/DeepDrive/tensorflow/2018-01-01__11-11-11AM_train')
    parser.add_argument('-v', '--verbose', help='Increase output verbosity',
                        action='store_true')
    parser.add_argument('--camera-rigs', nargs='?', default=None, help='Name of camera rigs to use')

    args = parser.parse_args()
    if args.verbose:
        logs.set_level(logging.DEBUG)

    if args.camera_rigs:
        camera_rigs = camera_config.rigs[args.camera_rigs]
    else:
        camera_rigs = camera_config.rigs['baseline_rigs']

    if args.train:
        from tensorflow_agent.train import train
        train.run(resume_dir=args.resume_train)
    elif args.manual:
        done = False
        render = False
        episode_count = 1
        env = None
        try:
            env = deepdrive_env.start(args.env_id)
            log.info('Manual drive mode')
            for episode in range(episode_count):
                if episode == 0 or done:
                    obz = env.reset()
                else:
                    obz = None

                while True:
                    action = deepdrive_env.action()
                    obz, reward, done, _ = env.step(action)
                    if render:
                        env.render()
                    if done:
                        env.reset()
        except KeyboardInterrupt:
            log.info('keyboard interrupt detected, closing')
        except Exception as e:
            raise e
        finally:
            if env:
                env.close()
        log.info('Last episode complete, closing')
    else:
        from tensorflow_agent import agent
        if args.record and not args.record_recovery_from_random_actions:
            args.let_game_drive = True
        agent.run(should_record=args.record, net_path=args.net_path, env_id=args.env_id,
                  run_baseline_agent=args.baseline, render=args.render, camera_rigs=camera_rigs,
                  should_record_recovery_from_random_actions=args.record_recovery_from_random_actions,
                  let_game_drive=args.let_game_drive)

log = logs.get_log(__name__)

if __name__ == '__main__':
    main()

