import argparse
import logging

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
    parser.add_argument('-n', '--net-path', nargs='?', default=None,
                        help='Path to the tensorflow checkpoint you want to test drive. '
                             'i.e. /home/a/DeepDrive/tensorflow/2018-01-01__11-11-11AM_train/model.ckpt-98331')
    parser.add_argument('-c', '--resume-train', nargs='?', default=None,
                        help='Path to the tensorflow training session you want to resume, '
                             'i.e. /home/a/DeepDrive/tensorflow/2018-01-01__11-11-11AM_train')
    parser.add_argument('-v', '--verbose', help='Increase output verbosity',
                        action='store_true')
    args = parser.parse_args()
    if args.verbose:
        logs.set_level(logging.DEBUG)

    if args.train:
        from tensorflow_agent.train import train
        train.run(resume_dir=args.resume_train)
    elif args.net_path or args.baseline or args.record:
        from tensorflow_agent import agent

        agent.run(should_record=args.record, net_path=args.net_path, env_id=args.env_id,
                  run_baseline_agent=args.baseline)
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


log = logs.get_log(__name__)

if __name__ == '__main__':
    main()

