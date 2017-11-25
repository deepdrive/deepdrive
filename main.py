import argparse

import gym_deepdrive  # do not remove, registers enviornment

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env-id', nargs='?', default='DeepDrive-v0', help='Select the environment to run')
    parser.add_argument('-r', '--record', action='store_true', default=False,
                        help='Records game driving, including recovering from random actions')
    parser.add_argument('-b', '--benchmark', action='store_true', default=False,
                        help='Average score over five laps, with one lap warmup.')
    parser.add_argument('-t', '--train', action='store_true', default=False,
                        help='Trains tensorflow agent on stored driving data')
    parser.add_argument('-n', '--net-path', nargs='?', default=None,
                        help='Path to the tensorflow checkpoint you want to test drive. '
                             'i.e. /home/a/DeepDrive/tensorflow/2018-01-01__11-11-11AM_train/model.ckpt-98331')
    parser.add_argument('-c', '--resume-train', nargs='?', default=None,
                        help='Path to the tensorflow training session you want to resume, '
                             'i.e. /home/a/DeepDrive/tensorflow/2018-01-01__11-11-11AM_train')
    args = parser.parse_args()
    if args.train:
        from tensorflow_agent.train import train
        train.run(resume_dir=args.resume_train)
    else:
        from tensorflow_agent import agent

        agent.run(should_record=args.record, net_path=args.net_path, env_id=args.env_id,
                  should_benchmark=args.benchmark)


if __name__ == '__main__':
    main()

