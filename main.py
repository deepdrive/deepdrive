import argparse

import gym

import gym_deepdrive  # forward register gym enviornment
from utils import get_log
import config as c


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env-id', nargs='?', default='DeepDrive-v0', help='Select the environment to run')
    parser.add_argument('-r', '--record', action='store_true', default=False,
                        help='Records game driving, including recovering from random actions')
    parser.add_argument('-b', '--benchmark', action='store_true', default=False,
                        help='Benchmarks driving performance and records the results to CSV')
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
    args = parser.parse_args()
    if args.train:
        from tensorflow_agent.train import train
        train.run(resume_dir=args.resume_train)
    elif args.net_path or args.baseline:
        from tensorflow_agent import agent

        agent.run(should_record=args.record, net_path=args.net_path, env_id=args.env_id,
                  should_benchmark=args.benchmark, run_baseline_agent=args.baseline)
    elif args.manual:
        done = False
        render = False
        episode_count = 1
        env = gym.make(args.env_id)
        env = gym.wrappers.Monitor(env, directory=c.GYM_DIR, force=True)
        env.seed(0)
        try:
            deepdrive_env = env.env
            deepdrive_env.start_dashboard()
            if args.benchmark:
                log.info('Benchmarking enabled - will save results to %s', c.BENCHMARK_DIR)
                deepdrive_env.init_benchmarking()

            log.info('Manual drive mode')
            for episode in range(episode_count):
                if episode == 0 or done:
                    obz = env.reset()
                else:
                    obz = None

                while True:
                    action = deepdrive_env.get_noop_action_array()
                    obz, reward, done, _ = env.step(action)
                    if render:
                        env.render()
                    if done:
                        env.reset()
        except KeyboardInterrupt:
            log.info('keyboard interrupt detected, closing')
            env.close()
        except Exception as e:
            env.close()
            raise e
        env.close()
        log.info('Last episode complete, closing')


log = get_log(__name__)

if __name__ == '__main__':
    main()

