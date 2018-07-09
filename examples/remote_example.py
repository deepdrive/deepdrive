
import deepdrive
from gym_deepdrive.envs.deepdrive_gym_env import Action


def main():
    env = deepdrive.start()
    forward = Action(throttle=1)
    done = False
    while True:
        while not done:
            observation, reward, done, info = env.step(forward)
        print('Episode finished')
        done = env.reset()

if __name__ == '__main__':
    main()
