
import deepdrive
from gym_deepdrive.envs.deepdrive_gym_env import Action


def main():
    env = deepdrive.start(is_remote_client=True, render=True)
    forward = Action(throttle=1)
    done = False
    while True:
        while not done:
            observation, reward, done, info = env.step(forward.as_gym())

        print('Episode finished')
        done = env.reset()

if __name__ == '__main__':
    main()
