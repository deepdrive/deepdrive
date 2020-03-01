import time

import sim
from sim.action import Action

def main():
    env = sim.start(is_remote_client=True, render=True, is_sync=True)
    forward = Action(throttle=1)
    done = False
    while env.is_open:
        while not done:
            observation, reward, done, info = env.step(forward)
        env.reset()
        print('Episode finished')
        done = False
    print('Env closed')

if __name__ == '__main__':
    main()
