import time

import sim
from sim.uepy_client import rpc
from sim.world import get_agents, get_agent_positions


def main():
    env = sim.start(is_sync=True, render=True, enable_traffic=True)
    forward = sim.action(throttle=1, steering=0, brake=0)
    done = False
    while True:
        while not done:
            observation, reward, done, info = env.step(forward)
            # agents = get_agents()
            # positions = get_agent_positions()
            start = time.time()
            [rpc('get_42') for _ in range(40)]
            print('~~~~~~~~~~ get_42\'s took %r' % (time.time() - start))

        env.reset()
        print('Episode finished')
        done = False


if __name__ == '__main__':
    main()
