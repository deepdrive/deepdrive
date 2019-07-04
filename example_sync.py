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
        env.reset()
        print('Episode finished')
        done = False


if __name__ == '__main__':
    main()
