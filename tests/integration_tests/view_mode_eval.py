import sim
from sim.action import Action


# TODO: Use python docker

# Usage: Please make sure to start a fresh api/server.py as
# connection life cycle is not properly implemented yet

def main():
    env = sim.start(is_remote_client=True, render=True)
    forward = Action(throttle=1)
    done = False
    while True:
        while not done:
            observation, reward, done, info = env.step(forward)

        print('Episode finished')
        done = env.reset()


if __name__ == '__main__':
    main()
