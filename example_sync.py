import sim


def main():
    env = sim.start(is_sync=True, render=True)
    forward = sim.action(throttle=1, steering=0, brake=0)
    while True:
        observation, reward, done, info = env.step(forward)


if __name__ == '__main__':
    main()
