import sim


def main():
    env = sim.start()
    forward = sim.action(throttle=1, steering=0, brake=0)
    done = False
    while not done:
        observation, reward, done, info = env.step(forward)


if __name__ == '__main__':
    main()
