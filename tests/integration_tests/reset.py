import sim


def main():
    # TODO: Add some asserts and get working on Jenkins
    env = sim.start(is_sync=True)
    forward = sim.action(throttle=1, steering=0, brake=0)
    done = False
    i = 0
    while 1:
        i += 1
        observation, reward, done, info = env.step(forward)
        if i % 10 == 0:
            env.reset()



if __name__ == '__main__':
    main()