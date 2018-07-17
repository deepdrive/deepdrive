import deepdrive as dd

def main():
    env = dd.start()
    forward = dd.action(throttle=1, steering=0, brake=0)
    done = False
    while not done:
        observation, reward, done, info = env.step(forward)

if __name__ == '__main__':
    main()
