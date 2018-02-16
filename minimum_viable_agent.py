import deepdrive_env

def main():
    env = deepdrive_env.start()
    forward = deepdrive_env.action(throttle=1, steering=0, brake=0)
    done = False
    while not done:
        observation, reward, done, info = env.step(forward)

if __name__ == '__main__':
    main()
