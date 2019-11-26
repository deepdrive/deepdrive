from deepdrive_api.client import get_action, Client


def main():
    env = Client(is_remote_client=True, render=True)
    forward = get_action(throttle=1, steering=0, brake=0)
    done = False
    while not done:
        observation, reward, done, info = env.step(forward)
    env.close()
    print('Episode finished')

if __name__ == '__main__':
    main()