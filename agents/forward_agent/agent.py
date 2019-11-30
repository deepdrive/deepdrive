from deepdrive_api.client import get_action, Client


# def main():
#     env = Client(is_remote_client=True, render=True)
#     forward = get_action(throttle=1, steering=0, brake=0)
#     done = False
#     while not done:
#         observation, reward, done, info = env.step(forward)
#     env.close()
#     print('Episode finished')

# Code from lazydriver submission; to test performance
def main():
    env = Client(is_remote_client=True, render=True)
    done = False
    c = 1.0
    while not done:
        forward = get_action(throttle=0.5, steering=-1.0*c, brake=0)
        observation, reward, done, info = env.step(forward)
        c = c * 0.9
    env.close()
    print('Episode end')

if __name__ == '__main__':
    main()