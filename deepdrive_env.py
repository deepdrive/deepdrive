import gym_deepdrive  # forward registers gym enviornment
import gym

def start(env='DeepDrive-v0'):
    return gym.make(env)
