import pytest
import gym
import gym_deepdrive
from gym_deepdrive.envs.deepdrive_env import DeepDriveRewardCalculator


def test_speed_reward():
    reward = DeepDriveRewardCalculator.get_speed_reward(cmps=833, time_passed=0.1)
    assert reward == pytest.approx(0.59976)

