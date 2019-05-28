from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from future.builtins import (int, range)
import numpy as np
import pytest
from numpy.random import RandomState
import tempfile
import os

os.environ['DEEPDRIVE_DIR'] = os.path.join(tempfile.gettempdir(), 'testdeepdrive')

import utils
from sim.reward_calculator import RewardCalculator

try:
    import tensorflow as tf
    import tf_utils
except ImportError:
    print('Tensorflow not found, skipping tests')
    tf = None
    tf_utils = None


@pytest.fixture()
def tf_sess():
    if tf:
        with tf.Session() as sess:
            yield sess
    else:
        yield None


def test_lane_deviation_penalty():
    penalty = RewardCalculator.get_lane_deviation_penalty(lane_deviation=100, time_passed=0.1)
    assert penalty == pytest.approx(0)
    penalty = RewardCalculator.get_lane_deviation_penalty(lane_deviation=300, time_passed=0.1)
    assert penalty == pytest.approx(9)
    penalty = RewardCalculator.get_lane_deviation_penalty(lane_deviation=300, time_passed=1e8)
    assert penalty == pytest.approx(100)
    penalty = RewardCalculator.get_lane_deviation_penalty(lane_deviation=300, time_passed=1e-8)
    assert penalty == pytest.approx(0, abs=1e-6)
    penalty = RewardCalculator.get_lane_deviation_penalty(lane_deviation=0, time_passed=0.1)
    assert penalty == pytest.approx(0)
    penalty = RewardCalculator.get_lane_deviation_penalty(lane_deviation=1e8, time_passed=0.1)
    assert penalty == pytest.approx(100)
    with pytest.raises(ValueError):
        RewardCalculator.get_lane_deviation_penalty(lane_deviation=-1, time_passed=0.1)


def test_gforce_penalty():
    penalty = RewardCalculator.get_gforce_penalty(gforces=1, time_passed=0.1)
    assert penalty == pytest.approx(2.4)
    penalty = RewardCalculator.get_gforce_penalty(gforces=5, time_passed=1e8)
    assert penalty == pytest.approx(100)
    penalty = RewardCalculator.get_gforce_penalty(gforces=5, time_passed=1e-8)
    assert penalty == pytest.approx(0, abs=1e-5)
    penalty = RewardCalculator.get_gforce_penalty(gforces=0, time_passed=0.1)
    assert penalty == pytest.approx(0)
    penalty = RewardCalculator.get_gforce_penalty(gforces=1e8, time_passed=0.1)
    assert penalty == pytest.approx(100)
    with pytest.raises(ValueError):
        RewardCalculator.get_gforce_penalty(gforces=-5, time_passed=1e8)


def test_progress_reward():
    progress_reward, speed_reward, _ = RewardCalculator.get_progress_and_speed_reward(progress=100, time_passed=0.1)
    assert progress_reward == pytest.approx(1.) and speed_reward == pytest.approx(1.5)
    progress_reward, speed_reward, _ = RewardCalculator.get_progress_and_speed_reward(progress=100, time_passed=1)
    assert progress_reward == pytest.approx(1.) and speed_reward == pytest.approx(0.15)
    progress_reward, speed_reward, _ = RewardCalculator.get_progress_and_speed_reward(progress=3, time_passed=0.1)
    assert progress_reward == pytest.approx(0.03) and speed_reward == pytest.approx(0.00135)
    progress_reward, speed_reward, _ = RewardCalculator.get_progress_and_speed_reward(progress=3, time_passed=1e-8)
    assert progress_reward == pytest.approx(0.03, abs=1e-6) and speed_reward == pytest.approx(100.0)  # Should clip
    progress_reward, speed_reward, _ = RewardCalculator.get_progress_and_speed_reward(progress=0, time_passed=0.1)
    assert progress_reward == pytest.approx(0.) and speed_reward == pytest.approx(0.)
    progress_reward, speed_reward, _ = RewardCalculator.get_progress_and_speed_reward(progress=-10, time_passed=0.1)
    assert progress_reward == pytest.approx(-0.1) and speed_reward == pytest.approx(-0.015)
    progress_reward, speed_reward, _ = RewardCalculator.get_progress_and_speed_reward(progress=1e8, time_passed=0.1)
    assert progress_reward == pytest.approx(100.) and speed_reward == pytest.approx(100.)  # Should clip
    progress_reward, speed_reward, _ = RewardCalculator.get_progress_and_speed_reward(progress=-1e8, time_passed=0.1)
    assert progress_reward == pytest.approx(0.) and speed_reward == pytest.approx(0.)  # lap complete, zero out

    # Test invariance of sampling frequency
    p1, r1 = episode_progress_reward(hz=1,   total_secs=10)
    p2, r2 = episode_progress_reward(hz=2,   total_secs=10)
    p3, r3 = episode_progress_reward(hz=0.5, total_secs=10)
    assert p1 == pytest.approx(p2) and r1 == pytest.approx(r2) and p2 == pytest.approx(p3) and r2 == pytest.approx(r3)


def episode_progress_reward(hz, total_secs):
    total_progress = total_speed_reward = 0
    time_passed = 1 / hz
    progress = 1000 / hz
    for i in range(int(total_secs * hz)):
        progress_reward, speed_reward, _ = RewardCalculator.get_progress_and_speed_reward(progress=progress,
                                                                                       time_passed=time_passed)
        total_progress += progress_reward
        total_speed_reward += speed_reward
    return total_progress, total_speed_reward


def test_preprocess_image(tf_sess):
    rng = RandomState(0)
    img = rng.rand(1920, 1200)
    p1 = utils.preprocess_image(img)
    if tf_sess is not None:
        p2 = tf_utils.preprocess_image(img, tf_sess)
        assert np.max(p1 - p2) <= 1
    assert p1[0][0] == 194
    assert p1[-1][-1] == 193
    assert p1[p1.shape[0] // 2][p1.shape[1] // 2] == 126
    assert list(p1[p1.shape[0] // 3])[:100] == [179, 149, 229, 252, 225, 223, 243, 96, 211, 198, 212, 249, 190, 247,
                                                197, 241, 163, 172, 247, 237, 101, 120, 41, 90, 206, 94, 185, 73, 172,
                                                158, 225, 224, 207, 207, 248, 194, 150, 250, 242, 146, 211, 215, 237,
                                                40, 244, 239, 202, 208, 233, 147, 123, 251, 199, 168, 250, 254, 144,
                                                252, 224, 215, 227, 216, 199, 161, 115, 253, 135, 77, 105, 183, 176,
                                                143, 117, 202, 203, 243, 122, 215, 219, 124, 187, 254, 247, 133, 76, 39,
                                                242, 205, 236, 192, 192, 103, 193, 251, 35, 248, 146, 221, 46, 186]


def test_preprocess_depth(tf_sess):
    rng = RandomState(0)
    depth = rng.rand(1920, 1200)
    p1 = utils.preprocess_depth(depth)
    if tf_sess is not None:
        p2 = tf_utils.preprocess_image(depth, tf_sess)
        assert np.max(p1 - p2) <= 1
    actual = list(p1.flatten()[p1.size // 3:p1.size // 3 + 100])
    expected = [0.002514684773514457, 0.0041350987147154988, 0.00067722453287451874, 6.0382635299006502e-05,
                0.00082243375692568252, 0.000881154928328925, 0.0003015844024133575, 0.0089413299711420015,
                0.0012668913602365804, 0.0017493899057672484, 0.0012297302821309404, 0.00014536841866812092,
                0.0020726431321461147, 0.00019011504751303418, 0.0017598160353674602, 0.00034701931169399186,
                0.0033050015045214651, 0.0028537698444585334, 0.00018769877621070891, 0.00046990796258813728,
                0.0082884580709213836, 0.0063293130559742986, 0.02410169598058122, 0.0098601405999210537,
                0.0014280333911834955, 0.009190239693315394, 0.0022671623482907188, 0.01291206819927891,
                0.0028541757203758389, 0.0035968311309652603, 0.00080925787214987009, 0.00084134980953235545,
                0.0014126927712115813, 0.0013990039275166234, 0.00016829517477263774, 0.0018799619925731936,
                0.0040850057101084773, 0.00010090817999749792, 0.00033563212584004454, 0.0043546132884055479,
                0.0012573811077964459, 0.0011374194166438882, 0.00045884926209596846, 0.02479401838835903,
                0.00027300163051291075, 0.00039739814613985009, 0.0015890552578129568, 0.0013716619839378632,
                0.0005784882590990733, 0.0042358330268640679, 0.0060610271132032099, 9.6305884230573046e-05,
                0.0016883902282087526, 0.0030552248204359788, 0.00011060120551770693, 9.7939084781005703e-07,
                0.0044259935197632737, 5.5585308283770396e-05, 0.00085706376157983081, 0.0011429927648087563,
                0.00073759259177592473, 0.0010953049908472632, 0.0016877924426287776, 0.0034122647675984574,
                0.0068016515868866353, 4.6687312682328084e-05, 0.0050817640491964402, 0.01197324427120547,
                0.0078595122255692915, 0.0023387198527116881, 0.0026548165841310317, 0.0045282857380305784,
                0.0065714183871598449, 0.0015699930675901715, 0.0015339906295552164, 0.00030075231572516281,
                0.0061212924601392986, 0.0011359644398976096, 0.0010003071142535968, 0.0059864809477887075,
                0.0021943800369656633, 1.4161412765171874e-05, 0.00017803344629484447, 0.005255369976266736,
                0.012265895735866821, 0.0251301315909821, 0.00032247668159115258, 0.0014601893690483472,
                0.00049330646650431163, 0.0019625317886386704, 0.0019577928547728561, 0.0080632104203648101,
                0.0019493655410045428, 8.9598907012599079e-05, 0.028066647603396212, 0.00017480272492269728,
                0.0043565111781060008, 0.00094263459389571725, 0.021517855433459819, 0.002208296477107196]
    assert np.max(actual - np.array(expected)) < 1e-7
