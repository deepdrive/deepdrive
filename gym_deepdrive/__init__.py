from gym.envs.registration import register

# TODO: Move this into the deepdrive package

register(
    id='DeepDrive-v0',
    entry_point='gym_deepdrive.envs.deepdrive_env:DeepDriveEnv',
    kwargs=dict(
        cameras=[dict(img_shape=(227, 227, 3))],
        preprocess_with_tensorflow=False,
    ),
)

register(
    id='DeepDrivePreproTensorflow-v0',
    entry_point='gym_deepdrive.envs.deepdrive_env:DeepDriveEnv',
    kwargs=dict(
        cameras=[dict(img_shape=(227, 227, 3))],
        preprocess_with_tensorflow=True,
    ),
)
