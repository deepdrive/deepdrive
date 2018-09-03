from gym.envs.registration import register


# Use deepdrive.start to parameterize environment. Parameterizing here leads to combinitorial splosion.
register(
    id='Deepdrive-v0',
    entry_point='sim.gym_env:DeepDriveEnv',
    kwargs=dict(),
)
