import sim
import config
from sim import DrivingStyle


def main():
    env = sim.start(is_sync=True,
                    render=True,
                    enable_traffic=True,
                    experiment='my_experiment',
                    cameras=[config.DEFAULT_CAM],
                    fps=config.DEFAULT_FPS,  # Agent steps per second
                    sim_step_time=config.DEFAULT_SIM_STEP_TIME,
                    is_discrete=False,  # Discretizes the action space
                    driving_style=DrivingStyle.NORMAL,
                    is_remote_client=False,
                    max_steps=500,
                    max_episodes=100,
                    should_record=True,  # HDF5/numpy recordings
                    recording_dir=config.RECORDING_DIR,
                    randomize_view_mode=False,
                    view_mode_period=None,  # Domain randomization
                    randomize_sun_speed=False,
                    randomize_shadow_level=False,
                    randomize_month=False)

    forward = sim.action(throttle=1, steering=0, brake=0)
    done = False
    while True:
        while not done:
            observation, reward, done, info = env.step(forward)
        env.reset()
        print('Episode finished')
        done = False


if __name__ == '__main__':
    main()
