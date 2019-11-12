from typing import Tuple, List

from sim import DrivingStyle

import config as c


class SimArgs:
    experiment: str = None
    env_id: str = 'Deepdrive-v0'
    sess = None  # tensorflow Session
    start_dashboard: bool = True
    cameras: List[dict] = None
    use_sim_start_command: bool = False
    render: bool = False
    fps: int = c.DEFAULT_FPS
    combine_box_action_spaces: bool = False
    is_discrete: bool = False
    preprocess_with_tensorflow:bool = False
    is_sync: bool = False
    driving_style: str = DrivingStyle.NORMAL.as_string()
    is_remote_client: bool = False
    enable_traffic: bool = False
    ego_mph: float = None
    view_mode_period: int = None  # TODO: Change to view_mode_period_ms
    max_steps: int = None
    max_episodes: int = None
    should_record: bool = False
    recording_dir: str = c.RECORDING_DIR
    image_resize_dims: Tuple[int] = None
    should_normalize_image: bool = True
    reset_returns_zero: bool = True  # TODO: Change once dagger agents confirmed working with True
    eval_only: bool = False
    upload_gist: bool = False
    public: bool = False
    client_main_args: dict = None
    sim_step_time: float = c.DEFAULT_SIM_STEP_TIME
    randomize_view_mode: bool = False
    randomize_sun_speed: bool = False
    randomize_shadow_level: bool = False
    randomize_month: bool = False
    is_botleague: bool = False
    scenario_index: int = c.DEFAULT_SCENARIO_INDEX
    map: str = c.CANYONS_MAP_NAME
    path_follower: bool = False

    def __init__(self, **kwargs):
        for k in kwargs:
            if not hasattr(self, k):
                raise RuntimeError('Invalid sim start arg: %s' % k)
            setattr(self, k, kwargs[k])

    def get_vars(self):
        for (k, v) in vars(self).items():
            if not k.startswith('__'):
                yield k, v

    def to_dict(self):
        ret = {k: v for k,v in self.get_vars()}
        return ret


