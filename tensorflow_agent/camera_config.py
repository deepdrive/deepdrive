import logs
import config

log = logs.get_log(__name__)

rigs = [
    [config.DEFAULT_CAM],
    [dict(name='forward_cam_wide_90', field_of_view=90, capture_width=340, capture_height=227,
         relative_position=[150, 1.0, 200],
         relative_rotation=[0.0, 0.0, 0.0])],
    [dict(name='semi_tall_cam_wide', field_of_view=110, capture_width=340, capture_height=227,
          relative_position=[150, 1.0, 400],
          relative_rotation=[0.0, -15.0, 0.0])],
]
