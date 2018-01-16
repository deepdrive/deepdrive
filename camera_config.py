import logs
import config

log = logs.get_log(__name__)

# First dimension is rotated through at the end of the episode. Second dimension is for separate cameras on the car.
rigs = {
    'baseline_rigs': [
        [config.DEFAULT_CAM],
        [dict(name='forward_cam_wide_90', field_of_view=90, capture_width=340, capture_height=227,
              relative_position=[150, 1.0, 200],
              relative_rotation=[0.0, 0.0, 0.0])],
        [dict(name='semi_tall_cam_wide', field_of_view=110, capture_width=340, capture_height=227,
              relative_position=[150, 1.0, 400],
              relative_rotation=[0.0, -15.0, 0.0])],
    ],
    'three_cam_rig': [[
        dict(name='forward_cam', field_of_view=60, capture_width=512, capture_height=289,
             relative_position=[150, 1.0, 200],
             relative_rotation=[0.0, 0.0, 0.0]),
        dict(name='left_cam', field_of_view=60, capture_width=512, capture_height=289,
             relative_position=[150, -150., 200],
             relative_rotation=[0.0, 0.0, 0.0]),
        dict(name='right_cam', field_of_view=60, capture_width=512, capture_height=289,
             relative_position=[150, 150., 200],
             relative_rotation=[0.0, 0.0, 0.0])
    ]]
}
