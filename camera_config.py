import logs
import config

log = logs.get_log(__name__)

# Rigs are two dimensional arrays where...
# cameras in the first dimension are rotated through at the end of the episode during recording and...
# cameras in the second dimension create multiple simultaneously rendering views from the vehicle.
rigs = {
    'baseline_rigs': [
        [config.DEFAULT_CAM],
        [dict(name='forward cam 90 FOV', field_of_view=90, capture_width=340, capture_height=227,
              relative_position=[150, 1.0, 200],
              relative_rotation=[0.0, 0.0, 0.0])],
        [dict(name='semi-truck tall cam 110 FOV', field_of_view=110, capture_width=340, capture_height=227,
              relative_position=[150, 1.0, 400],
              relative_rotation=[0.0, -15.0, 0.0])],
    ],
    'three_cam_rig': [[
        dict(name='forward cam', field_of_view=60, capture_width=512, capture_height=289,
             relative_position=[150, 1.0, 200],
             relative_rotation=[0.0, 0.0, 0.0]),
        dict(name='left cam', field_of_view=60, capture_width=512, capture_height=289,
             relative_position=[150, -150., 200],
             relative_rotation=[0.0, 0.0, 0.0]),
        dict(name='right cam', field_of_view=60, capture_width=512, capture_height=289,
             relative_position=[150, 150., 200],
             relative_rotation=[0.0, 0.0, 0.0])
    ]]
}
