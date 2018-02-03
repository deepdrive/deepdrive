import logs
import config

log = logs.get_log(__name__)

faq_cam = dict(name='faq_cam', field_of_view=60, capture_width=512, capture_height=512,
             relative_position=[150, 1., 200],
             relative_rotation=[0.0, 0.0, 0.0])

# First dimension is rotated through at the end of the episode. Second dimension is for separate cameras on the car.
rigs = {
    'baseline_rigs': [
        [faq_cam] * 8,
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
