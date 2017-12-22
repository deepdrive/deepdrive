import logs

log = logs.get_log(__name__)

cameras = [
    dict(name='alexnet_forward_cam_60', field_of_view=60, capture_width=227, capture_height=227,
         relative_position=[150, 1.0, 200],
         relative_rotation=[0.0, 0.0, 0.0]),
    dict(name='forward_cam_90', field_of_view=90, capture_width=340, capture_height=227,
         relative_position=[150, 1.0, 200],
         relative_rotation=[0.0, 0.0, 0.0]),
]
