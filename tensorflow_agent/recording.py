import logs

log = logs.get_log(__name__)

cameras = [
    dict(field_of_view=60, capture_width=227, capture_height=227, relative_position=[0.0, 0.0, 0.0],
         relative_rotation=[0.0, 0.0, 0.0]),
    dict(field_of_view=90, capture_width=512, capture_height=256, relative_position=[1.0, 1.0, 1.0],
         relative_rotation=[0.0, 0.0, 0.0]),
]
