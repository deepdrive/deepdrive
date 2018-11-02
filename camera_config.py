from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

from copy import deepcopy

import logs
import config as c
from sim.view_mode import ViewMode

log = logs.get_log(__name__)

# Rigs are two dimensional arrays where...
# cameras in the first dimension are rotated through at the end of the episode during recording and...
# cameras in the second dimension create multiple simultaneously rendering views from the vehicle.
rigs = {
    'default_rig': [[c.DEFAULT_CAM]],
    'baseline_rigs': [
        [c.DEFAULT_CAM],
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

DEFAULT_BASE_COLOR_CAM = deepcopy(c.DEFAULT_CAM)
DEFAULT_BASE_COLOR_CAM['view_mode'] = ViewMode.BASE_COLOR.value
rigs['default_base_color_rig'] = [[DEFAULT_BASE_COLOR_CAM]]
