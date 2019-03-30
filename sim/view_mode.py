from __future__ import (absolute_import, division,
                        print_function, unicode_literals)


from future.builtins import (int, open, round,
                             str)

from enum import Enum

import deepdrive_client
import config as c


class ViewMode(Enum):
    NORMAL = ''
    WORLD_NORMAL = 'WorldNormal'
    BASE_COLOR = 'BaseColor'
    ROUGHNESS = 'Roughness'
    REFLECTIVITY = 'Reflectivity'
    AMBIENT_OCCLUSION = 'AmbientOcclusion'
    DEPTH_HEAT_MAP = 'HeatMap'
    SPECULARITY = 'Specularity'


class ViewModeController(object):
    def __init__(self, period=None, client_id=None):
        self.client_id = None
        self.period = period
        self.client_id = client_id
        self.steps_since_switch = None
        self.modes = list(ViewMode.__members__.values())
        self.num_modes = len(self.modes)
        self.current = ViewMode.NORMAL
        self.view_index = None
        self.reset()

    def step(self, client_id):
        self.client_id = client_id
        if self.should_switch():
            self.view_index += 1
            if self.view_index >= self.num_modes:
                self.view_index = 0
            self.set_view_mode(self.modes[self.view_index])
            self.steps_since_switch = 0
        else:
            self.steps_since_switch += 1

    def should_switch(self):
        if self.period is not None:
            return self.steps_since_switch == (self.period - 1)
        else:
            return False

    def reset(self):
        self.steps_since_switch = 0
        self.view_index = 0

    def current_mode_name(self):
        return self.current.name.lower()

    def set_view_mode(self, view_mode, cam_id=-1):
        # Passing a cam_id of -1 sets all cameras with the same view mode
        if self.client_id is None:
            raise RuntimeError('Client id not set. HINT: Call env.step() '
                               'at least once before setting view mode')
        deepdrive_client.set_view_mode(self.client_id, cam_id, view_mode.value)
        self.current = view_mode

    def set_random(self):
        self.set_view_mode(c.rng.choice(list(ViewMode.__members__.values())))

