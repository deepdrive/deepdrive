from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from future.builtins import (int, open, round,
                             str)

import deepdrive_simulation

import config as c
import logs
log = logs.get_log(__name__)


SHADOW_RANGE = list(range(0, 5))

def set_capture_graphics(shadow_level):
    """Set shadow quality to level where level is an int between 0 and 4
        Note: We can set texture level, ambient occlusion, etc... but it only affects
        the display window, not the captured cameras.
    """
    assert np.issubdtype(shadow_level, int)
    shadow_level = int(shadow_level)
    assert shadow_level in SHADOW_RANGE
    settings = deepdrive_simulation.SimulationGraphicsSettings()
    settings.shadow_quality = shadow_level
    deepdrive_simulation.set_graphics_settings(settings)


def randomize_shadow_level():
    set_capture_graphics(shadow_level=c.rng.choice(SHADOW_RANGE))
