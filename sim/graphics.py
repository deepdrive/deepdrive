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
    if not isinstance(shadow_level, int) and np.issubdtype(shadow_level, int):
        raise ValueError('Shadow level should be an integer')
    shadow_level = int(shadow_level)
    if shadow_level not in SHADOW_RANGE:
        raise ValueError('Shadow level should be between 0 and 4')
    settings = deepdrive_simulation.SimulationGraphicsSettings()
    settings.shadow_quality = shadow_level
    deepdrive_simulation.set_graphics_settings(settings)


def randomize_shadow_level():
    level = c.rng.choice(SHADOW_RANGE)
    log.info('Setting new shadow quality level (%r/%r)', level, max(SHADOW_RANGE))
    set_capture_graphics(shadow_level=level)
