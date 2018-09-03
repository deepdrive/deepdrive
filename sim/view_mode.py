from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (int, open, round,
                             str)

from enum import Enum


class ViewMode(Enum):
    NORMAL = ''
    WORLD_NORMAL = 'WorldNormal'
    BASE_COLOR = 'BaseColor'
    ROUGHNESS = 'Roughness'
    REFLECTIVITY = 'Reflectivity'
    AMBIENT_OCCLUSION = 'AmbientOcclusion'
    DEPTH_HEAT_MAP = 'HeatMap'