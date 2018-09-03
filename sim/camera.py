from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (int, open, round,
                             str)

class Camera(object):
    def __init__(self, name, field_of_view, capture_width, capture_height, relative_position, relative_rotation):
        self.name = name
        self.field_of_view = field_of_view
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.relative_position = relative_position
        self.relative_rotation = relative_rotation
        self.connection_id = None
