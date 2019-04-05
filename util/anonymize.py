from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)


import os


def anonymize_user_home(data):
    if not isinstance(data, str):
        return data
    else:
        return data.replace(os.path.expanduser("~"), '~')
