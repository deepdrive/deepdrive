import os
import pkgutil
import sys

import absl
from absl.flags import FlagValues

import tensorflow as tf
from six.moves import reload_module


def commandeer_tf_flags(create_flags, kwargs):
    pkgpath = os.path.dirname(absl.flags.__file__)
    for _, name, _ in pkgutil.walk_packages([pkgpath]):
        reload_module(sys.modules['absl.flags.' + name])

    # Hack to reset global flags
    reload_module(absl.flags._flagvalues)
    reload_module(absl.flags._defines)
    reload_module(absl.flags)
    reload_module(tf.flags)

    tf.app.flags.DEFINE_integer

    flags = create_flags()
    flags.__dict__['__wrapped']([sys.argv[0]])  # Hack to avoid parsing sys.argv in flags
    for k in kwargs:
        setattr(flags, k, kwargs[k])
