import os
import logging
from logging.handlers import RotatingFileHandler
import sys

import config as c


os.makedirs(c.LOG_DIR, exist_ok=True)
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_rotator = RotatingFileHandler(os.path.join(c.LOG_DIR, 'log.txt'), maxBytes=(1048576 * 5), backupCount=7)
log_rotator.setFormatter(log_format)
log_level = logging.INFO
all_loggers = []

def get_log(namespace, rotator=log_rotator):
    ret = logging.getLogger(namespace)
    ret.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_format)
    ret.addHandler(ch)
    ret.addHandler(rotator)
    all_loggers.append(ret)
    return ret


def set_level(level):
    global log_level
    log_level = level
    for l in all_loggers:
        l.setLevel(level)

def log_manual():
    test_log_rotator = RotatingFileHandler(os.path.join(c.LOG_DIR, 'test.txt'), maxBytes=3, backupCount=7)
    log1 = get_log('log1', rotator=test_log_rotator)
    log2 = get_log('log2', rotator=test_log_rotator)
    log1.info('asdf')
    log2.info('zxcv')