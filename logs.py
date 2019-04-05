import os
import logging
from logging.handlers import RotatingFileHandler
import sys

import config as c
from util.anonymize import anonymize_user_home

os.makedirs(c.LOG_DIR, exist_ok=True)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_level = logging.INFO
all_loggers = []
rotators = {}


def get_log(namespace, filename='log.txt'):
    """
    Use separate filenames for separate processes to avoid rollover errors on Windows
    https://stackoverflow.com/questions/16712638
    """
    rotator = get_log_rotator(filename)
    ret = logging.getLogger(namespace)
    ret.addFilter(AnonymizeFilter())
    if ret.parent != ret.root:
        # Avoid duplicate log messages in multiprocessing scenarios
        # where module is imported twice
        print('Warning, using parent logger to avoid nested loggers and duplicate errors messages')
        return ret.parent
    ret.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_format)
    ret.addHandler(ch)
    ret.addHandler(rotator)
    all_loggers.append(ret)
    return ret


def get_log_rotator(filename):
    if filename in rotators:
        return rotators[filename]
    rotator = RotatingFileHandler(os.path.join(c.LOG_DIR, filename), maxBytes=(1048576 * 5), backupCount=7)
    rotator.setFormatter(log_format)
    rotators[filename] = rotator
    return rotator


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


class AnonymizeFilter(logging.Filter):
    def filter(self, record):
        anon = anonymize_user_home
        record.msg = anonymize_user_home(record.msg)
        record.args = tuple(anon(a) for a in record.args)
        return True
