from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import os
import tempfile
import zipfile

import requests
from clint.textui import progress

from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import config as c
import logs

log = logs.get_log(__name__)


def download(url, directory, warn_existing=True, overwrite=False):
    """
    Download and unzip
    """
    if has_stuff(directory, warn_existing, overwrite):
        return
    else:
        os.makedirs(directory, exist_ok=True)

    log.info('Downloading %s to %s...', url, directory)

    request = requests.get(url, stream=True)
    filename = url.split('/')[-1]
    if '?' in filename:
        filename = filename[:filename.index('?')]
    location = os.path.join(tempfile.gettempdir(), filename)
    with open(location, 'wb') as f:
        if request.status_code == 404:
            raise RuntimeError('Download URL not accessible %s' % url)
        total_length = int(request.headers.get('content-length'))
        for chunk in progress.bar(request.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
            if chunk:
                f.write(chunk)
                f.flush()

    log.info('done.')
    zip_ref = zipfile.ZipFile(location, 'r')
    log.info('Unzipping temp file %s to %s...', location, directory)
    try:
        zip_ref.extractall(directory)
    except Exception:
        print('You may want to close all programs that may have these files '
              'open or delete existing '
              'folders this is trying to overwrite')
        raise
    finally:
        zip_ref.close()
        os.remove(location)
        log.info('Removed temp file %s', location)


def has_stuff(path, warn_existing=False, overwrite=False):
    # TODO: Remove overwrite as a parameter, doesn't make sense here.
    if os.path.exists(path) and (dir_has_stuff(path) or file_has_stuff(path)):
        if warn_existing:
            print('%s exists, do you want to re-download and overwrite any existing files (y/n)?' % path, end=' ')
            overwrite = input()
            if 'n' in overwrite.lower():
                print('USING EXISTING %s - Try rerunning and overwriting if you run into problems.' % path)
                return True
        elif not overwrite:
            return True
    return False


def dir_has_stuff(path):
    return os.path.isdir(path) and os.listdir(path)


def file_has_stuff(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0