from __future__ import print_function
import os
import sys
import zipfile
import argparse

try:
   input = raw_input
except NameError:
   pass

try:
    # Python 3
    from urllib.parse import urlparse, urlencode
    from urllib.request import urlopen, Request, urlretrieve
    from urllib.error import HTTPError
except ImportError:
    # Python 2
    from urlparse import urlparse
    from urllib import urlencode, urlretrieve
    from urllib2 import urlopen, Request, HTTPError


def download_file(url, path):
    import requests
    """Good for downloading large files from dropbox as it sets gzip headers and decodes automatically on download"""
    with open(path, "wb") as f:
        print('Downloading %s' % url)
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()


def download_folder(url, dirname, warn_existing=True):
    """Useful for downloading a folder / zip file from dropbox and unzipping it to path"""
    if os.path.exists(dirname) and warn_existing:
        print('%s exists, do you want to re-download and overwrite the existing files (y/n)?' % dirname, end=' ')
        overwrite = input()
        if 'n' in overwrite.lower():
            print('USING EXISTING %s - Try rerunning and overwriting if you run into problems.' % dirname)
            return
    print('Downloading %s to %s' % (url, dirname) + '...')
    location = urlretrieve(url)
    print('done.')
    location = location[0]
    zip_ref = zipfile.ZipFile(location, 'r')
    print('Unzipping temp file %s to %s' % (location, dirname) + '...')
    try:
        zip_ref.extractall(dirname)
        print('done.')
    except Exception:
        print('You may want to close all programs that may have these files open or delete existing '
              'folders this is trying to overwrite')
        raise
    finally:
        zip_ref.close()
        os.remove(location)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download files and folders')
    parser.add_argument('--zip-dir-url', required=True)
    parser.add_argument('--dest', required=True)
    args = parser.parse_args()
    download_folder(args.zip_dir_url, args.dest)
