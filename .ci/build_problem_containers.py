import os
from glob import glob
from os.path import join

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(DIR)

def main():
    bot_dirs = glob(f'{join(ROOT, "botleague")}/bots/*')
    problem_dirs = glob(f'{join(ROOT, "botleague")}/problems/*')

    # Build base image



    result = os.system('docker run -v /var/run/docker.sock:/var/run/docker.sock')

    # Get names of docker files, build them

if __name__ == '__main__':
    main()
