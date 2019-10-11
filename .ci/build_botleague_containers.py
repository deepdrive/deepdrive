import os
import sys
from glob import glob
from os.path import join

from utils import get_tag_build_id

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(DIR)

def main():
    bot_dirs = glob(f'{join(ROOT, "botleague")}/bots/*')
    problem_dirs = glob(f'{join(ROOT, "botleague")}/problems/*')

    os.environ['TAG_BUILD_ID'] = get_tag_build_id()

    # Get names of docker files, build them
    for pdir in problem_dirs + bot_dirs:
        exit_code = os.system(f'cd {pdir} && make && make push')
        if exit_code != 0:
            raise RuntimeError('Error building problem container, check above')


if __name__ == '__main__':
    main()
