import os
import sys
from glob import glob
from os.path import join

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(DIR)


def main():
    bot_dirs = glob(f'{join(ROOT, "botleague")}/bots/*')
    problem_dirs = glob(f'{join(ROOT, "botleague")}/problems/*')

    # Get names of docker files, build them
    for pdir in problem_dirs + bot_dirs:
        exit_code = os.system(f'cd {pdir} && make && make push')
        if exit_code != 0:
            raise RuntimeError(f'Error building {pdir} container, check above')


if __name__ == '__main__':
    main()
