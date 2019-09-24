import os
import sys
from glob import glob
from os.path import join

import docker

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(DIR)

def main():

    if not '--problems-only' in sys.argv:
        bot_dirs = glob(f'{join(ROOT, "botleague")}/bots/*')

    build_problem_containers()


def build_problem_containers():
    problem_dirs = glob(f'{join(ROOT, "botleague")}/problems/*')

    for pdir in problem_dirs:
        os.system(f'cd {pdir} && make && make push')

    # Get names of docker files, build them


if __name__ == '__main__':
    main()
