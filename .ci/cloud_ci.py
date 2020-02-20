import json
import os
import sys
from typing import List, Tuple

import time
from datetime import datetime
from os.path import dirname, realpath, join

from botleague_helpers.ci import build_and_run_botleague_ci, run_botleague_ci, \
    dbox
from box import Box, BoxList
from loguru import logger as log

from problem_constants.constants import SUPPORTED_PROBLEMS

import get_tag_build_id

# from logs import log

"""
Here we run the build and integration tests on our own infrastructure so that
we can have full control over how the build is run, but still keep track builds
in a standard way with a nice, free, hosted, UI in Circle / Travis.
"""

DIR = dirname(realpath(__file__))

@log.catch(reraise=True)
def main():
    commit = os.environ['CIRCLE_SHA1']
    branch = os.environ['CIRCLE_BRANCH']
    # Circle builds / pushes the candidate deepdrive and botleague containers
    run_botleague_ci_for_deepdrive_build(branch, commit)

def run_botleague_ci_for_deepdrive_build(branch, commit):

    def set_version(problem_def, version):
        # Deepdrive sim sets the problem version, which is appropriate since
        # it determines most of the differences between versions.
        date_str = datetime.utcnow().strftime('%Y-%m-%d_%I-%M-%S%p')
        problem_def.rerun = f'deepdrive-build-{date_str}'

    pr_message = 'Auto generated commit for deepdrive-build CI'
    container_postfix = get_tag_build_id.main()
    passed_ci = run_botleague_ci(
        branch=branch,
        version=commit,
        pr_message=pr_message,
        set_version_fn=set_version,
        supported_problems=SUPPORTED_PROBLEMS,
        container_postfix=container_postfix)
    if not passed_ci:
        raise RuntimeError('Failed Botleague CI')

if __name__ == '__main__':
    main()
