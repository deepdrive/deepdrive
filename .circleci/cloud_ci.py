import json
import os
import sys
from typing import List, Tuple

import time
from datetime import datetime
from os.path import dirname, realpath, join

from botleague_helpers.ci import build_and_run_botleague_ci, run_botleague_ci
from box import Box, BoxList

import problem_constants.constants

from loguru import logger as log
# from logs import log

"""
Here we run the build and integration tests on our own infrastructure so that
we can have full control over how the build is run, but still keep track builds
in a standard way with a nice, free, hosted, UI.
"""

DIR = dirname(realpath(__file__))

@log.catch(reraise=True)
def main():
    build_and_run_botleague_ci(
        build_url='https://sim.deepdrive.io/build-deepdrive',
        run_botleague_ci_wrapper_fn=run_botleague_ci_for_deepdrive_build)


def run_botleague_ci_for_deepdrive_build(branch, commit, job):
    build_results = Box.from_json(job.results.json_results_from_logs)

    def set_version(problem_def, version):
        # Deepdrive sim sets the problem version, which is appropriate since
        # it determines most of the differences between versions.
        date_str = datetime.utcnow().strftime('%Y-%m-%d_%I-%M-%S%p')
        problem_def.rerun = f'deepdrive-build-{date_str}'

    pr_message = 'Auto generated commit for deepdrive-build CI'
    passed_ci = run_botleague_ci(
        branch=branch,
        version=commit,
        pr_message=pr_message,
        set_version_fn=set_version,
        supported_problems=problem_constants.constants.SUPPORTED_PROBLEMS)

    if passed_ci:
        # TODO: Promote botleague containers to defaults
        pass

if __name__ == '__main__':
    main()
