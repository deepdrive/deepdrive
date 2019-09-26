#!/usr/bin/env bash

set -e  # Abort script at first error, when a command exits with non-zero status (except in until or while loops, if-tests, list constructs)
set -u  # Attempt to use undefined variable outputs error message, and forces an exit
set -x  # Similar to verbose mode (-v), but expands commands
set -o pipefail  # Causes a pipeline to return the exit status of the last command in the pipe that returned a non-zero return value.

git clone --depth=1 --branch ${DEEPDRIVE_BRANCH} https://github.com/deepdrive/deepdrive
cd deepdrive
git checkout -qf ${DEEPDRIVE_COMMIT}

# Build base container
make

python3 -u .ci/build_problem_containers.py
