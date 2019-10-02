#!/usr/bin/env bash

set -e  # Abort script at first error, when a command exits with non-zero status (except in until or while loops, if-tests, list constructs)
set -u  # Attempt to use undefined variable outputs error message, and forces an exit
set -x  # Similar to verbose mode (-v), but expands commands
set -o pipefail  # Causes a pipeline to return the exit status of the last command in the pipe that returned a non-zero return value.

export DEEPDRIVE_BRANCH=`git rev-parse --abbrev-ref HEAD`
export DEEPDRIVE_COMMIT=`git rev-parse HEAD`

DIR=`dirname "$0"`
cd ${DIR}/..

./.ci/build_botleague_containers.sh
