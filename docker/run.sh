#!/usr/bin/env bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VERSION=`${DIR}/../bin/get_version.sh`

docker run -it --net=host --runtime=nvidia deepdriveio/deepdrive:env-${VERSION} "$@"