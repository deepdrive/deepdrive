#!/usr/bin/env bash

set -e


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cat ${DIR}/../VERSION | sed 's/ //g'