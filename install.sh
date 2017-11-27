#!/usr/bin/env bash

set -euo pipefail

echo 'Please elevate to sudo...'  # Necessary for saving to /opt/
sudo bash -c ":"

# Check for python 3.5+

py="$(python -u scripts/check_py_version.py)"

# Download sim

py_save=`which ${py}`
echo "Downloading simulator - it's about ~1GB, so may take some time."
sudo bash -c "${py_save} -u scripts/download.py --zip-dir-url https://d1y4edi1yk5yok.cloudfront.net/sim/deepdrive-2.0.201801010909.zip --dest /opt/deepdrive"
#sudo bash -c "${py_save} -u scripts/download.py --zip-dir-url https://s3-us-west-1.amazonaws.com/deepdrive/sim/test-download.zip --dest /opt/dltest"

# Install python dependencies

if ! [ -x "$(command -v pipenv)" ]; then
    echo 'Error: pipenv is not installed. - HINT: Install with `sudo pip install pipenv` or `sudo pip3 install pipenv`' >&2
    exit 1
fi
pipenv install

if [ -z ${PIPENV_ACTIVE+x} ]; then
   pipenv shell
fi

bash -c "${py} -u scripts/check_tf_version.py"

pytest

# Run the tutorial.sh


# If tensorflow installed, run the imitation learning agent, else run forward agent
# Download the weights to /var/deepdrive

# Pause the game, ask them to press j
# Pause the game, ask them to change the camera position
# Pause the game, ask them to change the steering coeff

# To rerun this tutorial, run tutorial.sh