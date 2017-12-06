#!/usr/bin/env bash

set -euo pipefail

# Check for python 3.5+
py="$(python -u install/check_py_version.py)"
py_save=`which ${py}`

# Install python dependencies
sudo apt-get install -y python3-tk  || :  # For matplotlib dashboard (optional)
if ! [ -x "$(command -v pipenv)" ]; then
    sudo ${py_save} -m pip install pipenv
fi
pipenv install
set +e
pipenv run python -u install/check_tf_version.py
tf_valid=$?
set -e

echo "Running sanity tests"
DEEPDRIVE_DIR=/tmp/testdeepdrive pipenv run pytest tests/test_sanity.py
echo "Tests successful"

if [ "$tf_valid" -eq "0" ]; then
    echo "Starting baseline agent"
    bin/run_baseline_agent.sh
else
    echo "Starting sim in manual mode"
    bin/drive_manually.sh
fi

# Run the tutorial.sh
# Mute with u
# Escape to see menu / controls
# Change cam with 1, 2, 3
# Alt-tab to get back to agent.py
# Change the throttle out in agent.py

# Pause the game, ask them to press j
# Pause the game, ask them to change the camera position
# Pause the game, ask them to change the steering coeff

# To rerun this tutorial, run tutorial.sh
