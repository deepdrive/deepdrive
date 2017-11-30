#!/usr/bin/env bash

set -euo pipefail

# For saving to /opt/
echo 'Please elevate to sudo to install to /opt...'
sudo bash -c ":"

# Check for python 3.5+
py="$(python -u scripts/check_py_version.py)"

# Download sim
py_save=`which ${py}`
echo "Downloading ~1GB simulator"
sim_url="https://d1y4edi1yk5yok.cloudfront.net/sim/deepdrive-sim-2.0.20171127052011.zip"
sudo bash -c "${py_save} -u scripts/download.py --zip-dir-url $sim_url --dest /opt/deepdrive"

# Install python dependencies
sudo apt-get install -y python3-tk  || :  # For matplotlib dashboard (optional)
if ! [ -x "$(command -v pipenv)" ]; then
    sudo pip3 install pipenv || sudo pip install pipenv
fi
pipenv install
if [ -z ${PIPENV_ACTIVE+x} ]; then
   pipenv shell
fi
bash -c "${py} -u scripts/check_tf_version.py"

# Run quick sanity tests
pytest tests/test_sanity.py

# Download weights if tensorflow to /var/deepdrive
echo "Downloading ~700MB in weights "
weights_url="https://d1y4edi1yk5yok.cloudfront.net/weights/baseline_agent_weights.zip"
sudo bash -c "${py_save} -u scripts/download.py --zip-dir-url $weights_url --dest /var/deepdrive"

# Start the agent in a new terminal
echo "Starting simulator in new window, this will take a few seconds the first time around."
x-terminal-emulator -e scripts/new_terminal_helper.sh "$py_save" main.py --benchmark -n /var/deepdrive/baseline_agent_weights/model.ckpt-122427

# Run the tutorial.sh
# Mute with u
# Escape to see menu / controls
# Change cam with 1, 2, 3
# Alt-tab to get back to agent.py
# Change the throttle out in agent.py


# If tensorflow installed, run the imitation learning agent, else run forward agent


# Pause the game, ask them to press j
# Pause the game, ask them to change the camera position
# Pause the game, ask them to change the steering coeff

# To rerun this tutorial, run tutorial.sh