.PHONY: package install commit clean bash version rerun server

# Short-cuts to build and run Deepdrive in Docker

# Usage:
#  make server  # runs sim
#  make run args="bin/domain_randomization_short_test.sh"  # run some agent
#
# Build and output artifacts:
# make artifacts
#
# Dev:
# make rerun args="python main.py --server"  # rebuilds container with local changes and runs
#
# To check the contents of last run container
# make commit args="mydebugcontainername"
# docker run -it mydebugcontainername bash

VERSION:=$(shell bin/get_version.sh)

# Non-lazy assignment to ensures the necessary directories are created,
#  i.e. ~/Deepdrive/*
CREATE_DIR:=$(shell DEEPDRIVE_DOCKER_HOST=1 python -c "import util.get_directories")

DEEPDRIVE_DIR:=$(shell docker/get_deepdrive_dir.sh)
DOCKER_DEEPDRIVE_DIR=/home/ue4/Deepdrive
TAG=deepdriveio/deepdrive:${VERSION}

# Volumes
DEEPDRIVE_VOL=-v $(DEEPDRIVE_DIR):/home/ue4/Deepdrive

LOG_DIR=$(DEEPDRIVE_DIR)/log
LOG_VOL=-v $(LOG_DIR):$(DOCKER_DEEPDRIVE_DIR)/log

RECORDINGS_DIR=$(DEEPDRIVE_DIR)/recordings
RECORDINGS_VOL=-v $(RECORDINGS_DIR):$(DOCKER_DEEPDRIVE_DIR)/recordings

RESULTS_DIR=$(DEEPDRIVE_DIR)/results
RESULTS_VOL=-v $(RESULTS_DIR):$(DOCKER_DEEPDRIVE_DIR)/results

TF_DIR=$(DEEPDRIVE_DIR)/tensorflow
TF_VOL=-v $(TF_DIR):$(DOCKER_DEEPDRIVE_DIR)/tensorflow

WEIGHTS_DIR=$(DEEPDRIVE_DIR)/weights
WEIGHTS_VOL=-v $(WEIGHTS_DIR):$(DOCKER_DEEPDRIVE_DIR)/weights

ARTIFACTS_DIRS=$(LOG_DIR) $(RECORDINGS_DIR) $(RESULTS_DIR) $(TF_DIR) $(WEIGHTS_DIR)
ARTIFACTS_VOLUMES:=$(LOG_VOL) $(RECORDINGS_VOL) $(RESULTS_VOL) $(TF_VOL) $(WEIGHTS_VOL)
MAKE_DIRS:=$(shell mkdir -p $(ARTIFACTS_DIRS))

DOCKER_OPTS=$(ARTIFACTS_VOLUMES) $(RUN_AS_ME) --net=host --runtime=nvidia
DD_RUN=docker run -it $(DOCKER_OPTS) deepdriveio/deepdrive:$(VERSION)

ARTIFACTS_FILE=$(RESULTS_DIR)/latest-artifacts.json
SERVER=$(DD_RUN) python main.py --server
PUBLIC=DEEPDRIVE_PUBLIC=true
EVAL=$(DEEPDRIVE_UPLOAD_ARTIFACTS=true)


# Pass args to make command, i.e.
#  make run args="echo yo did it!"
args=

install: build

dirs:
	echo $(ARTIFACTS_DIRS)

version:
	echo $(VERSION)

echo_dir:
	echo $(LOG_VOL)

rerun: build run

server:
	$(SERVER)

eval_server:
	$(EVAL) $(SERVER)

artifacts: build server
	find $(ARTIFACTS_FILE) 2> /dev/null

run:
	$(DD_RUN) $(args)

bash:
	$(DD_RUN) bash

commit:
	docker commit `docker ps --latest --format "{{.ID}}"` $(args)

build:
	docker build --build-arg version=$(VERSION) -t $(TAG) -f Dockerfile .


### Tests

public_domain_randomization:
	$(DD_RUN) bin/domain_randomization_short_test.sh