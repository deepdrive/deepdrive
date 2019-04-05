.PHONY: package install commit clean bash version

VERSION:=$(shell bin/get_version.sh)
DEEPDRIVE_DIR:=$(shell cat ~/.deepdrive/deepdrive_dir | sed 's/ //g')
DOCKER_DEEPDRIVE_DIR=/home/ue4/Deepdrive
TAG=deepdriveio/deepdrive:env-${VERSION}

# Volumes
DEEPDRIVE_VOL=-v $(DEEPDRIVE_DIR):/home/ue4/Deepdrive
LOG_VOL=-v $(DEEPDRIVE_DIR)/log:$(DOCKER_DEEPDRIVE_DIR)/log
RECORDINGS_VOL=-v $(DEEPDRIVE_DIR)/recordings:$(DOCKER_DEEPDRIVE_DIR)/recordings
RESULTS_VOL=-v $(DEEPDRIVE_DIR)/results:$(DOCKER_DEEPDRIVE_DIR)/results
TF_VOL=-v $(DEEPDRIVE_DIR)/tensorflow:$(DOCKER_DEEPDRIVE_DIR)/tensorflow
WEIGHTS_VOL=-v $(DEEPDRIVE_DIR)/weights:$(DOCKER_DEEPDRIVE_DIR)/weights
ARTIFACTS_VOLUMES=$(LOG_VOL) $(RECORDINGS_VOL) $(RESULTS_VOL) $(TF_VOL) $(WEIGHTS_VOL)

DOCKER_OPTS=$(ARTIFACTS_VOLUMES) --net=host --runtime=nvidia
DD_RUN=docker run -it $(DOCKER_OPTS) deepdriveio/deepdrive:env-3.0

# Pass args to make command, i.e.
#  make run args="echo yo did it!"
args=

install: build

version:
	echo $(VERSION)

print_version: version
	echo ${VERSION}

echo_dir:
	echo $(LOG_VOL)

rerun: build run

server: run

run:
	$(DD_RUN) $(args)

bash:
	$(DD_RUN) bash

commit:
	docker commit `docker ps --latest --format "{{.ID}}"` $(TAG)

build:
	docker build --build-arg version=$(VERSION) -t $(TAG) -f Dockerfile-env .