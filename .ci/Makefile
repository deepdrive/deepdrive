TAG=deepdriveio/deepdrive:ci

DEEPDRIVE_BRANCH:=$(shell git rev-parse --abbrev-ref HEAD)
DEEPDRIVE_COMMIT:=$(shell git rev-parse HEAD)
RUN=docker run -it -v /var/run/docker.sock:/var/run/docker.sock $(DOCKER_OPTS) $(TAG)

DOCKER_OPTS=--net=host -e DEEPDRIVE_BRANCH=$(DEEPDRIVE_BRANCH) -e DEEPDRIVE_COMMIT=$(DEEPDRIVE_COMMIT) -e DOCKER_USERNAME=${DOCKER_USERNAME} -e DOCKER_PASSWORD=${DOCKER_PASSWORD} -e BUILD_ID=0

build:
	docker build -t $(TAG) .

run:
	$(RUN)

bash:
	$(RUN) bash

push:
	docker push $(TAG)
