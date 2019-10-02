TAG=deepdriveio/deepdrive:problem_build

VERSION:=$(shell bin/get_version.sh)
DEEPDRIVE_BRANCH:=$(shell git rev-parse --abbrev-ref HEAD)
DEEPDRIVE_COMMIT:=$(shell git rev-parse HEAD)

DOCKER_OPTS=--net=host --runtime=nvidia -e DEEPDRIVE_BRANCH=$(DEEPDRIVE_BRANCH) -e DEEPDRIVE_COMMIT=$(DEEPDRIVE_COMMIT)

build:
	docker build -t $(TAG) .

run:
	docker run -it -v /var/run/docker.sock:/var/run/docker.sock $(DOCKER_OPTS) $(TAG)

bash:
	$(RUN) bash

push:
	docker push $(TAG)
