TAG=deepdriveio/deepdrive:ci

TAG_BUILD_ID:=$(shell python get_tag_build_id.py)
DEEPDRIVE_BRANCH:=$(shell git rev-parse --abbrev-ref HEAD)
DEEPDRIVE_COMMIT:=$(shell git rev-parse HEAD)
DOCKER_OPTS=-it -v /var/run/docker.sock:/var/run/docker.sock --net=host -e DEEPDRIVE_BRANCH=$(DEEPDRIVE_BRANCH) -e DEEPDRIVE_COMMIT=$(DEEPDRIVE_COMMIT) -e DOCKER_USER=${DOCKER_USER} -e DOCKER_PASS=${DOCKER_PASS}
BUILD_BOTLEAGUE=docker run $(DOCKER_OPTS) -e TAG_BUILD_ID=$(TAG_BUILD_ID) $(TAG)
PUSH_BOTLEAGUE=docker run $(DOCKER_OPTS) $(TAG)

build:
	docker build -t $(TAG) .

echo:
	echo $(TAG_BUILD_ID)

build_botleague:
	$(BUILD_BOTLEAGUE)

push_botleague:
	$(PUSH_BOTLEAGUE)

bash:
	$(BUILD_BOTLEAGUE) bash

push:
	docker push $(TAG)
