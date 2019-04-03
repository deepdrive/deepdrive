.PHONY: package install commit clean bash version

VERSION=`cat VERSION | sed 's/ //g'`
DEEPDRIVE_DIR=`cat ~/.deepdrive/deepdrive_dir | sed 's/ //g'`
VOLUMES=-v $(DEEPDRIVE_DIR):/home/ue4/Deepdrive
DOCKER_OPTS=$(VOLUMES) --net=host --runtime=nvidia
#DOCKER_OPTS=--net=host --runtime=nvidia
DD_RUN=docker run -it $(DOCKER_OPTS) deepdriveio/deepdrive:env-3.0
TAG=deepdriveio/deepdrive:env-${VERSION}

install: build

version:
	echo $(VERSION)

build_and_run: build run

run:
	$(DD_RUN)

bash:
	$(DD_RUN) bash

commit:
	docker commit `docker ps --latest --format "{{.ID}}"` $(TAG)

#bash:
#	docker run -it -v `pwd`/../..:/home/ue4/deepdrive deepdriveio/deepdrive:env-$(VERSION) bash

build:
	docker build --build-arg version=$(VERSION) -t $(TAG) -f Dockerfile-env .