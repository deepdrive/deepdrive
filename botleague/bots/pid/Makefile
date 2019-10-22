.PHONY: package install commit clean bash version

TAG=deepdriveio/deepdrive:bot_pid${TAG_BUILD_ID}
DOCKER_OPTS=--net=host --runtime=nvidia
RUN=docker run -it $(DOCKER_OPTS) $(TAG)

build:
	docker build -t $(TAG) .

run:
	$(RUN)

bash:
	$(RUN) bash

push:
	docker push $(TAG)
