TAG=deepdriveio/deepdrive:problem_build
DOCKER_OPTS=--net=host --runtime=nvidia
RUN=docker run -it -v /var/run/docker.sock:/var/run/docker.sock $(DOCKER_OPTS) $(TAG)

build:
	docker build -t $(TAG) .

run:
	$(RUN)

bash:
	$(RUN) bash

push:
	docker push $(TAG)
