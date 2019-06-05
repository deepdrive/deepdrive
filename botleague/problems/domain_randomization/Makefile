.PHONY: package install commit clean bash version

TAG=deepdriveio/deepdrive:domain_rand_problem
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