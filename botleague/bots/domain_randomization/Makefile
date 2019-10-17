.PHONY: package install commit clean bash version

TAG=deepdriveio/deepdrive:bot_domain_randomization${TAG_BUILD_ID}
DOCKER_OPTS=--net=host --runtime=nvidia -e WEIGHTS_DIR=/home/${DEEPDRIVE_USER}/deepdrive_weights -e WEIGHTS_NAME="domain_randomization_2019-03-04__02-11-53PM"
RUN=docker run -it $(DOCKER_OPTS) $(TAG)

build:
	docker build -t $(TAG) .

run:
	$(RUN)

bash:
	$(RUN) bash

push:
	docker push $(TAG)
