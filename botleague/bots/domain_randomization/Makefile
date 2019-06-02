.PHONY: package install commit clean bash version

TAG=deepdriveio/deepdrive:domain_rand_bot
RUN=docker run -it $(TAG)

build:
	docker build -t $(TAG) .

run:
	$(RUN)

bash:
	$(RUN) bash

push:
	docker push $(TAG)