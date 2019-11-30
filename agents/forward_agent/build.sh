
#!/usr/bin/env bash

readonly IMAGE_NAME_TAG=lazydriver/deepdrive:left-turn-agent

docker build -t ${IMAGE_NAME_TAG} .

docker push ${IMAGE_NAME_TAG}

