
#!/usr/bin/env bash

readonly IMAGE_NAME_TAG=bryanlee99/deepdrive:left-agent

docker build -t ${IMAGE_NAME_TAG} .

docker push ${IMAGE_NAME_TAG}

