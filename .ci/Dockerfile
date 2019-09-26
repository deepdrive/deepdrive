FROM python:3.7
RUN mkdir
WORKDIR problem-worker
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Bootstrap source into build container
CMD curl -s -N https://raw.githubusercontent.com/deepdrive/deepdrive/${DEEPDRIVE_COMMIT}/.ci/build_botleague_containers.sh | bash
