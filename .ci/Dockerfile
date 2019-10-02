FROM python:3.7
RUN apt-get update
RUN apt-get -y install apt-transport-https ca-certificates curl gnupg2 software-properties-common
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add -
RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
RUN apt-get update
RUN apt-get -y install docker-ce


RUN mkdir problem-worker
WORKDIR problem-worker
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Bootstrap source into build container
CMD curl -s -N https://raw.githubusercontent.com/deepdrive/deepdrive/${DEEPDRIVE_COMMIT}/.ci/build_botleague_containers.sh | bash
