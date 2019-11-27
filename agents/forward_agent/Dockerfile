# Build: docker build -t forward-agent .
# Run: docker run -it --net=host forward-agent
FROM python:3.7
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
COPY . /agent
WORKDIR /agent
CMD python agent.py
