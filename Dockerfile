# Definition of Submission container
ARG DOCKER_REGISTRY=docker.io
ARG ARCH=amd64
ARG MAJOR=daffy
ARG BASE_TAG=${MAJOR}-${ARCH}

FROM ${DOCKER_REGISTRY}/duckietown/dt-machine-learning-base-environment:${BASE_TAG}

ARG PIP_INDEX_URL="https://pypi.org/simple"
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# Setup any additional pip packages
COPY requirements.pin.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.pin.txt

COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN python3 -m pip install --no-cache-dir -r .requirements.txt

# let's copy all our solution files to our workspace
WORKDIR /submission
COPY solution.py ./
COPY tf_models /submission/tf_models
COPY model.py ./


CMD ["python3", "solution.py"]
