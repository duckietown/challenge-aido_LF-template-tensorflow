# Definition of Submission container
FROM frank1chude1qian/dt-machine-learning-base-environment:daffy-amd64

ARG PIP_INDEX_URL="https://pypi.org/simple"
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# Setup any additional pip packages
RUN pip3 install -U "pip>=20.2"
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN  pip3 install --use-feature=2020-resolver -r .requirements.txt

# let's copy all our solution files to our workspace
WORKDIR /submission
COPY solution.py ./
COPY tf_models /submission/tf_models
COPY model.py ./


ENTRYPOINT ["python3", "solution.py"]
