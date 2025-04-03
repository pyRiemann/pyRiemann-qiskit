FROM python:3.11-slim-buster
ADD pyriemann_qiskit /pyriemann_qiskit
ADD examples /examples
ADD setup.py /
ADD README.md /

RUN apt-get update
RUN apt-get -y install git

RUN apt-get --allow-releaseinfo-change update
RUN python -m pip install --upgrade pip
RUN apt-get -y install --fix-missing git-core
RUN apt-get -y install build-essential

RUN pip install .
RUN pip install .[docs]
RUN pip install .[tests]

## Creating folders for mne data
RUN mkdir /root/mne_data
RUN mkdir /home/mne_data

ENTRYPOINT [ "python", "/examples/ERP/classify_P300_bi.py" ]
