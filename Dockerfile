FROM python:3.9-slim-buster
ADD pyriemann_qiskit /pyriemann_qiskit
ADD examples /examples
ADD setup.py /
ADD README.md /

RUN apt-get update
RUN apt-get -y install git

RUN python setup.py develop
RUN pip install .[doc]

ENTRYPOINT [ "python", "/examples/ERP/classify_P300_bi.py" ]