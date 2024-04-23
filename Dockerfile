FROM python:3.9-slim-buster
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

RUN pip install urllib3==1.26.12
RUN pip install "numpy<1.24"
RUN pip install qiskit-terra==0.45.0
## Workaround for firestore
RUN pip install protobuf==4.25.3
## End workaround
RUN python setup.py develop
RUN pip install .[docs]
RUN pip install .[tests]

## Creating folders for mne data
RUN mkdir /root/mne_data
RUN mkdir /home/mne_data

## Workaround for firestore
# Install egg
RUN pip install protobuf==4.25.3
RUN pip install google_cloud_firestore==2.16.0
### Missing __init__ file in protobuf
RUN touch /usr/local/lib/python3.9/site-packages/protobuf-4.25.3-py3.9.egg/google/__init__.py
## google.cloud.location is never used in these files, and is missing in path.
RUN sed -i 's/from google.cloud.location import locations_pb2//g' '/usr/local/lib/python3.9/site-packages/google_cloud_firestore-2.16.0-py3.9.egg/google/cloud/firestore_v1/services/firestore/client.py'
RUN sed -i 's/from google.cloud.location import locations_pb2//g' '/usr/local/lib/python3.9/site-packages/google_cloud_firestore-2.16.0-py3.9.egg/google/cloud/firestore_v1/services/firestore/transports/base.py'
RUN sed -i 's/from google.cloud.location import locations_pb2//g' '/usr/local/lib/python3.9/site-packages/google_cloud_firestore-2.16.0-py3.9.egg/google/cloud/firestore_v1/services/firestore/transports/grpc.py'
RUN sed -i 's/from google.cloud.location import locations_pb2//g' '/usr/local/lib/python3.9/site-packages/google_cloud_firestore-2.16.0-py3.9.egg/google/cloud/firestore_v1/services/firestore/transports/grpc_asyncio.py'
RUN sed -i 's/from google.cloud.location import locations_pb2//g' '/usr/local/lib/python3.9/site-packages/google_cloud_firestore-2.16.0-py3.9.egg/google/cloud/firestore_v1/services/firestore/transports/rest.py'
RUN sed -i 's/from google.cloud.location import locations_pb2//g' '/usr/local/lib/python3.9/site-packages/google_cloud_firestore-2.16.0-py3.9.egg/google/cloud/firestore_v1/services/firestore/async_client.py'

ENTRYPOINT [ "python", "/examples/ERP/classify_P300_bi.py" ]
