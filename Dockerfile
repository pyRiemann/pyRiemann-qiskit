RUN pip install urllib3==1.26.12
RUN python setup.py develop
RUN pip install .[docs]
RUN pip install .[tests]

## Creating folders for mne data
RUN mkdir /root/mne_data
RUN mkdir /home/mne_data

## Workaround for firestore
### Missing __init__ file in protobuf
RUN touch /usr/local/lib/python3.8/site-packages/protobuf-4.22.1-py3.8-linux-x86_64.egg/google/__init__.py
## google.cloud.location is never used in these files, and is missing in path.
RUN sed -i 's/from google.cloud.location import locations_pb2//g' '/usr/local/lib/python3.8/site-packages/google_cloud_firestore-2.10.1-py3.8.egg/google/cloud/firestore_v1/services/firestore/client.py'
RUN sed -i 's/from google.cloud.location import locations_pb2//g' '/usr/local/lib/python3.8/site-packages/google_cloud_firestore-2.10.1-py3.8.egg/google/cloud/firestore_v1/services/firestore/transports/base.py'
RUN sed -i 's/from google.cloud.location import locations_pb2//g' '/usr/local/lib/python3.8/site-packages/google_cloud_firestore-2.10.1-py3.8.egg/google/cloud/firestore_v1/services/firestore/transports/grpc.py'
RUN sed -i 's/from google.cloud.location import locations_pb2//g' '/usr/local/lib/python3.8/site-packages/google_cloud_firestore-2.10.1-py3.8.egg/google/cloud/firestore_v1/services/firestore/transports/grpc_asyncio.py'
RUN sed -i 's/from google.cloud.location import locations_pb2//g' '/usr/local/lib/python3.8/site-packages/google_cloud_firestore-2.10.1-py3.8.egg/google/cloud/firestore_v1/services/firestore/transports/rest.py'
RUN sed -i 's/from google.cloud.location import locations_pb2//g' '/usr/local/lib/python3.8/site-packages/google_cloud_firestore-2.10.1-py3.8.egg/google/cloud/firestore_v1/services/firestore/async_client.py'

ENTRYPOINT [ "python", "/examples/ERP/classify_P300_bi.py" ]
