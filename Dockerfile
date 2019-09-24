FROM nvcr.io/nvidia/tensorflow:19.07-py3

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt install -y --no-install-recommends wget git vim

RUN apt install -y protobuf-compiler python-pil python-lxml python-tk

RUN  python3 -m pip install lxml
RUN  python3 -m pip install matplotlib

RUN cd ~ &&\
    git clone -b faster-rcnn https://github.com/dl-framework-benchmark/models.git
RUN cd /root/models/research && \
    wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip && \
    unzip protobuf.zip
RUN  cd /root/models/research && \
     ./bin/protoc object_detection/protos/*.proto --python_out=.
ENV PYTHONPATH $PYTHONPATH:/root/models/research/:/root/models/research/slim
ADD ./scripts /root/scripts
