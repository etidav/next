ARG TENSORFLOW_BASE=tensorflow/tensorflow:2.5.0-gpu
FROM ${TENSORFLOW_BASE} as base_img

ARG TENSORFLOW_BASE
RUN echo "" && echo "Friendly info: this docker uses the base image: ${TENSORFLOW_BASE}" && echo ""

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
RUN apt-get update

# Install python 3.7
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.7 -y

# Add 3.7 to the available alternatives
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 100
RUN python3.7 -m pip install --upgrade pip

# python install from pip3
COPY docker/requirements/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt && \
    rm /requirements.txt
RUN pip3 uninstall nvidia_cublas_cu11 -y

# Make data and libs accressible
COPY data /next/data
COPY run /next/run
COPY utils /next/utils
COPY model /next/model
COPY exp /next/exp
RUN mkdir -p /next/result
ENV PYTHONPATH=$PYTHONPATH:/next

# Define workdir
WORKDIR /next

# Define the correct entrypoint
ENTRYPOINT []
