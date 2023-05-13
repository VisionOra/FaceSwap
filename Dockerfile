FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && \
    apt-get install -y wget && \
    wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    apt-get update && \
    apt-get install -y git && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    apt install -y libprotobuf-dev protobuf-compiler && \
    apt-get clean

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools


WORKDIR /faceswap2
COPY ./requirements.txt /faceswap2/requirements.txt
RUN pip install -r /faceswap2/requirements.txt


# Use external volume for data
ENV NVIDIA_VISIBLE_DEVICES 1
CMD ["python", "main.py"]
