# Obraz bazowy z Pythonem
FROM python:3.8-slim

# Zależności
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libopenblas-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# MXNet
RUN pip install numpy==1.23.5 && pip install mxnet

# OpenCV
RUN pip install opencv-python


# Katalog roboczy
WORKDIR /app

# Ustaw polecenie uruchamiające aplikację
CMD ["python", "/dataset/script.py"]


# docker build -t mxnet-dataset .
# docker run -v /Users/dawid/PycharmProjects/FaceAuthorization/dataset:/dataset mxnet-dataset

# docker buildx build --platform linux/amd64 -t mxnet-dataset_x86 . --load
# docker run -it -v /Users/dawid/PycharmProjects/FaceAuthorization/dataset:/dataset  mxnet-dataset_x86