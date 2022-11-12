FROM python:3 as dev
WORKDIR /src

COPY . .

RUN apt-get -y update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip setuptools wheel
RUN apt-get install -y python3-opencv
RUN pip3 install -r requirements.txt
