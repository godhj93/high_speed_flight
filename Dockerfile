FROM tensorflow/tensorflow:2.8.0-gpu

RUN apt-get update && apt-get install sudo
RUN apt-get install lsb-core lsb-release -y 
RUN apt install nano && apt install python3-pip
RUN pip install --upgrade pip
RUN apt install wget git zip -y

RUN cd ~/ && git clone https://github.com/godhj93/high_speed_flight.git
RUN cd ~/high_speed_flight && pip install -r requirements.txt
