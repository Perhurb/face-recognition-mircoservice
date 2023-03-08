FROM python:3.9-slim
RUN mkdir -p /opt/microservices
COPY requirements.txt /opt/microservices
COPY alignment.py /opt/microservices
COPY shape_predictor_5_face_landmarks.dat /opt/microservices
WORKDIR /opt/microservices
RUN apt-get update
RUN apt-get install -y build-essential cmake
RUN apt-get install -y libopenblas-dev liblapack-dev
RUN apt-get install -y libx11-dev libgtk-3-dev
RUN apt-get install -y python3-dev python3-numpy python3-pip
RUN pip3 install --no-cache-dir -r ./requirements.txt

EXPOSE 9080
CMD [ "python3", "alignment.py", "9080" ]