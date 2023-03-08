FROM python:3.9-slim
RUN mkdir -p /opt/microservices
COPY requirements.txt /opt/microservices
COPY recognition-redis.py /opt/microservices
COPY protos /opt/microservices/protos
WORKDIR /opt/microservices
RUN apt-get update
RUN apt-get install -y build-essential cmake
RUN apt-get install -y libopenblas-dev liblapack-dev
RUN apt-get install -y libx11-dev libgtk-3-dev
RUN apt-get install -y python3-dev python3-numpy python3-pip
RUN pip3 install --no-cache-dir -r ./requirements.txt
RUN python3 -m grpc_tools.protoc -I protos/ --python_out=. --grpc_python_out=. protos/face.proto

EXPOSE 9080
CMD [ "python3", "recognition-redis.py", "9080" ]