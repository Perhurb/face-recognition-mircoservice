FROM python:3.9-slim
RUN mkdir -p /opt/microservices
COPY requirements.txt /opt/microservices
COPY recognition-redis.py /opt/microservices
WORKDIR /opt/microservices
RUN pip3 install --no-cache-dir -r ./requirements.txt
RUN python3 -m grpc_tools.protoc -I protos/ --python_out=. --grpc_python_out=. protos/face.proto

EXPOSE 9080
CMD [ "python3", "recognition-redis.py", "9080" ]