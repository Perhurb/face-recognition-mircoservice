FROM python:3.9-slim
RUN mkdir -p /opt/microservices
COPY requirements.txt /opt/microservices
COPY representation.py /opt/microservices
COPY dlib_face_recognition_resnet_model_v1.dat /opt/microservices
WORKDIR /opt/microservices
RUN pip3 install --no-cache-dir -r ./requirements.txt
RUN python3 -m grpc_tools.protoc -I protos/ --python_out=. --grpc_python_out=. protos/face.proto

EXPOSE 9080
CMD [ "python3", "representation.py", "9080" ]