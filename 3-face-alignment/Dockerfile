FROM python:3.9-slim
RUN mkdir -p /opt/microservices
COPY requirements.txt /opt/microservices
COPY alignment.py /opt/microservices
COPY shape_predictor_5_face_landmarks.dat /opt/microservices
WORKDIR /opt/microservices
RUN pip3 install --no-cache-dir -r ./requirements.txt

EXPOSE 9080
CMD [ "python3", "alignment.py", "9080" ]