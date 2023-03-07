FROM python:3.9-slim
RUN mkdir -p /opt/microservices
COPY templates /opt/microservices/templates
COPY static /opt/microservices/static
COPY requirements.txt /opt/microservices
COPY front.py /opt/microservices
WORKDIR /opt/microservices
RUN pip3 install --no-cache-dir -r ./requirements.txt

EXPOSE 9080
CMD [ "python3", "front.py", "9080" ]