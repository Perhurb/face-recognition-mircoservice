# coding:utf-8
from __future__ import print_function

import logging
import os
import sys

import dlib
import numpy as np
import grpc
from flask import Flask
from flask import jsonify
from flask import request
import msgpack_numpy as m

from face_pb2 import RecRequest
from face_pb2_grpc import FaceRecognitionServiceStub

app = Flask(__name__)
logging.getLogger().setLevel(level=logging.INFO)
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

recognition_service = "localhost" if (os.environ.get("FACE_RECOGNITION_SERVICE") is None) else os.environ.get("FACE_RECOGNITION_SERVICE")
recognition_port = 5005 if (os.environ.get("FACE_RECOGNITION_PORT") is None) else os.environ.get("FACE_RECOGNITION_PORT")
recognitionHostname = recognition_service+":"+str(recognition_port)
@app.route('/', methods=['POST'])  # 添加路由
def process():
    image = request.files["face_with_size"].read()
    if image is None:
        raise ValueError("no face data")
    image = m.unpackb(image)
    face_encodings = np.array(face_encoder.compute_face_descriptor(image, 1))
    with grpc.insecure_channel(recognitionHostname) as channel:
        stub = FaceRecognitionServiceStub(channel)
        face_enc_list = face_encodings.tolist()
        req = RecRequest()
        req.feature.extend(face_enc_list)
        logging.info("%s will be invoked with RPC, stub is %s", recognitionHostname, stub)
        recognition_result = stub.FaceRecognition(req).name
        logging.info(recognition_result)
        data = {"result": recognition_result}
        return jsonify(data)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        logging.error("usage: %s port" % (sys.argv[0]))
        sys.exit(-1)
    p = int(sys.argv[1])
    logging.info("start at port %s" % p)
    if sys.platform == "linux":
        app.run(host='::', port=p, debug=True, threaded=True)
    else:
        app.run(host='0.0.0.0', port=p, debug=True, threaded=True)
