import json
import logging
import math
import os
import requests
import sys
import base64
from datetime import timedelta
from io import BytesIO
from time import time
from PIL import Image
import dlib

from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
from rediscluster import RedisCluster
from redis import Redis
import msgpack_numpy as m
import pickle

m.patch()

logging.getLogger().setLevel(logging.INFO)
app = Flask(__name__, static_folder='static')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg', 'JPEG'}
face_detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

redis_service = 'localhost' if (os.environ.get("REDIS_SERVICE") is None) else os.environ.get("REDIS_SERVICE")
redis_port = 6379 if (os.environ.get("REDIS_PORT") is None) else os.environ.get("REDIS_PORT")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        # 获取上传的文件对象
        picture = request.files["picture"]
        if not (picture and allowed_file(picture.filename)):
            return jsonify({"error": 1001, "msg": u"请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
        align = request.form.get('align')

        start = time()
        recognition_result = detection(picture, align)
        end = time()
        image_base64 = imageFile_to_base64(picture)
        return render_template('upload_ok.html', recognition_result=recognition_result,
                               time_consume=end - start,
                               image_base64=image_base64)
    return render_template('upload.html')


class FaceRecognition:
    def __init__(self):
        self.data_from = None
        self.min_distance = None
        self.index = None
        self.length = None
        self.labels = None
        self.encodings = None
        self.expires = 0

    def load_data_from_redis(self):
        try:
            r = RedisCluster(host=redis_service, port=int(redis_port), skip_full_coverage_check=True)
            self.length = r.llen("labels")
            self.labels = r.lrange("labels", 0, self.length)
            self.encodings = r.lrange("encodings", 0, self.length)
            self.data_from = "redis"
        except:
            with open("lfw_train.model", "rb") as f:
                data = pickle.load(f)
                self.labels = data["labels"]
                self.encodings = data["encodings"]
                self.length = len(self.labels)
                self.data_from = "local"

    def face_compute(self, image):
        img_enc = np.array(face_encoder.compute_face_descriptor(image, 1))
        if self.expires < time():
            self.load_data_from_redis()
            self.expires = time() + 20
        self.index = math.inf
        self.min_distance = math.inf
        logging.info("data from %s", self.data_from)
        for i in range(self.length):
            if self.data_from == "redis":
                face_encoding = m.unpackb(self.encodings[i])
            else:
                face_encoding = self.encodings[i]
            face_distance = np.linalg.norm(face_encoding - img_enc)
            if face_distance < self.min_distance:
                self.index = i
                self.min_distance = face_distance
        if self.min_distance < 0.6:
            recognition_result = self.labels[self.index]
        else:
            recognition_result = "Unknown"
        return recognition_result


def detection(picture, align):
    img = Image.open(picture)
    img = np.array(img)
    detections, scores, _ = face_detector.run(img, 1)
    resp = []
    if len(detections) > 0:
        for idx, d in enumerate(detections):
            response = extra_face(img, d, align)
            if response is not None:
                resp.append(response)
    return resp

def extra_face(img_np, detection, align):
    left = detection.left()
    right = detection.right()
    top = detection.top()
    bottom = detection.bottom()
    detected_face = img_np[max(0, top): min(bottom, img_np.shape[0]), max(0, left): min(right, img_np.shape[1])]
    if align:
        rect_dict = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
        res = alignment(rect_dict, img_np)
    else:
        face_resized = cv2.resize(detected_face, (150, 150))
        face_rec = FaceRecognition()
        res = face_rec.face_compute(face_resized)
    return res

def alignment(rect, img_np, size=150):
    rect_dlib = dlib.rectangle(rect['left'], rect['top'], rect['right'], rect['bottom'])
    img_shape = sp(img_np, rect_dlib)
    aligned_face = dlib.get_face_chip(img_np, img_shape, size=size)
    face_rec = FaceRecognition()
    res = face_rec.face_compute(aligned_face)
    return res

# 将图片文件转为base64编码字符串
def imageFile_to_base64(image):
    image_file = image.stream  # 转化为流对象
    image_file.seek(0)
    image_binary = image_file.read()
    try:
        image = Image.open(BytesIO(image_binary)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unsupported image format: {e}")
    with BytesIO() as buffer:
        format = image.format if image.format else "JPEG"
        image.save(buffer, format=format)
        image_base64 = base64.b64encode(buffer.getvalue())
        return image_base64.decode('utf-8')


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
