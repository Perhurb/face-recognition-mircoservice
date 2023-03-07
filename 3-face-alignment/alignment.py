# coding:utf-8
from __future__ import print_function

import json
import sys
import threading
import requests
import cv2
import dlib
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
import os
import logging
import dlib
import numpy as np
import msgpack_numpy as m

representation_serviceName = "localhost" if (os.environ.get("FACE_REPRESENTATION_SERVICE") is None) else os.environ.get(
    "FACE_REPRESENTATION_SERVICE")
representation_port = 5004 if (os.environ.get("FACE_REPRESENTATION_PORT") is None) else os.environ.get(
    "FACE_REPRESENTATION_PORT")
representation_service = "http://" + representation_serviceName + ":" + str(representation_port)

logging.getLogger().setLevel(logging.INFO)
app = Flask(__name__)
sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")


@app.route('/', methods=['POST'])  # 添加路由
def process():
    # 创建线程锁对象
    rect = request.files["rect"].read()
    rect = json.loads(rect)
    rect_dlib = dlib.rectangle(rect['left'], rect['top'], rect['right'], rect['bottom'])
    img = request.files["face_unaligned"].read()
    img_dec = m.unpackb(img)

    resp = alignment(img_dec, rect_dlib)
    return resp


def alignment(img, rect, size=150):
    """
    :param size: 裁定后的大小
    :param img: 人脸检测后的图片
    :param rect: 原始图片.
    :return: a list of extracted faces.For simplicity, only return a dict of a face
    """
    img_shape = sp(img, rect)
    aligned_face = dlib.get_face_chip(img, img_shape, size=size)
    img_encoded = m.packb(aligned_face)
    files = {'face_with_size': img_encoded}
    response = requests.post(url=representation_service, files=files)
    return make_response(response.content)


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
