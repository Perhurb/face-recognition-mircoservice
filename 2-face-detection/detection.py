# coding:utf-8
from __future__ import print_function

import json
import logging
import os
import sys

import cv2
import dlib
import msgpack_numpy as m
import requests
from PIL import Image
from flask import Flask
from flask import jsonify
from flask import request
from werkzeug.datastructures import FileStorage
import numpy as np

alignment_serviceName = "localhost" if (os.environ.get(" FACE_ALIGNMENT_SERVICE") is None) else os.environ.get(
    "FACE_ALIGNMENT_SERVICE")
alignment_port = 5003 if (os.environ.get("FACE_ALIGNMENT_PORT") is None) else os.environ.get("FACE_ALIGNMENT_PORT")
alignment_service = "http://" + alignment_serviceName + ":" + str(alignment_port)

representation_serviceName = "localhost" if (os.environ.get("FACE_REPRESENTATION_SERVICE") is None) else os.environ.get(
    "FACE_REPRESENTATION_SERVICE")
representation_port = 5004 if (os.environ.get("FACE_REPRESENTATION_PORT") is None) else os.environ.get(
    "FACE_REPRESENTATION_PORT")
representation_service = "http://" + representation_serviceName + ":" + str(representation_port)

logging.getLogger().setLevel(logging.INFO)
app = Flask(__name__)
face_detector = dlib.get_frontal_face_detector()
m.patch()


# 创建线程锁对象


@app.route('/', methods=['POST'])  # 添加路由
def process():
    image: FileStorage = request.files["picture"]
    align = request.headers.get("align")
    print(align)
    res = detection(image, align)
    data = {"result": res}
    return jsonify(data)


def detection(img, align):
    """
    :param align: 是否采用面部矫正
    :param img: a path, url, base64 or numpy array.
    :return: a list of extracted faces.For simplicity, only return a dict of a face
    """
    img = Image.open(img)
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
        files = {'rect': json.dumps(rect_dict),
                 'face_unaligned': m.packb(img_np)}
        response = curl_res(alignment_service, files)
    else:
        face_resized = cv2.resize(detected_face, (150, 150))
        img_encoded = m.packb(face_resized)
        files = {'face_with_size': img_encoded}
        response = curl_res(representation_service, files)
    return response


def curl_res(url, files, ):
    response = requests.post(url=url, files=files)
    if response.status_code != 200:
        logging.error("err: %s".format(response.text))
        return None
    else:
        return response.json()


def load_image(img):
    """Load image from path, url, base64 or numpy array.
    Args:
        img: a numpy or file.
    Returns:
        numpy array: the loaded image.
    """
    # The image is already a numpy array
    if type(img).__module__ == np.__name__:
        return img
    return cv2.imread(img)


def run():
    if len(sys.argv) < 2:
        logging.error("usage: %s port" % (sys.argv[0]))
        sys.exit(-1)
    p = int(sys.argv[1])
    logging.info("start at port %s" % p)
    if sys.platform == "linux":
        app.run(host='::', port=p, debug=True, threaded=True)
    else:
        app.run(host='0.0.0.0', port=p, debug=True, threaded=True)


if __name__ == '__main__':
    run()
