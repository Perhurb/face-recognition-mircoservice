### 前端界面 执行顺序为front detection_manual_for_front recognition(可以选择由redis或者文件中读取)
import json
import logging
import os
import requests
import sys
import base64
from datetime import timedelta
from io import BytesIO
from time import time
from PIL import Image

from flask import Flask, request, render_template, jsonify

# 定义detection服务
detection_serviceName = "localhost" if (os.environ.get("FACE_DETECTION_SERVICE") is None) else os.environ.get(
    "FACE_DETECTION_SERVICE")
detection_port = 5002 if (os.environ.get("FACE_DETECTION_PORT") is None) else os.environ.get("FACE_DETECTION_PORT")
detection_service = "http://" + str(detection_serviceName) + ":" + str(detection_port)

logging.getLogger().setLevel(logging.INFO)
app = Flask(__name__, static_folder='static')
app.send_file_max_age_default = timedelta(seconds=1)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg', 'JPEG'}


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
        files = {'picture': (picture.filename, picture, picture.mimetype)}
        start = time()
        try:
            response = requests.post(url=detection_service, files=files)
            response_loads = json.loads(response.text)
            recognition_result = response_loads["result"]
        except:
            recognition_result = "服务开小差了，请稍后再试"
        end = time()
        image_base64 = imageFile_to_base64(picture)
        return render_template('upload_ok.html', recognition_result=recognition_result,
                               time_consume=end - start,
                               image_base64=image_base64)
    return render_template('upload.html')


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
