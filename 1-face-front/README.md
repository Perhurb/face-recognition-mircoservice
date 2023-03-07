这是一个人脸识别项目中的一部分，由front->detection->alignment->representation->dateface对比 五个部分组成

由于front会请求detection服务，所以必须在环境变量中设置detection服务的参数
```angular2html
FACE_DETECTION_SERVICE="your detection's host"
FACE_DETECTION_PORT="your detection's port"
```