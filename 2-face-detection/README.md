这是一个人脸识别项目中的一部分，由front->detection->alignment->representation->dateface对比 五个部分组成

由于detection会请求面部矫正服务和特征提取服务，所以需要设置相应的环境变量，否则会默认访问本地的服务
```angular2html
FACE_ALIGNMENT_SERVICE="your alignment's host"
FACE_ALIGNMENT_PORT="your alignment's port"
FACE_REPRESENTATION_SERVICE="your representation's host"
FACE_REPRESENTATION_PORT="your representation's port"
```