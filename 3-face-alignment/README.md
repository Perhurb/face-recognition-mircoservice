这是一个人脸识别项目中的一部分，由front->detection->alignment->representation->dateface对比 五个部分组成

由于alignment会请求特征提取服务，所以需要设置相应的环境变量，否则会默认访问本地的服务
```angular2html
FACE_REPRESENTATION_SERVICE="your representation's host"
FACE_REPRESENTATION_PORT="your representation's port"
```