这是一个人脸识别项目中的一部分，由front->detection->alignment->representation->dateface对比 五个部分组成

由于representation会访问database比较服务，他们通过grpc通信，所以遵循了定义好的通信协议