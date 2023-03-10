import math
import sys
from concurrent import futures
import grpc
import logging
import os
import pickle
from face_pb2 import RecRequest, RecResponse
from face_pb2_grpc import FaceRecognitionServiceServicer, add_FaceRecognitionServiceServicer_to_server

from redis import Redis
from rediscluster import RedisCluster
import msgpack_numpy as m
import numpy as np


logging.getLogger().setLevel(logging.INFO)
m.patch()

redis_service = 'face-redis' if (os.environ.get("REDIS_SERVICE") is None) else os.environ.get("REDIS_SERVICE")
redis_port = 6379 if (os.environ.get("REDIS_PORT") is None) else os.environ.get("REDIS_PORT")
# r = Redis(host=redis_service, port=int(redis_port))
# r = RedisCluster(host=redis_service, port=int(redis_port), skip_full_coverage_check=True)


class FaceRecognition(FaceRecognitionServiceServicer):
    def __init__(self):
        self.data = None
        self.min_distance = None
        self.index = None
        self.length = None
        self.labels = None
        self.encodings = None

    def load_data(self):
        with open("lfw_train.model", "rb") as f:
            self.data = pickle.load(f)
            self.labels = self.data["labels"]
            self.encodings = self.data["encodings"]
            self.length = len(self.labels)

    def FaceRecognition(self, request, context):
        self.load_data()
        img_enc = np.array(request.feature)
        self.index = math.inf
        self.min_distance = math.inf
        for i in range(self.length):
            face_encoding = self.encodings[i]
            face_distance = np.linalg.norm(face_encoding - img_enc)
            if face_distance < self.min_distance:
                self.index = i
                self.min_distance = face_distance

        if self.min_distance < 0.6:
            recognition_result = self.labels[self.index]
        else:
            recognition_result = "Unknown"
        return RecResponse(name=recognition_result)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_FaceRecognitionServiceServicer_to_server(FaceRecognition(), server)
    server.add_insecure_port('[::]:{0}'.format(sys.argv[1]))
    logging.info("??????%s?????????..." % sys.argv[1])
    server.start()
    logging.info("???????????????...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()

