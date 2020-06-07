import os
import sys
import json
import time
import grpc
import subprocess
import setproctitle
import requests
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
from deepctr.models import DIEN, DeepFM

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_xy_fd():

    feature_columns = [SparseFeat('user',3,embedding_dim=10),SparseFeat(
        'gender', 2,embedding_dim=4), SparseFeat('item_id', 3 + 1,embedding_dim=8), SparseFeat('cate_id', 2 + 1,embedding_dim=4),DenseFeat('pay_score', 1)]
    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=3 + 1,embedding_dim=8,embedding_name='item_id'), maxlen=4),
                        VarLenSparseFeat(SparseFeat('hist_cate_id', 2 + 1,embedding_dim=4, embedding_name='cate_id'), maxlen=4)]

    behavior_feature_list = ["item", "cate_id"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    cate_id = np.array([1, 2, 2])  # 0 is mask value
    pay_score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [3, 2, 1, 0], [1, 2, 0, 0]])
    hist_cate_id = np.array([[1, 2, 2, 0], [2, 2, 1, 0], [1, 2, 0, 0]])

    feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                    'hist_item_id': hist_iid, 'hist_cate_id': hist_cate_id, 'pay_score': pay_score}
    x = {name:feature_dict[name] for name in get_feature_names(feature_columns)}
    y = [1, 0, 1]
    return x, y, feature_columns, behavior_feature_list


def client_grpc_criteo():
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    input_dict_np = {
        'user':np.int32,
        'gender': np.int32,
        'item_id':np.int32,
        'cate_id':np.int32,
        'pay_score':np.float32,
        'hist_item_id':np.int32,
        'hist_cate_id':np.int32,
    }
    model_input = {k:np.array([v[0]], dtype=input_dict_np[k]) for k, v in x.items()}
   
    ## start grpc 
    grpc_port = 8500
    grpc_channel = grpc.insecure_channel('127.0.0.1:{}'.format(grpc_port))
    grpc_stub = prediction_service_pb2_grpc.PredictionServiceStub(grpc_channel)
    
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'criteo'
    request.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    
    x = model_input
    input_dict = {
        'user':tf.int32,
        'gender': tf.int32,
        'item_id':tf.int32,
        'cate_id':tf.int32,
        'pay_score':tf.float32,
        'hist_item_id':tf.int32,
        'hist_cate_id':tf.int32
    }
    for key in x:
        tensor = tf.contrib.util.make_tensor_proto(x[key], shape=list(x[key].shape), dtype=input_dict[key])
        request.inputs[key].CopyFrom(tensor)

    res = grpc_stub.Predict.future(request, 4.0)
    if res.code() == grpc.StatusCode.OK:
        results = dict()
        for key in res.result().outputs:
            results[key] = tf.contrib.utils.make_ndarray(res.result().outputs[key])
        print(results)

    else:
        print(res.details())


def client_grpc_fashion():
    # load dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # scale the values to 0.0 to 1.0
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # reshape for feeding into the model
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
    print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

    ## start grpc 
    grpc_port = 8500
    grpc_channel = grpc.insecure_channel('127.0.0.1:{}'.format(grpc_port))
    grpc_stub = prediction_service_pb2_grpc.PredictionServiceStub(grpc_channel)
    
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'fashion'
    request.model_spec.signature_name = 'predict'
    # image = test_images[0:1]
    image = np.ones([1,28,28,1], dtype=np.float32)
    tensor = tf.contrib.util.make_tensor_proto(image, shape=image.shape, dtype=tf.float32)
    request.inputs['images'].CopyFrom(tensor)

    for i in range(1):
        res = grpc_stub.Predict.future(request, 5.0)
        if res.code() == grpc.StatusCode.OK:
            results = dict()
            for key in res.result().outputs:
                results[key] = tf.make_ndarray(res.result().outputs[key])
            print(results)
        else:
            print(res.details())
            print(res.code())
            print(res.exception())
            print(res.traceback())
            print(res.is_active)


if __name__ == "__main__":
    client_grpc_fashion()
    client_grpc_criteo()
    
