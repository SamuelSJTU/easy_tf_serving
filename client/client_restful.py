import requests
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models import DIEN, DeepFM


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


def client_restful_fashion():
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

    data = json.dumps({"signature_name": "predict", "instances": test_images[0:3].tolist()})
    # print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))
    data = json.dumps({"signature_name": "predict", "instances": [{'images': test_images[0].tolist()}]})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/fashion:predict', data=data, headers=headers)
    json_response = json.loads(json_response.text)
    print(json_response)
    predictions = json_response['predictions']

    print('The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
    class_names[np.argmax(predictions[0])], np.argmax(predictions[0]), class_names[test_labels[0]], test_labels[0]))


def client_restful_criteo():
    data = pd.read_csv('./data/criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4)
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    
    # model_input = [data[name].iloc[0] for name in feature_names]
    # model_input = [{name:data[name].iloc[0]} for name in feature_names]
    model_input = [{name:data[name].iloc[0] for name in feature_names}]
    print(model_input)
    data = json.dumps({"signature_name": "serving_default", "instances": model_input}, cls=NpEncoder)
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/criteo:predict', data=data, headers=headers)
    json_response = json.loads(json_response.text)
    print(json_response)


def client_restful_criteo_v2():
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    model_input = [{name:np.array([data[0]]) for name, data in x.items()}]
    
    # model_input = [np.array([np.array(data[0]) for name, data in x.items()])]
    # model_input = [{name:data[name].iloc[0] for name in feature_names}]
    print(model_input)
    data = json.dumps({"signature_name": "serving_default", "instances": model_input}, cls=NpEncoder)
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/criteo:predict', data=data, headers=headers)
    json_response = json.loads(json_response.text)
    print(json_response)


if __name__ == "__main__":
    client_restful_fashion()  # mnist - fashion offical demo
    client_restful_criteo()  # criteo data
    client_restful_criteo_v2()