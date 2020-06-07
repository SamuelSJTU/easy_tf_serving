import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter

def convert_keras_model_to_tf_serving(model, output_dir, version):
    export_path = os.path.join(output_dir, str(version))
    print('export_path = {}\n'.format(export_path))
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format='tf',
        signatures=None
    )

    print('Saved model.')

def convert_model_to_tf_serving(inputs, outputs, signature_name, output_dir, version):
    export_path = os.path.join(output_dir, str(version))
    print('export_path = {}\n'.format(export_path))
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    builder = saved_model_builder.SavedModelBuilder(export_path)
    signature = predict_signature_def(inputs=inputs, outputs=outputs)

    sess = K.get_session()
    builder.add_meta_graph_and_variables(sess=sess,
                                        tags=[tag_constants.SERVING],
                                        signature_def_map={signature_name: signature})
    builder.save()

if __name__ == "__main__":
    import sys
    import numpy as np
    sys.path.insert(0, '/home/samuel/rsproject/DeepCTR')
    from deepctr.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
    from deepctr.models import DIN
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

    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    model = DIN(feature_columns, behavior_feature_list)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)
    print(model.input)
    print(model.output)
    
    # v1
    # convert_keras_model_to_tf_serving(model=model, output_dir='./model/criteo', version=1)

    # v2
    if isinstance(model.input, list):
        inputs = {input_tensor.name[:-4]:input_tensor  for input_tensor in model.input}
    else:
        input_tensor = model.input
        inputs = {input_tensor.name[:-4]:input_tensor}
    if isinstance(model.output, list):
        outputs = {output_tensor.name[:-4]:output_tensor  for output_tensor in model.output}
    else:
        output_tensor = model.output
        outputs = {output_tensor.name[:-4]:output_tensor}
    signature_name = 'serving_default'
    convert_model_to_tf_serving(inputs=inputs, outputs=outputs, signature_name=signature_name, 
            output_dir='./model/criteo', version=1)

