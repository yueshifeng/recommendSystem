# -*- coding: utf-8 -*-
import logging
import tensorflow as tf
import sys
import tensornet as tn
from tensorflow.python.keras import backend as K
class GradientLayer(tf.keras.layers.Layer):

    def __init__(self,  **kwargs):
        super(GradientLayer, self).__init__(**kwargs)
    def call(self, w):
        # self.bias = print_tensor(self.bias, message='print_tensor bias')
        x = tf.stop_gradient(w)
        return x
class printLayerv1(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(printLayerv1, self).__init__(**kwargs)

        def call(self, x):
            x = tf.keras.backend.print_tensor(x.get_weights(), message='sum_embs layer')
            return x

class FMLayer(tf.keras.layers.Dense):

    def __init__(self, seed=1024):
        super(FMLayer, self).__init__(8)
        self.seed = seed


    def build(self, input_shape):  # TensorShape of input when run call(), inference from inputs

        self.fm_matrix = self.add_variable(
                name="weight",
                shape=[input_shape[-1], 8],
                initializer=tf.keras.initializers.GlorotNormal(seed=None),
                #regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
        self.fm_fn_dense = tf.keras.layers.Dense(1, activation=None, name='deeepfmlinear')
        self.built = True


    def call(self, inputs, training=None):
        fm_matrix = self.fm_matrix
        fm_matrix_square = tf.math.square(fm_matrix)
        emb_inputs_square = tf.math.square(inputs)
        sum_square_by_row = tf.math.square(tf.linalg.matmul(inputs, fm_matrix))
        square_sum_by_row = tf.linalg.matmul(emb_inputs_square, fm_matrix_square)
        high_order_result = 0.5 * tf.math.reduce_sum(tf.math.subtract(sum_square_by_row, square_sum_by_row), axis=1, keepdims=True)
        linear_result = self.fm_fn_dense(inputs)
        fm = tf.math.add(high_order_result, linear_result)
    
        return fm
def fetch_embeddings(inputs):
    columns = {}
    for fea_id in inputs.keys():
        columns[fea_id] = tn.feature_column.category_column(key=fea_id, bucket_size=25600)

    embedding_columns = []
    for fea_id in inputs.keys():
        feature_column = tn.feature_column.embedding_column(name="emb_col_{}".format(fea_id),
                            categorical_column=columns[fea_id], dimension=32, combiner='mean')
        embedding_columns.append(feature_column)

    sparse_opt = tn.core.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

    sparse_embs = tn.layers.EmbeddingFeatures(embedding_columns, sparse_opt, name="sparse_emb_input")(inputs)

    return {fea_id : sparse_embs["emb_col_{}".format(fea_id)] for fea_id in inputs.keys()}
def create_deepFM_sub_model(slot_embs, deep_hidden_units):
    emb_inputs_total = []
    emb_dict = {}
    bais_slots = ['3051', '1570', '2039', '2544', '1568', '3376', '3365', '3369','2597']
    gerneral_slots = ['3371', '3367', '3377', "2599", "2148", "2153", "2162", "2165", "2169", "2123", "2125", "2127",
                      "2128", "2130", "2131", "2137", "2142", "2144", "2149", "2152", "2154", "2156", "1574", "1575",
                      "1576", "1577", "1582", "1589", "1590", "1591", "1592", "1593", "1594", "1614", "1616", "1624",
                      "1625", "1632", "1736", "1737", "1738", "1744", "1745", "1749", "2044", "2040", "2041", "2043",
                      "2045", "2047", "2048", "2049", "2050", "2051", "2052"]
    bais_inputs_all = []
    gerneral_inputs_all =[]
    for feature, shape in slot_embs:
        tmp_input = tn.layers.Input(name=str(feature.feature_id), feature=feature,
                          shape=shape[1:], dtype="float32")
        feature_id = str(feature.feature_id)
        if feature_id in bais_slots:
            bais_inputs_all.append(tmp_input)
        elif feature_id in gerneral_slots:
            gerneral_inputs_all.append(tmp_input)
        
        emb_dict[feature_id] = tmp_input
    # 需要添加映射关系。 sparse table name 到 dense 模型的 input tensor prefix
    # sparse table name 是 columns_group 的 key
    bais_inputs = [emb[:,0:16] for emb in bais_inputs_all]
    gerneral_inputs = [emb[:, 0:16] for emb in gerneral_inputs_all]
    gerneral_inputs.append(emb_dict['1568'][:,16:])
  
    emb_inputs = gerneral_inputs + bais_inputs#[emb for emb in emb_inputs_total]
    # 需要添加映射关系。 sparse table name 到 dense 模型的 input tensor prefix
    # sparse table name 是 columns_group 的 key
    #emb_inputs[0].trainable = False
    gerneral_inputs = tf.concat(gerneral_inputs, axis=1)
    fm_layer = FMLayer()
    fm = fm_layer(gerneral_inputs)
    bais_input = tf.concat(bais_inputs, axis=1)
    print("bais_input_dim: {}".format(bais_input.get_shape()))
    gerneral_input = gerneral_inputs #tf.concat([gerneral_inputs, fm],axis = 1)
    bais_shape = -1
    for i, unit in enumerate(deep_hidden_units):
        dense_layer = tf.keras.layers.Dense(unit, activation='relu', name='dnn_{}'.format(i),  kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00001, l2=0.00001))
        if i== 0:
            gerneral_input = dense_layer(gerneral_input)
        else:
            bais_dense_layer_one = tf.keras.layers.Dense(bais_shape, activation='relu', name='bais_dnn_one_{}'.format(i),
                                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00001,
                                                                                               l2=0.00001))
            bais_dense_layer_two = tf.keras.layers.Dense(bais_shape, activation='sigmoid', name='bais_dnn_two_{}'.format(i),
                                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00001,
                                                                                               l2=0.00001))
    
            bais_input_one = bais_dense_layer_one(bais_input)
            bais_input_two = bais_dense_layer_two(bais_input_one)*2
            gerneral_input = dense_layer(tf.math.multiply(gerneral_input,bais_input_two))#tf.concat([dense_layer(tf.math.multiply(gerneral_input,bais_input_two)), fm], axis=1)
        bais_shape = unit
    bais_dense_layer_final_one = tf.keras.layers.Dense(bais_shape, activation='relu', name='bais_dnn_one_{}'.format(3),
                                                       kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00001,
                                                                                                     l2=0.00001))
    bais_dense_layer_final_two = tf.keras.layers.Dense(bais_shape, activation='sigmoid',
                                                       name='bais_dnn_two_{}'.format(3),
                                                       kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00001,
                                                                                                     l2=0.00001))
    bais_input_final_one = bais_dense_layer_final_one(bais_input)
    print("bais_input_one_dim: {}".format(bais_input_final_one.get_shape()), ", i =", i)
    bais_input_final_two = bais_dense_layer_final_two(bais_input_final_one) * 2
    gerneral_input = tf.math.multiply(gerneral_input, bais_input_final_two)
    deep = tf.concat([gerneral_input, fm], axis=1)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(deep)
    task_output = {"video_id_rank_finish_nb_lr_rongh_bundle":tf.identity(output, name="video_id_rank_finish_nb_lr_rongh_bundle")}
    model = tn.model.Model(inputs=emb_dict, outputs=task_output, name="sub_model")
    
       
   
    return {
            "train": model,
            "predict": model,
    }


def DEEPFM(slots, dnn_hidden_units=(64,32)):
    logging.info('dnn_hidden_units={}'.format(dnn_hidden_units))

    feature_slots = {slot_id : tn.feature_column.FeatureSlot(slot_id) for slot_id in slots}
    sparse_features = {slot_id : tn.feature_column.Feature(feature_id=slot_id, feature_slot=feature_slots[slot_id], sparse=True)
                            for slot_id in slots}
    inputs = {fea_id : tn.layers.Input(name=fea_id, feature=feature,
                              shape=(None,), dtype="int64", sparse=True)
                            for fea_id, feature in sparse_features.items()}
    embs = fetch_embeddings(inputs)
    # 自定义优化器。feature_drop_show 设置为 -1 表示不删除 feasign
    emb_input_shapes = sorted([(sparse_features[fea_id], emb.shape) for fea_id, emb in embs.items()])
    

    models = create_deepFM_sub_model(emb_input_shapes, dnn_hidden_units)

    train_outputs = models['train'](embs)
    predict_outputs = models['predict'](embs)

    return {
        "train" : tn.model.Model(inputs=inputs, outputs=train_outputs, name="full_model"),
        "predict" : tn.model.Model(inputs=inputs, outputs=predict_outputs, name="full_model"),
    }



