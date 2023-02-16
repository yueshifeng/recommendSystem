# -*- coding: utf-8 -*-

import tensorflow as tf
import tensornet as tn

from ..config import config as C
from .layer import *


def dict_to_sorted_list(d, key_func=None):
    if key_func is None:
        key_func = lambda x: x[0]
    return [v for k, v in sorted(list(d.items()), key=key_func)]


def create_tower_teacher(emb_input_shapes, tower_name):
    sparse_emb_inputs = {
        str(feature.feature_id): tn.layers.Input(name="emb_{}_{}".format(tower_name, feature.feature_id),
                                                 feature=feature, dtype="float32", shape=shape[1:])
        for feature, shape in emb_input_shapes}
    weight_concat = tf.keras.layers.Concatenate(name='weight_concat_{}'.format(tower_name), axis=-1)(
        dict_to_sorted_list(sparse_emb_inputs))

    cross = CrossNet()(weight_concat)
    deep = tf.keras.layers.Dense(128, activation='relu')(weight_concat)
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)
    merge = tf.keras.layers.Concatenate(name='final_concat_{}'.format(tower_name), axis=-1)([deep, cross])
    output = tf.keras.layers.Dense(16, activation=None)(merge)
    output = tf.keras.layers.Dense(1, activation=None, name='pred_{}'.format(tower_name))(output)
    outputs = {'logit': tf.identity(output, name="{}_logit".format(tower_name))}

    model = tn.model.Model(inputs=sparse_emb_inputs, outputs=outputs, name="sub_model_" + tower_name)

    return model


def create_tower(emb_input_shapes, tower_name, output_dim=16, mask_tensor=None):
    sparse_emb_inputs = {
        str(feature.feature_id): tn.layers.Input(name="emb_{}_{}".format(tower_name, feature.feature_id),
                                                 feature=feature, dtype="float32", shape=shape[1:])
        for feature, shape in emb_input_shapes}
    weight_concat = tf.keras.layers.Concatenate(name='weight_concat_{}'.format(tower_name), axis=-1)(
        dict_to_sorted_list(sparse_emb_inputs))

    if mask_tensor is not None:
        ple_outputs = PLE(name='ple_{}'.format(tower_name), num_tasks=2, num_shared_experts=4,
                          num_specific_experts=4, expert_dnn_units=(32,), gate_dnn_units=(), expert_dnn_params=dict(),
                          gate_dnn_params=dict())(weight_concat)
        outputs = [DNN((output_dim,), output_activation='linear', l2_reg=0, dropout_rate=0,
                       name='{}_{}_emb'.format(task_name, tower_name))(ple_outputs[i]) for i, task_name in
                   enumerate(['td', 'hpld'])]
        outputs = tf.where(tf.reshape(tf.cast(mask_tensor == 1, dtype='bool'), (-1, 1)), outputs[1], outputs[0],
                           name='{}_emd'.format(tower_name))
        sparse_emb_inputs['4575'] = mask_tensor
    else:
        ple_outputs = PLE(name='ple_{}'.format(tower_name), num_tasks=1, num_shared_experts=4,
                          num_specific_experts=4, expert_dnn_units=(32,), gate_dnn_units=(), expert_dnn_params=dict(),
                          gate_dnn_params=dict())(weight_concat)
        outputs = DNN((output_dim,), output_activation='linear', l2_reg=0, dropout_rate=0,
                      name='{}_emb'.format(tower_name))(ple_outputs[0])

    # 在线需要通过tensor name获取output，将output的tensor name重新命名一下
    outputs = {'emb': tf.identity(outputs, name="{}_emb_output".format(tower_name))}

    model = tn.model.Model(inputs=sparse_emb_inputs, outputs=outputs, name="sub_model_" + tower_name)

    return model


def create_shallow_tower(deep_shapes):
    deep_inputs = [tn.layers.Input(name="shallow_{}".format(feature.feature_name),
                                   feature=feature, dtype="float32", shape=shape[1:])
                   for feature, shape in deep_shapes]

    deep = tf.keras.layers.Concatenate(name='shallow_deep_concact', axis=-1)(deep_inputs)

    for i, unit in enumerate([32]):
        deep = tf.keras.layers.Dense(unit, activation='relu', name='shallow_dnn_{}'.format(i))(deep)

    logit = tf.keras.layers.Dense(1, activation=None, name='logit_shallow')(deep)
    output = tf.sigmoid(logit)
    # 在线需要通过tensor name获取output，将output的tensor name重新命名一下
    final_output = tf.identity(output, name="final_output")
    logit = tf.identity(logit, name="logit_rename")
    outputs = {'logit': logit, 'final_output': final_output}
    return tn.model.Model(inputs=deep_inputs, outputs=outputs, name="shallow_model")


def fetch_embeddings(inputs):
    columns = {}
    for fea_id, _ in inputs.items():
        columns[fea_id] = tn.feature_column.category_column(key=fea_id, bucket_size=25600)

    embedding_columns = []
    for fea_id in C.USER_FEATURE_IDS:
        feature_column = tf.feature_column.embedding_column(columns[fea_id], dimension=C.USER_OUTPUT_DIM,
                                                            combiner='mean')
        embedding_columns.append(feature_column)

    for fea_id in C.ITEM_FEATURE_IDS:
        feature_column = tf.feature_column.embedding_column(columns[fea_id], dimension=C.ITEM_OUTPUT_DIM,
                                                            combiner='mean')
        embedding_columns.append(feature_column)

    # sparse_opt = tn.core.AdaGrad(learning_rate=0.01, initial_g2sum=0.1, initial_scale=0.1)
    sparse_opt = tn.core.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    sparse_embs = tn.layers.EmbeddingFeatures(embedding_columns, sparse_opt, name="sparse_emb_input")(inputs)

    feature_embeddings = {
        "user": {fea_id: sparse_embs[fea_id] for fea_id in C.USER_FEATURE_IDS},
        "item": {fea_id: sparse_embs[fea_id] for fea_id in C.ITEM_FEATURE_IDS},
        "teacher": {fea_id: sparse_embs[fea_id] for fea_id in C.USER_FEATURE_IDS + C.ITEM_FEATURE_IDS},
    }

    return feature_embeddings


def DSSM():
    feature_slots = {slot_id: tn.feature_column.FeatureSlot(slot_id) for slot_id in C.ALL_FEATURE_SLOT}

    sparse_features = {
        fea_id: tn.feature_column.Feature(feature_id=fea_id, feature_slot=feature_slots[slot_id], sparse=True)
        for fea_id, slot_id in C.ALL_FEATURE_ID_2_SLOT.items()}

    inputs = {fea_id: tn.layers.Input(name=fea_id, feature=feature,
                                      shape=(None,), dtype="int64", sparse=True)
              for fea_id, feature in sparse_features.items()}

    feature_embeddings = fetch_embeddings(inputs)
    dense_slot = tn.feature_column.FeatureSlot('4575')
    dense_feature = tn.feature_column.Feature(feature_id='4575', feature_slot=dense_slot, sparse=False)
    inputs['4575'] = tn.layers.Input(name='dense_weight_4575', feature=dense_feature, shape=(1,), dtype="float32",
                                     sparse=False)

    emb_input_shapes = [(sparse_features[fea_id], emb.shape) for fea_id, emb in feature_embeddings['user'].items()]
    user_tower_model = create_tower(emb_input_shapes, tower_name='user', output_dim=C.USER_OUTPUT_DIM,
                                    mask_tensor=inputs['4575'])

    emb_input_shapes = [(sparse_features[fea_id], emb.shape) for fea_id, emb in feature_embeddings['item'].items()]
    item_tower_model = create_tower(emb_input_shapes, tower_name='item', output_dim=C.ITEM_OUTPUT_DIM)

    feature_embeddings['user']['4575'] = inputs['4575']
    user_tower_outputs = user_tower_model(feature_embeddings['user'])
    item_tower_outputs = item_tower_model(feature_embeddings['item'])

    emb_input_shapes = [(sparse_features[fea_id], emb.shape) for fea_id, emb in feature_embeddings['teacher'].items()]
    teacher_model = create_tower_teacher(emb_input_shapes, tower_name='teacher')
    teacher_outputs = teacher_model(feature_embeddings['teacher'])
    teacher_logit = teacher_outputs['logit']
    teacher_output = tf.sigmoid(teacher_logit)

    user_sub_model_output_name = user_tower_model.output['emb'].name.replace(":0", "")
    item_sub_model_output_name = item_tower_model.output['emb'].name.replace(":0", "")
    shallow_user_input_feature = tn.feature_column.Feature(feature_name=user_sub_model_output_name, sparse=False)
    shallow_item_input_feature = tn.feature_column.Feature(feature_name=item_sub_model_output_name, sparse=False)
    shallow_tower_input_shapes = [(shallow_user_input_feature, user_tower_outputs['emb'].shape),
                                  (shallow_item_input_feature, item_tower_outputs['emb'].shape)]
    shallow_tower = create_shallow_tower(shallow_tower_input_shapes)

    student_outputs = shallow_tower([user_tower_outputs['emb'], item_tower_outputs['emb']])

    student_logit = student_outputs['logit']

    student_output = student_outputs['final_output']

    teacher_logit_no_grad = tf.stop_gradient(teacher_logit)
    kd_loss = KDLoss(name='distill')(student_logit, teacher_logit_no_grad)

    outputs = {
        'student': student_output,
        'teacher': teacher_output,
        'distill': kd_loss,
    }

    train_model = tn.model.Model(inputs=inputs, outputs=outputs, name="train_model")
    predict_output = {
        **outputs,
        # "sub_model_item" : item_tower_outputs,
        # "sub_model_user" : user_tower_outputs,
    }

    predict_model = tn.model.Model(inputs=inputs, outputs=predict_output, name="predict_model")

    return {
        "train": train_model,
        "predict": predict_model
    }


def mse_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_true = y_true * 1.0 / 1000
    one = tf.ones_like(y_true)
    wt_log = tf.math.log(tf.add(y_true, one))
    upper = tf.constant(5.3, dtype=tf.float32)
    y_true_clip = tf.where(tf.greater(wt_log, upper), upper, wt_log)
    loss = tf.reduce_mean(tf.square(y_true_clip - y_pred))
    return loss


def y_pred_loss(y_true, y_pred):
    loss = tf.reduce_mean(y_pred)
    return loss


def create_model():
    dssm_model = DSSM()

    dense_opt = tn.core.Adam(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    losses = {
        'student': tf.keras.losses.BinaryCrossentropy(),
        'teacher': tf.keras.losses.BinaryCrossentropy(),
        'distill': y_pred_loss,
    }
    metrics = {
        'student': [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(), tn.metric.CTR(), tn.metric.COPC()],
        'teacher': [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(), tn.metric.CTR(), tn.metric.COPC()],
        'distill': [y_pred_loss],
    }
    dssm_model['train'].compile(optimizer=tn.optimizer.Optimizer(dense_opt), loss=losses, metrics=metrics)

    return dssm_model


def test():
    model = create_model()
    model.summary()


if __name__ == "__main__":
    test()
