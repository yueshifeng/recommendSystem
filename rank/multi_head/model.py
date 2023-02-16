# -*- coding: utf-8 -*-
import logging
import os
from datetime import datetime

import tensorflow as tf
import tensornet as tn

from src.util.util import read_dataset, trained_delta_days, dump_predict
from interact_multihead_autoint_alllabel.model.config import Config as C

from interact_multihead_autoint_alllabel.model.mutiDnnAutointOrigin import AUTOINT

def test(a):
    print(a)


def cross_entropy(y_true, y_pred, a = 1):
    y_true = tf.cast(y_true, tf.float32)
    loss = - y_true * tf.math.log(y_pred + 1e-6) - (a - y_true) * tf.math.log(1.0 - y_pred + 1e-6)
    loss = tf.math.reduce_sum(loss, axis=-1, keepdims=True)
    return loss
# def cross_entropy2(y_true, y_pred, a = 1):
#     y_ori = y_true
#     y_true = tf.cast(y_true, tf.float32)
#     ones = tf.ones_like(y_ori)
#     zeros = tf.zeros_like(y_ori)
#     loss_mask = tf.where(tf.math.greater_equal(y_ori,zeros),ones,zeros)
#     loss_mask = tf.cast(loss_mask, tf.float32)
#     loss = - y_true * tf.math.log(y_pred + 1e-6) - (a - y_true) * tf.math.log(1.0 - y_pred + 1e-6)
#     loss = loss * loss_mask
#     loss = tf.math.reduce_sum(loss, axis=-1, keepdims=True)
#     return loss


# 函数签名是固定的。
# 即必须接受 model_param 这个 dict
# 返回的 model_result 包含两个参数
# 返回第一个 是 model，是 tn.model.Model，表示整个网络结构
# 返回第二个 是 sub_model，是 tn.model.Model，表示 dense 部分的网络结构，也就是除去 sparse 部分网络结构
# 如果模型是多个头，需要自己写 cross_entropy
def create_model_func(model_param):
    training = model_param.get('training', True)
    logging.info('training={}'.format(training))
    print('model_param={}'.format(model_param))
    logging.info('model_param={}'.format(model_param))
    print('C.LINEAR_SLOTS={}'.format(C.LINEAR_SLOTS))
    print('C.DENSE_SLOTS={}'.format(C.DENSE_SLOTS))
    model_result = AUTOINT(C.LINEAR_SLOTS, C.DENSE_SLOTS, training)
    #model_result = WDL(C.SPARSE_SLOTS, training)

    dense_opt = tn.core.Adam(learning_rate=0.00001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    model_result.model.compile(optimizer=tn.optimizer.Optimizer(dense_opt),
                               loss=cross_entropy,
                               metrics=['acc', tf.keras.metrics.AUC(), tn.metric.COPC()]
                               # 当 parse_input_func 返回第三个 sample weight 参数时,
                               # 需要设置 weighted_metrics
                               #,weighted_metrics=[tf.keras.metrics.AUC()]
                               ),

    return model_result
