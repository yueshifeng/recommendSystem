# -*- coding: utf-8 -*-
import logging
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensornet as tn

from rank_finish_ppnet_din.model.config import Config as C

from rank_finish_ppnet_din.model.VideoDNN import *
from tensorflow.python.keras import backend as K
embedding_dim = 16
def test(a):
    print(a)

def cross_entropy(y_true, y_pred, a = 1):
    y_true = tf.cast(y_true, tf.float32)
    loss = - y_true * tf.math.log(y_pred + 1e-6) - (a - y_true) * tf.math.log(1.0 - y_pred + 1e-6)
    #loss = tf.math.reduce_sum(loss, axis=-1, keepdims=True)
    loss = tf.math.reduce_mean(tf.math.reduce_sum(loss, axis=1), axis=0)
    return loss
    #default_graph = tf.compat.v1.get_default_graph()
    #reg_loss = tf.math.reduce_mean(tf.reduce_sum(tf.concat(default_graph.get_collection("regularization_losses"), axis=1) ,axis=1), axis=0)

# 函数签名是固定的。
# 即必须接受 model_param 这个 dict
# 返回的 model_result 包含两个参数
# 返回第一个 是 model，是 tn.model.Model，表示整个网络结构
# 返回第二个 是 sub_model，是 tn.model.Model，表示 dense 部分的网络结构，也就是除去 sparse 部分网络结构
# 如果模型是多个头，需要自己写 cross_entropy 
def create_model_func():
    logging.info('C.SLOTS={}'.format(C.SLOTS))
    print('C.SLOTS={}'.format(C.SLOTS))
    # model_result = VDNN(C.SLOTS)
    models = DEEPFM(C.SLOTS)
    #model_result = NFM(C.SLOTS)

    #dense_opt = tn.core.Adam(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    dense_opt = tn.core.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    losses = {
            'video_id_rank_finish_nb_lr_rongh_bundle': cross_entropy
        }
    metrics = {
            'video_id_rank_finish_nb_lr_rongh_bundle': ['acc', tf.keras.metrics.AUC()]
    }
    models["train"].compile(optimizer=tn.optimizer.Optimizer(dense_opt),
                        loss=losses,
                        metrics=metrics)

    return models


