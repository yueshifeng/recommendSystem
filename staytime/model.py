# -*- coding: utf-8 -*-
import logging
import os
from datetime import datetime

import tensorflow as tf
import tensornet as tn
from video_id_rank_staytime_mtl_ppnet_v7.model.custom_metrics import CustomAccuracy, CustomMAE, CustomMSE

from video_id_rank_staytime_mtl_ppnet_v7.model.config import Config as C
from video_id_rank_staytime_mtl_ppnet_v7.model.VideoDNN import mtl_net
from tensorflow.python.keras import backend as K

"""
loss = a*softmax_ce_loss + b*l2_loss
y_true/y_pred: (batch_size, label_nums+1)
"""


def custom_kl_loss(y_true, y_pred):
    print("loss y_true shape: ", y_true.shape)
    print("loss y_pred shape: ", y_pred.shape)
    y_true_1 = y_true[:, 0:C.multiclass_num]
    y_pred_1 = y_pred[:, 0:C.multiclass_num]

    y_true_1 = tf.cast(y_true_1, y_pred_1.dtype)
    y_true_1 = K.clip(y_true_1, K.epsilon(), 1)
    y_pred_1 = K.clip(y_pred_1, K.epsilon(), 1)
    loss = tf.math.reduce_sum(y_true_1 * tf.math.log(y_true_1 / y_pred_1), axis=-1)
    return loss


def cross_entropy(y_true, y_pred, a=1):
    y_true = tf.cast(y_true, tf.float32)
    loss = - y_true * tf.math.log(y_pred + 1e-6) - (a - y_true) * tf.math.log(1.0 - y_pred + 1e-6)
    return loss


def mse_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_true_clip = tf.where(y_true > 2.0, 2.0, y_true)
    loss = tf.reduce_mean(tf.square(y_true_clip - y_pred))
    tf.summary.scalar("train_mse_loss", loss)
    return loss


'''
 ' Huber loss.
 ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
 ' https://en.wikipedia.org/wiki/Huber_loss
'''


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.abs(error) < clip_delta

    squared_loss = 0.5 * tf.square(error)
    linear_loss = clip_delta * (tf.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)


# 函数签名是固定的。
# 返回的 model_result 包含两个参数
# 返回第一个 是 model，是 tn.model.Model，表示整个网络结构
# 返回第二个 是 sub_model，是 tn.model.Model，表示 dense 部分的网络结构，也就是除去 sparse 部分网络结构
# 如果模型是多个头，需要自己写 cross_entropy
def create_model_func():
    print('C.SLOTS={}'.format(C.SLOTS))
    print('C.SEQ_SLOTS={}'.format(C.SEQ_SLOTS))
    models = mtl_net(C.SLOTS, C.SEQ_SLOTS, 50, dnn_hidden_units=(256, 128))
    dense_opt = tn.core.Adam(learning_rate=0.0005, beta1=0.9, beta2=0.999, epsilon=1e-8)

    losses = {
        "video_id_rank_staytime_mtl_ppnet_v7_staytime": custom_kl_loss,
        "video_id_rank_staytime_mtl_ppnet_v7_shortplay": cross_entropy,
        "video_id_rank_staytime_mtl_ppnet_v7_longplay": cross_entropy}

    metrics = {
        'video_id_rank_staytime_mtl_ppnet_v7_staytime': [CustomAccuracy(), CustomMAE(), CustomMSE()],
        'video_id_rank_staytime_mtl_ppnet_v7_shortplay': [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
        'video_id_rank_staytime_mtl_ppnet_v7_longplay': [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
    }

    loss_weights = {"video_id_rank_staytime_mtl_ppnet_v7_staytime": 2,
                    "video_id_rank_staytime_mtl_ppnet_v7_shortplay": 2,
                    "video_id_rank_staytime_mtl_ppnet_v7_longplay": 1}

    models["train"].compile(optimizer=tn.optimizer.Optimizer(dense_opt), loss=losses, metrics=metrics,
                            loss_weights=loss_weights)

    return models
