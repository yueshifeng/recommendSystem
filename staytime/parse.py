# -*- coding: utf-8 -*-
import math
import os
from datetime import datetime

import tensorflow as tf
import tensornet as tn

from video_id_rank_staytime_mtl_ppnet_v7.model.config import Config as C


# 函数签名是固定的。
# 必须接受 example_proto 的对象
# 返回第一个参数 feature_dict 是特征列表的 dict
# 返回第二个参数 label 是一个 tensor
def parse_input_func(example_proto):
    fea_desc = {}
    fea_desc['extra_info'] = tf.io.FixedLenFeature([], tf.string, default_value="label")
    fea_desc['video_duration'] = tf.io.FixedLenFeature([], tf.int64)
    fea_desc['watch_duration'] = tf.io.FixedLenFeature([], tf.int64)

    for slot in set(C.SLOTS):
        fea_desc[slot] = tf.io.VarLenFeature(tf.int64)

    feature_dict = tf.io.parse_example(example_proto, fea_desc)
    wt = feature_dict.pop('watch_duration')
    feature_dict['example_id'] = feature_dict['extra_info']
    extra_info = feature_dict.pop('extra_info')

    # 短播label
    short_field = tf.constant(7000, dtype=tf.int64)
    pos = tf.ones_like(wt)
    neg = tf.zeros_like(wt)
    short_label = tf.where(tf.greater(wt, short_field), pos, neg)

    # 长播label
    long_field = tf.constant(18000, dtype=tf.int64)
    long_label = tf.where(tf.greater(wt, long_field), pos, neg)

    wt = tf.cast(wt, tf.float32)
    wt = tf.divide(wt, 1000.0)
    wt = tf.where(wt > 160.0, 160.0, wt)

    sample_cnt = tf.shape(wt)[0]
    bins = tf.constant([C.bin_list], dtype=tf.float32)
    bins = tf.repeat(bins, sample_cnt, axis=0)  # (batch_size, bins_dim)

    wt = tf.reshape(wt, [sample_cnt, 1])  # (batch_size,1)
    extra_info = tf.reshape(extra_info, [sample_cnt, 1])
    wt_ext = tf.repeat(wt, C.multiclass_num, axis=1)  # (batch_size, bins_dim)

    dist = tf.math.subtract(bins, wt_ext)  # (batch_size, bins_dim)
    absSquareDist = tf.math.square(tf.math.abs(dist))

    left = -19
    right = 180.5
    width = (right - left) / (C.multiclass_num - 1)
    sigma = 4
    div_num = tf.constant(math.sqrt(2 * math.pi) * sigma, dtype=tf.float32)
    label = tf.divide(tf.math.exp(tf.divide(absSquareDist, -2 * math.pow(sigma, 2))), div_num)
    label = tf.multiply(label, width)
    staytime_label = tf.concat([label, wt], -1)

    sample_weight = tf.where(tf.strings.regex_full_match(extra_info, ".*video_homepage_landing.*"), tf.ones_like(wt)*5, tf.ones_like(wt))

    y = {
        "video_id_rank_staytime_mtl_ppnet_v7_staytime" : staytime_label,
        "video_id_rank_staytime_mtl_ppnet_v7_shortplay" : short_label,
        "video_id_rank_staytime_mtl_ppnet_v7_longplay" :long_label,
    }
    return feature_dict, y, sample_weight

def dataset_reader(data_dir, dates, match_pattern, batch_size):
    ds_data_files = tn.data.list_files(
        data_dir, days=dates, match_pattern=match_pattern)

    dataset = ds_data_files.shard(
        num_shards=tn.core.shard_num(),
        index=tn.core.self_shard_id())

    dataset = dataset.interleave(lambda f: tf.data.TFRecordDataset(f, buffer_size=1024 * 100),
                                cycle_length=4, block_length=8,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(map_func=lambda example_proto: parse_input_func(example_proto), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #dataset = tn.data.BalanceDataset(dataset)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
