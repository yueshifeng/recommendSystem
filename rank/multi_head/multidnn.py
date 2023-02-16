# -*- coding: utf-8 -*-
import logging
import tensorflow as tf
import sys
import tensornet as tn
from src.pipeline.model_result import ModelResult
from src.model.feature_column import tn_category_columns_builder, embedding_columns_builder, create_emb_model
from src.pipeline.multi_sparse_table import MultiSparseTableInfo
import src.util.tools as tools
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Embedding, BatchNormalization, Activation, Flatten
from interact_multihead_autoint_alllabel.model.interacting_layer import InteractingLayer


def create_autoint_sub_model(slot_zip_user_embs, dense_inputs_map, deep_hidden_units, training):
    emb_input_shapes_with_slot = [(slot, emb.shape) for slot, emb in slot_zip_user_embs]
    #emb_inputs = [tf.keras.layers.Input(name="emb_{}".format(slot), dtype="float32", shape=shape[1:])
    #                   for slot, shape in emb_input_shapes_with_slot]
    emb_inputs = []
    logging.info("enter here===================")
    for slot, shape in emb_input_shapes_with_slot:
        sparse_layer = tf.keras.layers.Input(name="emb_{}".format(slot), dtype="float32", shape=shape[1:])
        #sparse_exp = sparse_layer[:,tf.newaxis, :]
        emb_inputs.append(sparse_layer)
    
    emb_3d_inputs = []
    for emb in emb_inputs:
        emb_3d_inputs.append(emb[:, tf.newaxis, :])

    # dense 数据输入。命名必须是 dense_XXX_slot
    dense_inputs = [tf.keras.layers.Input(name="dense_weight_{}".format(k), dtype="float32", shape=v.shape[1:])
                       for k, v in dense_inputs_map.items()]

    # 需要添加映射关系。 sparse table name 到 dense 模型的 input tensor prefix
    # sparse table name 是 columns_group 的 key
    MultiSparseTableInfo.add_sparse_table_name_mapping_to_input_tensor_prefix('linear', 'linear_emb_')
    #emb_inputs = [emb[:, 0:16] for emb in emb_total_inputs]
    #bias_inputs = [emb[:, 16:] for emb in emb_total_inputs]
    ''' 
    square_of_sum = tf.square(tf.math.reduce_sum(emb_inputs, axis=0, keepdims = False))
    logging.info('square_of_sum:', square_of_sum)
    square_emb_inputs = [emb_input * emb_input for emb_input in emb_inputs]
    sum_of_square = tf.math.reduce_sum(square_emb_inputs, axis=0, keepdims=False)
    cross_term = square_of_sum - sum_of_square
    logging.info('cross_term:', cross_term.get_shape())
    fm = 0.5 * tf.math.reduce_sum(cross_term, axis=-1, keepdims=True)
    logging.info('fm:', fm.get_shape())
    '''
    #all_inputs = tf.keras.layers.Concatenate(name='concacted', axis=-1)(emb_inputs + cross_term + dense_inputs)
    logging.info('before concat={}'.format(emb_inputs))
    all_inputs = tf.keras.layers.Concatenate(name='concacted', axis=1)(emb_3d_inputs)
    logging.info('all_inputs111={}'.format(all_inputs))
    #autoint_inputs = Flatten(all_inputs)
    #autoint_inputs = tf.keras.layers.Concatenate(name='concacted', axis=-1)(emb_inputs + dense_inputs)
    autoint_outputs = InteractingLayer(layer_num=1, unit_num=8, head_num=2, use_dropout=True, dropout_rate=0.2, use_res=True)(all_inputs)
    logging.info('auto int111={}'.format(autoint_outputs))
    autoint_outputs = Flatten()(autoint_outputs)
    logging.info('auto int={}'.format(autoint_outputs.get_shape()))
    
    #deep_result = tf.concat(emb_inputs, axis = 1)
    deep_result = Flatten()(all_inputs)
    #deep_result = tf.keras.layers.BatchNormalization(name="model_input_bn", trainable=training)(deep_result, training=training)
    for  i, unit in enumerate(deep_hidden_units):
        deep_result = tf.keras.layers.Dense(unit, activation='relu', name='dnn_{}'.format(i), kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00001, l2=0.00001))(deep_result)
        #deep_result = tf.keras.layers.BatchNormalization(name="model_input_bn_{}".format(i), trainable=training)(deep_result, training=training)

    logging.info('deep_result={}'.format(deep_result))
    '''
    lr_result = tf.reduce_sum(bias_inputs, axis=0, keepdims = False)
    print('lr_result:', lr_result.get_shape())
    '''
    #iresult = tf.concat([lr_result, fm, deep_result], axis = 1)
    result = tf.concat([deep_result, autoint_outputs], axis = 1)
    logging.info('result={}'.format(result))

    # -----begin mutihead------
    model_input_bn = result

    print("model_input_bn_dim: {}".format(model_input_bn.get_shape()))
    # -----experts block-----
    expert_num = 7
    expert_out_list = []
    for idx in range(expert_num+1):       
        expert_out = tf.keras.layers.Dense(32, activation="relu", name="expert_{}_fc1".format(idx),
                                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.001),
                                           kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(result)

        expert_out = tf.expand_dims(expert_out, 1)
        expert_out_list.append(expert_out)

    print("expert_out_list_len: ", len(expert_out_list))
    # (batch, expert_num, f_size=64)
    expert_out_list = tf.keras.layers.Concatenate(name="concat_expert_list", axis=1)(expert_out_list[0:7])

    print(expert_out_list.get_shape())
    print("expert_out_list_dim: {}".format(expert_out_list.get_shape()))

    gate_list = []
    #------gate block------
    num_label = 7
    for idx in range(num_label):
        gate = tf.keras.layers.Dense(expert_num, activation="softmax", name="gate_{}_fc2".format(idx),
                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.001),
                                     kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(model_input_bn)

        print ("gate_{}_dim: {}".format(idx, gate.get_shape()))
        # (batch, expert_num) -> (batch, expert_num, 1)
        gate = tf.expand_dims(gate, -1)
        print ("gate_{}_expanded_dim: {}".format(idx, gate.get_shape()))
       
        weighted_expert_output = expert_out_list * gate
        
        print ("weighted_expert_output_{}_dim: {}".format(idx, weighted_expert_output.get_shape()))
        weighted_expert_output = tf.reduce_sum(weighted_expert_output, 1)
        print ("weighted_expert_output_sum_{}_dim: {}".format(idx, weighted_expert_output.get_shape()))

        gate_list.append(weighted_expert_output)

    # like
    like = gate_list[0]
    # debug
    print("like ",like.get_shape())
    print(like)
    like_bn = like
    print("like_bn ",like_bn.get_shape())
    num_label = 1
    like_pred = tf.keras.layers.Dense(
        num_label, activation='sigmoid', name='like_pred')(like_bn)

    # click_comment
    click_comment = gate_list[1]
    # debug
    print("click_comment ",click_comment.get_shape())
    print(click_comment)
    click_comment_bn = click_comment
    print("click_comment_bn",click_comment_bn.get_shape())
    num_label = 1
    click_comment_pred = tf.keras.layers.Dense(
        num_label, activation='sigmoid', name='click_comment_pred')(click_comment_bn)

    # comment
    comment = gate_list[2]
    # debug
    print("comment ",comment.get_shape())
    print(comment)
    comment_bn = comment
    print("comment_bn",comment_bn.get_shape())
    num_label = 1
    comment_pred = tf.keras.layers.Dense(
        num_label, activation='sigmoid', name='comment_pred')(comment_bn)

    # share
    #share = gate_list[3]
    # debug
    #print("share ",share.get_shape())
    #print(share)
    #share_bn = share
    #print("share_bn",share_bn.get_shape())
    #num_label = 1
    #share_pred = tf.keras.layers.Dense(
    #    num_label, activation='sigmoid', name='share_pred')(share_bn)
    
    # click_sharing
    click_sharing = gate_list[3]
    # debug
    print("click_sharing ",click_sharing.get_shape())
    print(click_sharing)
    click_sharing_bn = click_sharing
    print("click_sharing_bn",click_sharing_bn.get_shape())
    num_label = 1
    click_sharing_pred = tf.keras.layers.Dense(
             num_label, activation='sigmoid', name='click_sharing_pred')(click_sharing_bn)
    
    # follow
    follow = gate_list[4]
    # debug
    print("follow ",follow.get_shape())
    print(follow)
    follow_bn = follow
    print("follow_bn",follow_bn.get_shape())
    num_label = 1
    follow_pred = tf.keras.layers.Dense(
                  num_label, activation='sigmoid', name='follow_pred')(follow_bn)

    # click_avatar
    click_avatar = gate_list[5]
    # debug
    print("click_avatar ",click_avatar.get_shape())
    print(click_avatar)
    click_avatar_bn = click_avatar
    print("click_avatar_bn",click_avatar_bn.get_shape())
    num_label = 1
    click_avatar_pred = tf.keras.layers.Dense(
                       num_label, activation='sigmoid', name='click_avatar_pred')(click_avatar_bn)

    # unlike
    unlike = gate_list[6]
    # debug
    print("unlike ",unlike.get_shape())
    print(unlike)
    unlike_bn = unlike
    print("unlike_bn",unlike_bn.get_shape())
    num_label = 1
    unlike_pred = tf.keras.layers.Dense(
                         num_label, activation='sigmoid', name='unlike_pred')(unlike_bn)

    y_pred = [like_pred,click_comment_pred,comment_pred,click_sharing_pred,follow_pred,click_avatar_pred,unlike_pred]

    # 在多label训练下, 这里需要设置各个label顺序以告知训练平台, 各个 output 的名字.
    from src.pipeline.multi_label import MultiLabelInfo
    MultiLabelInfo.label_list = ["like_pred", "click_comment_pred","comment_pred","click_sharing_pred","follow_pred","click_avatar_pred","unlike_pred"]

    return tn.model.Model(inputs=emb_inputs, outputs=y_pred, name="sub_model")

def AUTOINT(linear_features, dense_features, training, dnn_hidden_units=(32,16,)):
    features_set = set(linear_features)
    features = list(features_set)
    # features 可能每次的顺序都不一样，所以需要排序以保证一样
    features.sort()
    logging.info('dnn_hidden_units={}'.format(dnn_hidden_units))
    columns_group = {}
    tn_category_columns = tn_category_columns_builder(features)
    columns_group["linear"] = embedding_columns_builder(linear_features, tn_category_columns, 8, combiner='mean')
    sparse_inputs = {}
    for slot in features:
        sparse_inputs[slot] = tf.keras.layers.Input(name=slot, shape=(None,), dtype="int64", sparse=True)
    dense_inputs = {}
    for slot in dense_features:
        dense_inputs[slot] = tf.keras.layers.Input(name=slot, dtype="float32", shape=(1))

    # 自定义优化器。feature_drop_show 设置为 -1 表示不删除 feasign
    #sparse_opt = tn.core.AdaGrad(learning_rate=0.01
    #                             , initial_g2sum=0.1
    #                             , initial_scale=0.1
    #                             , feature_drop_show=-1)
    sparse_opt = tn.core.Adam(learning_rate=0.00005, beta1=0.9, beta2=0.999, epsilon=1e-8)

    emb_model = create_emb_model(features, columns_group, sparse_optimizer=sparse_opt)
    [linear_embs] = emb_model(sparse_inputs)
    slot_zip_linear_embs = zip(linear_features, linear_embs)
    #slot_zip_embs = zip(slots, embs[0])

    sub_model = create_autoint_sub_model(slot_zip_linear_embs, dense_inputs, dnn_hidden_units, training)

    output = sub_model(linear_embs + list(dense_inputs.values()))
    inputs = tools.merge_dict(sparse_inputs, dense_inputs)
    ret = ModelResult()

    ret.model = tn.model.Model(inputs=inputs, outputs=output, name="full_model")
    ret.sub_model = sub_model
    ret.model_predict = tn.model.Model(inputs=inputs, outputs=output, name="full_model", example_id_slot='extra_info')

    # 这个是 debug 用的，正式跑的时候可以不输出
    #ret.model_whit_input = tn.model.Model(inputs=inputs
    #                                      , outputs=output
    #                                      , name="full_model_whit_input"
    #                                      # example_id_slot 是指样本 id
    #                                      , example_id_slot='extra_info'
    #                                      , need_y=True)
    return ret
