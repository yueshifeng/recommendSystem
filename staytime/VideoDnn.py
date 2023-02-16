# -*- coding: utf-8 -*-
import logging
import tensorflow as tf
import tensornet as tn

from video_id_rank_staytime_mtl_ppnet_v7.model.config import Config as C
from tensorflow.python.keras.layers import Lambda
from video_id_rank_staytime_mtl_ppnet_v7.model.layers import DeepCrossLayer, DIN


def ffm_block(slot_dict, ffm_slots):
    ffm = []
    ffm_dim = 0
    for x_list, y_list, dim in ffm_slots:
        for x in x_list:
            for y in y_list:
                x_emb = slot_dict[x]
                y_emb = slot_dict[y]
                x_emb = tf.keras.layers.Dense(dim, activation=None, name='ffm_x_%s_%s_%d' % (x,y,dim))(x_emb)
                y_emb = tf.keras.layers.Dense(dim, activation=None, name='ffm_y_%s_%s_%d' % (x,y,dim))(y_emb)
                ffm.append(tf.multiply(x_emb, y_emb))
                ffm_dim += dim
    ffm = tf.keras.layers.Concatenate(name='concat_ffm', axis=-1)(ffm)
    logging.info("ffm_dim={}".format(ffm_dim))
    return ffm

def create_moe_sub_model(sparse_embs_input_shapes, seq_input_shapes, deep_hidden_units):

    logging.info("sparse_embs_input_shapes={}".format(sparse_embs_input_shapes))
    logging.info("seq_input_shapes={}".format(seq_input_shapes))

    USER_SLOTS = ['1568', '1589', '2039', '1570']
    ITEM_SLOTS = ['1591', '1593', '1737', '1614']
    BIAS_SLOTS = ['3051', '1570', '2039', '2544', '1568', '3376', '3365', '3369', '2597', '1737', '1593', '1591',
                  '1589', '1614']
    emb_inputs = []
    general_emb_dict = {}
    general_inputs = []
    bias_inputs = []
    for feature, shape in sparse_embs_input_shapes:
        tmp_input = tn.layers.Input(name=str(feature.feature_id), feature=feature,
                                    shape=shape[1:], dtype="float32")
        feature_id = str(feature.feature_id)
        emb_inputs.append(tmp_input)
        if feature_id in BIAS_SLOTS:
            bias_inputs.append(tmp_input[:, 16:])
        general_inputs.append(tmp_input[:, 0:16])
        general_emb_dict[feature_id] = tmp_input[:, 0:16]
    logging.info("general_inputs={}".format(general_inputs))
    logging.info("bias_inputs={}".format(bias_inputs))

    # din序列建模
    query_emb_videoid = general_emb_dict['1591']
    query_emb_authorid = general_emb_dict['1593']
    query_emb_l1cate = general_emb_dict['1737']
    seq_dict = {}
    seq_emb_inputs = []
    din_embs = []
    for feature, seq_tensor_shape, seq_mask_shape in seq_input_shapes:
        seq_emb_input = tn.layers.Input(name="seq_emb_{}".format(feature.feature_id),
                                        feature=feature, dtype="float32", shape=seq_tensor_shape[1:],
                                        feature_type="sequence")
        seq_mask_input = tn.layers.Input(name="seq_mask_{}".format(feature.feature_id),
                                         feature=feature, dtype="bool", shape=seq_mask_shape[1:],
                                         feature_type="sequence_mask")
        seq_emb_inputs.append(seq_emb_input)
        seq_emb_inputs.append(seq_mask_input)
        seq_dict[str(feature.feature_id)] = (seq_emb_input[:,:,0:16], seq_mask_input)
    for fea_id, (seq_emb_input, seq_mask_input) in seq_dict.items():
        if fea_id == '2125':
            query_emb = query_emb_videoid
        elif fea_id == '2128':
            query_emb = query_emb_authorid
        else:
            query_emb = query_emb_l1cate
        din_emb = DIN(name='din_{}'.format(fea_id))(query_emb, seq_emb_input, seq_mask_input)
        din_embs.append(din_emb)
    logging.info("DIN output is:{}".format(din_embs))

    # SENet reweight fea
    general_inputs_reweight = []
    senet_unit1 = len(general_inputs) / 4
    senet_unit2 = len(general_inputs)
    inputs_squeeze = tf.keras.layers.Concatenate(name='emb_inputs_squeeze_concacted', axis=-1)(general_inputs)
    logging.info('inputs_squeeze concat={}'.format(inputs_squeeze))
    inputs_squeeze_nograd = tf.stop_gradient(inputs_squeeze)
    senet_output1 = tf.keras.layers.Dense(senet_unit1, activation='relu', name='senet_squeeze_layer1')(
        inputs_squeeze_nograd)
    logging.info('senet_squeeze_layer1={}'.format(senet_output1))
    senet_output2 = 2 * tf.keras.layers.Dense(senet_unit2, activation='sigmoid', name='senet_extract_layer2')(
        senet_output1)
    logging.info('senet_extract_layer2={}'.format(senet_output2))
    senet_split_outputs = tf.split(senet_output2, senet_unit2, axis=1)
    logging.info('senet_split_outputs={}'.format(senet_split_outputs))
    for emb_input, senet_split_output in zip(general_inputs, senet_split_outputs):
        general_inputs_reweight.append(tf.keras.layers.multiply([emb_input, senet_split_output]))

    # multiplay fea
    emb_multiply_user_inputs = [general_emb_dict[feature_id] for feature_id in USER_SLOTS]
    emb_multiply_item_inputs = [general_emb_dict[feature_id] for feature_id in ITEM_SLOTS]
    multiply_user_inputs = tf.keras.layers.Concatenate(name='multiply_user_concated', axis=-1)(emb_multiply_user_inputs)
    multiply_item_inputs = tf.keras.layers.Concatenate(name='multiply_item_concated', axis=-1)(emb_multiply_item_inputs)
    multiply_result = tf.keras.layers.multiply([multiply_user_inputs, multiply_item_inputs])
    multiply_result = tf.keras.layers.ReLU()(multiply_result)
    logging.info('multiply_result={}'.format(multiply_result))

    # FM cross fea
    sum_embs = tf.reduce_sum(general_inputs_reweight, axis=0)
    sum_square_embs = sum_embs * sum_embs
    second_order_square_emb = [item * item for item in general_inputs_reweight]
    second_order_square_emb_sum = tf.reduce_sum(second_order_square_emb, axis=0)
    cross_term = sum_square_embs - second_order_square_emb_sum
    print("cross_term: ", cross_term)
    fm_logit = 0.5 * tf.math.reduce_sum(cross_term, axis=-1, keepdims=True)
    print("fm_logit: ", fm_logit)

    # FFM cross fea
    ffm_slots = [[['1568', '1589', '2039', '1570'],['1591', '1593', '1737', '1614'],8]]
    ffm_cross_term = ffm_block(general_emb_dict, ffm_slots)
    logging.info("ffm_cross_term={}".format(ffm_cross_term))

    concated_input = tf.keras.layers.Concatenate(name='concacted', axis=-1)(
        general_inputs_reweight + [cross_term] + [multiply_result] + [ffm_cross_term] + din_embs)
    logging.info("concated_input={}".format(concated_input))

    gate_input = tf.keras.layers.Concatenate(name='gate_concacted', axis=-1)(bias_inputs)
    logging.info("gate_input={}".format(gate_input))

    # build expert layer, include ppnet
    expert_outs = []
    for i in range(C.num_experts):
        deep = concated_input
        logging.info("gate_input_{}={}".format(i, gate_input))

        for j, unit in enumerate(deep_hidden_units):
            # build gate layers
            gate_output = tf.keras.layers.Dense(unit, activation='relu', name='gate_{}_{}_1'.format(i, j))(gate_input)
            gate_output = tf.keras.layers.Dense(unit, activation='sigmoid', name='gate_{}_{}_2'.format(i, j))(
                gate_output)
            gate_output = tf.multiply(gate_output, 2)
            # gate_output = tf.keras.layers.Lambda(print_tensor)(gate_output)
            logging.info("gate_output_{}_{}={}".format(i, j, gate_output))

            # multiply dnn output and gate
            deep = tf.keras.layers.Dense(unit, activation='relu', name='expert_output_{}_{}'.format(i, j))(deep)
            deep = tf.multiply(gate_output, deep)
            logging.info("expert_output_{}_{}={}".format(i, j, deep))
        expert_outs.append(deep)

    expert_concat = Lambda(lambda x: tf.stack(x, axis=1))(expert_outs)  # None,num_experts,dim
    print("expert_concat: ", expert_concat.get_shape())

    mmoe_outs = []
    for i in range(C.num_tasks):  # one mmoe layer: nums_tasks = num_gates
        # build gate layers
        gate_out = concated_input
        for j, unit in enumerate([64, 32]):
            gate_out = tf.keras.layers.Dense(unit, activation='relu', name='gate_{}_{}'.format(i, j))(gate_out)
        gate_out = tf.keras.layers.Dense(C.num_experts, activation='softmax', name='gate_output_{}'.format(i))(gate_out)
        gate_out = Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)
        # gate multiply the expert
        gate_mul_expert = Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis=1, keepdims=False),
                                 name='gate_mul_expert_' + C.task_names[i])([expert_concat, gate_out])
        mmoe_outs.append(gate_mul_expert)

    # staytime
    cross_feature = DeepCrossLayer(num_layer=3)(concated_input)
    mmoe_ext_out = tf.keras.layers.Concatenate(name='stack', axis=-1)([mmoe_outs[0], cross_feature])
    staytime_output = tf.keras.layers.Dense(C.multiclass_num, activation=None, name='staytime_output')(mmoe_ext_out)
    staytime_output = tf.nn.softmax(staytime_output)
    print("y_pred: ", staytime_output)
    # 添加预估的期望时长
    wt_bins = tf.constant(C.bin_list, dtype=tf.float32)
    wt_bins = tf.reshape(wt_bins, [C.multiclass_num, 1])
    # for 线上infer
    staytime_pred = tf.matmul(staytime_output, wt_bins, name=C.task_names[0])  # (batch_size, 1)
    staytime_pred = tf.where(staytime_pred < 0.0, 0.0, staytime_pred)
    # for loss计算
    final_y_pred = tf.concat([staytime_output, staytime_pred], -1)  # (batch_size, label_nums+1)

    # shortplay
    shorplay_deep_logit = tf.keras.layers.Dense(1, activation='relu', name='tower_deep_{}'.format(C.task_names[1]))(
        mmoe_outs[1])
    shorplay_tower_input = tf.concat([fm_logit, shorplay_deep_logit], axis=1)
    shortplay_pred = tf.keras.layers.Dense(1, activation='sigmoid', name=C.task_names[1])(shorplay_tower_input)

    # longplay
    longplay_deep_logit = tf.keras.layers.Dense(1, activation='relu', name='tower_deep_{}'.format(C.task_names[2]))(
        mmoe_outs[2])
    longplay_tower_input = tf.concat([fm_logit, longplay_deep_logit], axis=1)
    longplay_pred = tf.keras.layers.Dense(1, activation='sigmoid', name=C.task_names[2])(longplay_tower_input)

    # 在线需要通过tensor name获取output，将output的tensor name重新命名一下
    task_outs = {
        "video_id_rank_staytime_mtl_ppnet_v7_staytime": tf.identity(final_y_pred,
                                                                    name="video_id_rank_staytime_mtl_ppnet_v7_staytime_l"),
        "video_id_rank_staytime_mtl_ppnet_v7_shortplay": tf.identity(shortplay_pred,
                                                                     name="video_id_rank_staytime_mtl_ppnet_v7_shortplay_l"),
        "video_id_rank_staytime_mtl_ppnet_v7_longplay": tf.identity(longplay_pred,
                                                                    name="video_id_rank_staytime_mtl_ppnet_v7_longplay_l"),
    }

    task_preds = {
        "video_id_rank_staytime_mtl_ppnet_v7_staytime": tf.identity(staytime_pred,
                                                                    name="video_id_rank_staytime_mtl_ppnet_v7_staytime"),
        "video_id_rank_staytime_mtl_ppnet_v7_shortplay": tf.identity(shortplay_pred,
                                                                     name="video_id_rank_staytime_mtl_ppnet_v7_shortplay"),
        "video_id_rank_staytime_mtl_ppnet_v7_longplay": tf.identity(longplay_pred,
                                                                    name="video_id_rank_staytime_mtl_ppnet_v7_longplay"),
    }

    return {
        "sub_model_train": tn.model.Model(inputs=emb_inputs + seq_emb_inputs, outputs=task_outs, name="sub_model1"),
        "sub_model_predict": tn.model.Model(inputs=emb_inputs + seq_emb_inputs, outputs=task_preds, name="sub_model2")
    }

def fetch_embeddings_seq(inputs, seq_fea_ids, seq_max_len):
    columns = {}
    for fea_id in inputs.keys():
        columns[fea_id] = tn.feature_column.category_column(key=fea_id, bucket_size=81920)

    embedding_columns = []
    for fea_id in inputs.keys():
        feature_column = tn.feature_column.embedding_column(name="emb_col_{}".format(fea_id),
                            categorical_column=columns[fea_id], dimension=32, combiner='mean')
        embedding_columns.append(feature_column)

    for fea_id in seq_fea_ids:
        feature_column = tn.feature_column.embedding_column(name="emb_col_seq_{}".format(fea_id),
                            categorical_column=columns[fea_id], dimension=32, combiner=None, seq_max_len=seq_max_len)
        embedding_columns.append(feature_column)

    sparse_opt = tn.core.AdaGrad(learning_rate=0.005, initial_g2sum=0.1, initial_scale=0.1)

    # EmbeddingFeature返回按embedding_column作为key的的字典
    # 如果embedding_column的combiner为None，那么key的value返回两个tensor，一个为3D的embedding，一个为mask tensor，用元组表示
    sparse_embs = tn.layers.EmbeddingFeatures(embedding_columns, sparse_opt, name="sparse_emb_input")(inputs)

    feature_embeddings = {
        "no_seq" : {fea_id : sparse_embs["emb_col_{}".format(fea_id)] for fea_id in inputs.keys()},
        "seq" : {fea_id : sparse_embs["emb_col_seq_{}".format(fea_id)] for fea_id in seq_fea_ids},
    }

    return feature_embeddings


def fetch_embeddings(inputs):
    columns = {}
    for fea_id in inputs.keys():
        columns[fea_id] = tn.feature_column.category_column(key=fea_id, bucket_size=81920)

    embedding_columns = []
    for fea_id in inputs.keys():
        feature_column = tn.feature_column.embedding_column(name="emb_col_{}".format(fea_id),
                                                            categorical_column=columns[fea_id], dimension=32,
                                                            combiner='mean')
        embedding_columns.append(feature_column)

    sparse_opt = tn.core.AdaGrad(learning_rate=0.005, initial_g2sum=0.1, initial_scale=0.1)

    sparse_embs = tn.layers.EmbeddingFeatures(embedding_columns, sparse_opt, name="sparse_emb_input")(inputs)

    return {fea_id: sparse_embs["emb_col_{}".format(fea_id)] for fea_id in inputs.keys()}


def mtl_net(slots, seq_slots, seq_max_len, dnn_hidden_units=(64, 32)):
    logging.info('dnn_hidden_units={}'.format(dnn_hidden_units))
    feature_slots = {slot_id: tn.feature_column.FeatureSlot(slot_id) for slot_id in slots}
    sparse_features = {
        slot_id: tn.feature_column.Feature(feature_id=slot_id, feature_slot=feature_slots[slot_id], sparse=True)
        for slot_id in slots}

    inputs = {fea_id: tn.layers.Input(name=fea_id, feature=feature,
                                      shape=(None,), dtype="int64", sparse=True)
              for fea_id, feature in sparse_features.items()}

    embs = fetch_embeddings_seq(inputs, seq_slots, seq_max_len)
    sparse_embs = embs['no_seq']
    sparse_embs_seq = embs['seq']

    seq_input_shapes = []
    sparse_embs_input_shapes = [(sparse_features[fea_id], sparse_embs[fea_id].shape) for fea_id in sorted(sparse_embs.keys())]
    for fea_id in sorted(sparse_embs_seq.keys()):
        embedding_3d, seq_mask = sparse_embs_seq[fea_id]
        seq_input_shapes.append((sparse_features[fea_id], embedding_3d.shape, seq_mask.shape))

    models = create_moe_sub_model(sparse_embs_input_shapes, seq_input_shapes, dnn_hidden_units)

    # todo sub_model的input顺序必须与子图里面定义的一致
    sub_model_input_emb = [sparse_embs[fea_id] for fea_id in sorted(sparse_embs.keys())]
    for fea_id in sorted(sparse_embs_seq.keys()):
        embedding_3d, seq_mask = sparse_embs_seq[fea_id]
        sub_model_input_emb.append(embedding_3d)
        sub_model_input_emb.append(seq_mask)

    train_outputs = models['sub_model_train'](sub_model_input_emb)
    predict_outputs = models['sub_model_predict'](sub_model_input_emb)

    return {
        "train": tn.model.Model(inputs=inputs, outputs=train_outputs, name="full_model"),
        "predict": tn.model.Model(inputs=inputs, outputs=predict_outputs, name="full_model"),
    }
