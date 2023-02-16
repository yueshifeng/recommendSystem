# -*- coding: utf-8 -*-
import logging
import tensorflow as tf
import tensornet as tn


def cross_entropy(y_true, y_pred, a = 1):
    y_true = tf.cast(y_true, tf.float32)
    loss = - y_true * tf.math.log(y_pred + 1e-6) - (a - y_true) * tf.math.log(1.0 - y_pred + 1e-6)
    #loss = tf.math.reduce_sum(loss, axis=-1, keepdims=True)
    loss = tf.math.reduce_mean(tf.math.reduce_sum(loss, axis=1), axis=0)
    return loss

class SingleSlot:
    def __init__(self, slot_id, emb_size, is_single):
        self.slot_id = slot_id
        self.intervals = list()
        self.last_start = -1
        self.last_end = -1
        self.total_emb_size = 0
        self.update_intervals(emb_size, is_single)
    def update_intervals(self, emb_size, is_single):
        self.last_start = self.last_end + 1
        self.last_end = self.last_start + emb_size - 1
        if is_single:
            self.intervals.append([self.last_start, self.last_end + 1])
        self.total_emb_size += emb_size

class BaseModel:
    def __init__(self, model_config):
        logging.info("enter input layer============")
        self.max_embedding_size = 0
        self.model_config = model_config
        # 解析sparse feature
        sparse_features = self.model_config['feature_slot']['sparse_feature']
        sparse_features_list = []
        sparse_slot_dict = {}
        bias_slot_dict = {}
        for k in sparse_features.keys():
            slot = sparse_features.get(k)['slot_id'][0]
            if slot in sparse_slot_dict:
                sparse_slot_dict[slot].update_intervals(sparse_features.get(k)['emb_size'], False if "bias" in sparse_features.get(k) else True)
            else:
                sparse_slot_dict[slot] = SingleSlot(slot, sparse_features.get(k)['emb_size'], False if "bias" in sparse_features.get(k) else True)
            if "bias" in sparse_features.get(k):
                if "bias_type" not in sparse_features.get(k):
                    raise Exception("bias_type could not be null")
                bias_type = sparse_features.get(k)["bias_type"]
                if slot in bias_slot_dict:
                    bias_slot_dict[slot][bias_type] = [sparse_slot_dict[slot].last_start, sparse_slot_dict[slot].last_end + 1]
                else:
                    bias_slot_dict[slot] = dict()
                    bias_slot_dict[slot][bias_type] = [sparse_slot_dict[slot].last_start, sparse_slot_dict[slot].last_end + 1]
            for slot in set(sparse_features.get(k)['slot_id']):
                sparse_features_list.append(slot)

        sparse_features_list = list(set(sparse_features_list))

        # 解析sequence feature
        sequence_features = self.model_config['feature_slot']['sequence_feature']
        sequence_features_list = []
        for k in sequence_features.keys():
            slot = sequence_features.get(k)['slot_id'][0]
            if slot not in sparse_slot_dict:
                sparse_slot_dict[slot] = SingleSlot(slot, sequence_features.get(k)['emb_size'], True)
            else:
                raise Exception("sequence feature " + slot + "has been defined more than once")
            for slot in set(sequence_features.get(k)['slot_id']):
                sequence_features_list.append(slot)
        
        features_set = set(sparse_features_list + sequence_features_list)
        sparse_slots = list(features_set)
        # features 可能每次的顺序都不一样，所以需要排序以保证一样
        sparse_slots.sort()

        dense_features = self.model_config['feature_slot']['dense_feature']
        dense_features_list = []
        for k in dense_features.keys():
            dense_features_list.append(dense_features.get(k)['slot_id'])

        # todo 这里支持序列建模时，序列特征要抽出来
        max_embed_size = 0
        for k in sparse_slot_dict.keys():
            if max_embed_size < sparse_slot_dict.get(k).total_emb_size:
                max_embed_size = sparse_slot_dict.get(k).total_emb_size
        logging.info("max embedding size={}".format(str(max_embed_size)))

        #self.feature_slots = {slot_id : tn.feature_column.FeatureSlot(slot_id) for slot_id in sparse_slots + dense_features_list}
        featureid_to_slot = {'42285': '9517', '42284': '9516', '42287': '9519', '42286': '9518', '41119': '8351', '41268': '8500', '41269': '8501', '41266': '8498', '41267': '8499', '41264': '8496', '41265': '8497', '41262': '8494', '41263': '8495', '40888': '8120', '41296': '8528', '41123': '8355', '41234': '8466', '40904': '8136', '42283': '9515', '40907': '8139', '41855': '9087', '41854': '9086', '41857': '9089', '40550': '7782', '41233': '8465', '40955': '8187', '40908': '8140', '41231': '8463', '41339': '8571', '41189': '2602', '41232': '3305', '40948': '8180', '40949': '8181', '41237': '8469', '41236': '8468', '40944': '8176', '40945': '8177', '40946': '8178', '40947': '8179', '41187': '2602', '40941': '8173', '40942': '8174', '40943': '8175', '42296': '9528', '42297': '9529', '42294': '9526', '42295': '9527', '42292': '9524', '42293': '9525', '42290': '9522', '41122': '8354', '41253': '8485', '42289': '9521', '41251': '8483', '41250': '8482', '41129': '8361', '42298': '9530', '42288': '9520', '41239': '8471', '41858': '9090', '41238': '8470', '40887': '8119', '41196': '3306', '40952': '8184', '41842': '9074', '41840': '9072', '41841': '9073', '41222': '8454', '41300': '8532', '41303': '8535', '41341': '8573', '41225': '8457', '41188': '2602', '41229': '3305', '41283': '8515', '41132': '8364', '41133': '8365', '41130': '8362', '41131': '8363', '40547': '7779', '41171': '3306', '41235': '8467', '40880': '8112', '40883': '8115', '40882': '8114', '40885': '8117', '40884': '8116', '41839': '9071', '41838': '9070', '41837': '9069', '41836': '9068', '41835': '9067', '41834': '9066', '41833': '9065', '41832': '9064', '41831': '9063', '40545': '7777', '40953': '8185', '40546': '7778', '41223': '8455', '41856': '9088', '42291': '9523', '40951': '8183', '41313': '8545', '40950': '8182', '41331': '8563', '40549': '7781', '42313': '9545', '42312': '9544', '42311': '9543', '42310': '9542', '42317': '9549', '42316': '9548', '42315': '9547', '42314': '9546', '41121': '8353', '41242': '8474', '41676': '3303', '40954': '8186', '41271': '8503', '41230': '3305', '41270': '8502', '40893': '8125', '40886': '8118', '41861': '9093', '40894': '8126', '41859': '9091', '40551': '7783', '42299': '9531', '41675': '3303', '41674': '3303', '41120': '8352', '40905': '8137', '42304': '9536', '42305': '9537', '42306': '9538', '42307': '9539', '42300': '9532', '42301': '9533', '42302': '9534', '42303': '9535', '42308': '9540', '42309': '9541', '41244': '8476', '41245': '8477', '41246': '8478', '41247': '8479', '41240': '8472', '41241': '8473', '40850': '8082', '41243': '8475', '41252': '8484', '41860': '9092', '41202': '3306', '41248': '8480', '41249': '8481'}

        self.feature_slots = {}
        for feature_id in sparse_slots + dense_features_list:
            if feature_id in featureid_to_slot:
                slot_id = featureid_to_slot[feature_id]
            else:
                slot_id = feature_id

            self.feature_slots[feature_id] = tn.feature_column.FeatureSlot(slot_id)
        # 用feature_id后续可以做到同一个不同的feature_id公用同一个slot的embedding空间
        # 用feature_id后续可以做到同一个不同的feature_id公用同一个slot的embedding空间
        sparse_features = {slot_id : tn.feature_column.Feature(feature_id=slot_id, feature_slot=self.feature_slots[slot_id], sparse=True)
                                for slot_id in sparse_slots}

        sparse_inputs = {fea_id : tn.layers.Input(name=fea_id, feature=feature,
                              shape=(None,), dtype="int64", sparse=True)
                            for fea_id, feature in sparse_features.items()}

        sparse_embs = self.fetch_embeddings(sparse_inputs, max_embed_size)
        self.sparse_emb_input = sorted([(sparse_features[fea_id], emb) for fea_id, emb in sparse_embs.items()])

        dense_features = {slot_id : tn.feature_column.Feature(feature_id=slot_id, feature_slot=self.feature_slots[slot_id], sparse=False)
                            for slot_id in dense_features_list}

        self.dense_inputs = {fea_id : tn.layers.Input(name=fea_id, feature=feature,
                                shape=(1), dtype="float32", sparse=False)
                                for fea_id, feature in dense_features.items()}
        self.dense_emb_input = sorted([(dense_features[fea_id], emb) for fea_id, emb in self.dense_inputs.items()])

        # 产出模型的输入
        emb_dict = {}
        # 这个是线上infer的sub_model的inut
        self.sub_model_emb_inputs = []
        for feature, emb in self.sparse_emb_input:
            tmp_input = tn.layers.Input(name="emb_{}".format(feature.feature_id),
                                    feature=feature, dtype="float32", shape=emb.shape[1:])
            self.sub_model_emb_inputs.append(tmp_input)
            emb_dict[str(feature.feature_id)] = tmp_input

        for feature, emb in self.dense_emb_input:
            tmp_input = tn.layers.Input(name="dense_weight_{}".format(feature.feature_id),
                                    feature=feature, dtype="float32", shape=emb.shape[1:])
            self.sub_model_emb_inputs.append(tmp_input)

        emb_structure_input = []
        gate_feature_list = ['1568', '1570', '1578', '1591', '1593', '1614', '1736', '1737', '2039', '2599', '3051', '3303', '3389', '1576', '1577', '1578']
        emb_gate_input = []
        for slot in sparse_slot_dict.keys():
            for s in sparse_slot_dict.get(slot).intervals:
                ss = [str(i) for i in s]
                logging.info("common slot emb_size={}".format(slot + ": " + ",".join(ss)))
                emb_structure_input.append(emb_dict[slot][:, s[0]:s[1]])
                if slot in gate_feature_list:
                    emb_gate_input.append(emb_dict[slot][:, s[0]:s[1]])

        # bias input
        emb_bias_input = {}
        for slot in sorted(bias_slot_dict.keys()):
            for bias_type in bias_slot_dict.get(slot).keys():
                logging.info("bias emb_size={}".format(slot + ": "+ bias_type + "," + str(bias_slot_dict[slot][bias_type][0]) + "," + str(bias_slot_dict[slot][bias_type][1])))
                if bias_type in emb_bias_input:
                    emb_bias_input[bias_type].append(emb_dict[slot][:, bias_slot_dict[slot][bias_type][0]:bias_slot_dict[slot][bias_type][1]])
                else:
                    emb_bias_input[bias_type] = []
                    emb_bias_input[bias_type].append(emb_dict[slot][:, bias_slot_dict[slot][bias_type][0]:bias_slot_dict[slot][bias_type][1]])
        
        self.emb_structure_input = emb_structure_input
        self.emb_bias_input = emb_bias_input
        self.emb_gate_input = emb_gate_input
        self.inputs = { **sparse_inputs, **self.dense_inputs }

    def optimizer_layer(self, optimizer):
        if optimizer == 'adam':
            sparse_opt = tn.core.Adam(learning_rate=0.00005, beta1=0.9, beta2=0.999, epsilon=1e-8)
        return sparse_opt

    def model_layer(self):
        self.output = None

    def output_layer(self, output):
        # self.sub_model_emb_inputs: tn.layers.Input
        # self.sparse_emb_input: tn.feature_column.embedding_colum
        sub_model = tn.model.Model(inputs=self.sub_model_emb_inputs, outputs=output, name="sub_model")

        emb_inputs = [ emb for _, emb in self.sparse_emb_input] +  [ emb for _, emb in self.dense_emb_input]
        output = sub_model(emb_inputs)
        
        model = tn.model.Model(inputs=self.inputs, outputs=output, name="full_model")
        
        losses = {
                'video_id_rank_hp_ctr_addfeasetwo_click': cross_entropy,
                'video_id_rank_hp_ctr_addfeasetwo_effect_click': cross_entropy
        }
        metrics = {
                'video_id_rank_hp_ctr_addfeasetwo_click': ['acc', tf.keras.metrics.AUC(), tn.metric.COPC()],
                'video_id_rank_hp_ctr_addfeasetwo_effect_click': ['acc', tf.keras.metrics.AUC(), tn.metric.COPC()]
        }
        weighted_metrics = {
                'video_id_rank_hp_ctr_addfeasetwo_click': ['acc', tf.keras.metrics.AUC(), tn.metric.COPC()],
                'video_id_rank_hp_ctr_addfeasetwo_effect_click': ['acc', tf.keras.metrics.AUC(), tn.metric.COPC()]
        }

        dense_opt = tn.core.Adam(learning_rate=0.00005, beta1=0.9, beta2=0.999, epsilon=1e-8)
        model.compile(optimizer=tn.optimizer.Optimizer(dense_opt),
                        loss=losses,
                        metrics=metrics,
                        weighted_metrics=weighted_metrics)

        return {
            "train" : model,
            "predict" : model,
        }

    def fetch_embeddings(self, inputs, max_embed_size):
        columns = {}
        for fea_id, _ in inputs.items():
            columns[fea_id] = tn.feature_column.category_column(key=fea_id, bucket_size=265000)

        embedding_columns = []
        for fea_id, _ in inputs.items():
            feature_column = tf.feature_column.embedding_column(columns[fea_id],
                        dimension=max_embed_size, combiner='mean')
            embedding_columns.append(feature_column)

        sparse_opt = self.optimizer_layer('adam')

        sparse_embs = tn.layers.EmbeddingFeatures(embedding_columns, sparse_opt, name="sparse_emb_input")(inputs)
        return sparse_embs

