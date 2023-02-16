# -*- coding: utf-8 -*-
import sys
import logging
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.python.keras.layers import Lambda
from .base_model import BaseModel

from .common_module.interacting_layer import InteractingLayer
from .common_module.multi_dense_layer import MultiLayerDense

class Model(BaseModel):
    def __init__(self, model_config, **kwargs):
        super(Model, self).__init__(model_config,  **kwargs)

    def output_layer(self, output):
        return super(Model, self).output_layer(output)

    def model_layer(self):
        logging.info("enter model layer============")

        emb_inputs_squeeze = []
        for emb in self.emb_structure_input:
            emb_inputs_squeeze.append(tf.reduce_mean(emb,axis=1,keepdims=True))
        
        ## senet
        inputs_squeeze = tf.keras.layers.Concatenate(name='emb_inputs_squeeze_concacted', axis=1)(emb_inputs_squeeze)
        logging.info('inputs_squeeze concat={}'.format(inputs_squeeze))
        senet_reduction = 4
        senet_unit1 = len(emb_inputs_squeeze) // senet_reduction
        inputs_squeeze_nograd = tf.stop_gradient(inputs_squeeze)
        senet_output = tf.keras.layers.Dense(senet_unit1, activation='relu', name='senet_squeeze_layer')(inputs_squeeze_nograd)
        logging.info('senet_squeeze_layer={}'.format(senet_output))
        senet_unit2 = len(emb_inputs_squeeze)
        senet_output = 2 * tf.keras.layers.Dense(senet_unit2, activation='sigmoid', name='senet_extract_layer')(senet_output)
        logging.info('senet_extract_layer={}'.format(senet_output))
        senet_split_outputs = tf.split(senet_output, len(emb_inputs_squeeze), axis=1)

        emb_inputs_reweight = []
        for emb_input,senet_split_output in zip(self.emb_structure_input, senet_split_outputs):
            emb_inputs_reweight.append(tf.keras.layers.multiply([emb_input, senet_split_output]))
        
        # 所有inputs的feature embedding concat
        emb_3d_inputs = []
        for i,emb in enumerate(emb_inputs_reweight):
            emb_output = tf.keras.layers.Dense(8, activation=None, name='emb_linear_map_{}'.format(i))(emb)
            emb_3d_inputs.append(emb_output[:, tf.newaxis, :])

        autoint_inputs = tf.keras.layers.Concatenate(name='autoint_concated', axis=1)(emb_3d_inputs)
        logging.info('autoint_inputs={}'.format(autoint_inputs))

        # model structure
        ## autoint
        autoint_outputs = InteractingLayer(layer_num=1,
                                           unit_num=8,
                                           head_num=2,
                                           use_dropout=True,
                                           dropout_rate=0.2,
                                           use_res=True)(autoint_inputs)
        autoint_outputs = Flatten()(autoint_outputs)
        logging.info('autoint_outputs={}'.format(autoint_outputs))
        
        ## ppnet
        ppnet_inputs = tf.keras.layers.Concatenate(name='ppnet_concacted', axis=1)(self.emb_bias_input['ppnet'])
        logging.info('ppnet_inputs={}'.format(ppnet_inputs))
        ppnet_gate = 2 * tf.keras.layers.Dense(256+64+8+256+64+8+32+16, activation='sigmoid', name='dnn_ppnet_gate')(ppnet_inputs)
        logging.info('ppnet_gate={}'.format(ppnet_gate))
        ppnet_gate_list = tf.split(ppnet_gate, [256,64,8,256,64,8,32,16],axis=1)

        ## deep
        # deep_result = MultiLayerDense(units=[32,16], activation='relu')(deep_result)
        deep_result = tf.keras.layers.Concatenate(name='ori_concacted', axis=1)(emb_inputs_reweight)
        deep_hidden_units = [32,16]
        for  i, unit in enumerate(deep_hidden_units):
            deep_result = tf.keras.layers.Dense(unit, activation=None, name='dnn_{}'.format(i), kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00001, l2=0.00001))(deep_result)
            deep_result = tf.keras.layers.multiply([deep_result, ppnet_gate_list[i+6]])
            deep_result = tf.keras.layers.ReLU()(deep_result)
        logging.info('deep_result={}'.format(deep_result))
        
        ## multiply
        multiply_user_inputs = tf.keras.layers.Concatenate(name='multiply_user_concated', axis=1)(self.emb_bias_input['multiply_user'])
        multiply_item_inputs = tf.keras.layers.Concatenate(name='multiply_item_concated', axis=1)(self.emb_bias_input['multiply_item'])
        multiply_result = tf.keras.layers.multiply([multiply_user_inputs, multiply_item_inputs])
        multiply_result = tf.keras.layers.ReLU()(multiply_result)
        logging.info('multiply_result={}'.format(multiply_result))

        # concat autoint deep multiply
        result = tf.keras.layers.Concatenate(name='all_input_concated', axis=1)([deep_result, autoint_outputs, multiply_result])
        logging.info('result input={}'.format(result))
        
        ## can
        can_inputs = tf.keras.layers.Concatenate(name='can_concacted', axis=1)(self.emb_bias_input['can'])
        can_inputs = tf.keras.layers.Dense(8*6+6+6*4+4, activation=None, name='dnn_can')(can_inputs)
        can_inputs_list = tf.split(can_inputs,[8*6,6,6*4,4],axis=1)
        can_weight1 = tf.reshape(can_inputs_list[0],[-1,8,6])
        can_bias1 = tf.reshape(can_inputs_list[1],[-1,1,6])
        can_weight2 = tf.reshape(can_inputs_list[2],[-1,6,4])
        can_bias2 = tf.reshape(can_inputs_list[3],[-1,1,4])
        logging.info('can_weight1={}, can_bias1={}, can_weight2={}, can_bias2={}'.format(can_weight1, can_bias1, can_weight2, can_bias2))

        ## mmoe
        expert_outs = []
        num_experts = 3
        mmoe_expert_hidden_units = [512,256]
        gate_input = tf.keras.layers.Concatenate(name='gate_concacted', axis=1)(self.emb_gate_input)
        for i in range(num_experts):
            expert_result = result
            for j, unit in enumerate(mmoe_expert_hidden_units):
                gate_output = tf.keras.layers.Dense(unit, activation='relu', name='gate_{}_{}_1'.format(i,j))(gate_input)
                gate_output = 2 * tf.keras.layers.Dense(unit, activation='sigmoid', name='gate_{}_{}_2'.format(i,j))(gate_output)
                logging.info("gate_output_{}_{}={}".format(i, j, gate_output))

                expert_result = tf.keras.layers.Dense(unit, activation='relu', name='expert_output_{}_{}'.format(i,j))(expert_result)
                expert_result = tf.multiply(gate_output, expert_result)
                logging.info("expert_output_{}_{}={}".format(i, j, expert_result))
            expert_outs.append(expert_result)
        expert_concat = Lambda(lambda x: tf.stack(x, axis=1))(expert_outs)  # None,num_experts,dim
        logging.info("expert_concat={}".format(expert_concat))

        mmoe_outs = []
        num_tasks = 2
        mmoe_gate_hidden_units = [256,32]
        # gate: input_dim -> 256 -> 32 -> num_tasks
        for i in range(num_tasks):
            gate_out = result
            for j, unit in enumerate(mmoe_gate_hidden_units):
                gate_out = tf.keras.layers.Dense(unit, activation='relu', name='gate_{}_{}'.format(i, j))(gate_out)
            gate_out = tf.keras.layers.Dense(num_experts, activation='softmax', name='gate_output_{}'.format(i))(gate_out)
            gate_out = Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)
            gate_mul_expert = Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis=1, keepdims=False),
                                    name='gate_mul_expert_task{}'.format(i))([expert_concat, gate_out])
            mmoe_outs.append(gate_mul_expert)

        ## output mlp
        output_dnn_hidden_units=[64,8]
        task_names = ['video_id_rank_hp_ctr_addfeasetwo_click', 'video_id_rank_hp_ctr_addfeasetwo_effect_click']
        final_results = {}
        # 256 -> 64 -> 8 -> 1
        for i in range(num_tasks):
            result = mmoe_outs[i]
            for j, unit in enumerate(output_dnn_hidden_units):
                if j == 0:
                    result = tf.keras.layers.multiply([result, ppnet_gate_list[i*3]])
                    result = tf.keras.layers.ReLU()(result)
                result = tf.keras.layers.Dense(unit, activation=None, name='task{}_dnn2_{}'.format(i,j), kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00001, l2=0.00001))(result)
                result = tf.keras.layers.multiply([result, ppnet_gate_list[i*3+j+1]])
                result = tf.keras.layers.ReLU()(result)
                if j == len(output_dnn_hidden_units) - 1:
                    result_expand = tf.expand_dims(result, axis=1)
                    can_result = tf.matmul(result_expand, can_weight1) + can_bias1
                    can_result = tf.keras.layers.ReLU()(can_result)
                    can_result = tf.matmul(can_result, can_weight2) + can_bias2
                    can_result = tf.keras.layers.ReLU()(can_result)
                    can_result = tf.squeeze(can_result, axis=1)
                    result = tf.keras.layers.Concatenate(name='task{}_result_concacted'.format(i), axis=1)([result, can_result])
            output = tf.keras.layers.Dense(1, activation='sigmoid')(result)
            output = tf.clip_by_value(output, 1e-6, 1.0)
            output = tf.identity(output, name=task_names[i])
            logging.info('task{}_logits_output={}'.format(i, output))
            final_results[task_names[i]] = output

        self.output = final_results

    def run(self):
        self.model_layer()
        return self.output_layer(self.output)
