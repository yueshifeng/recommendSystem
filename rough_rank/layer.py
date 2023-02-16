# -*- coding:utf-8 -*-

import tensorflow as tf


class Similarity(tf.keras.layers.Layer):
    """
      Input shape
        - A list of two 2D tensor with shape: ``(batch_size, input_dim)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
    """

    def __init__(self, use_sigmoid=False, **kwargs):
        self.use_sigmoid = use_sigmoid
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        user_emb, item_emb = inputs
        output = tf.reduce_sum(tf.multiply(user_emb, item_emb), axis=-1, keepdims=True)
        if self.use_sigmoid:
            output = tf.sigmoid(output)
        return output

    def get_config(self):
        config = {'use_sigmoid': self.use_sigmoid}
        base_config = super().get_config()
        base_config.update(config)
        return base_config


class DNN(tf.keras.layers.Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **output_activation**: Activation function to use in the last layer.If ``None``,it will be same as ``activation``.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=None, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_dropout = self.dropout_rate > 0
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super().__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i), shape=(hidden_units[i], hidden_units[i + 1]),
                                        initializer=tf.keras.initializers.GlorotNormal(seed=self.seed),
                                        regularizer=tf.keras.regularizers.L2(self.l2_reg), trainable=True) for i in
                        range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i), shape=(self.hidden_units[i],),
                                     initializer=tf.keras.initializers.Zeros(), trainable=True) for i in
                     range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]
        if self.use_dropout:
            self.dropout_layers = [
                tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i if self.seed is not None else None) for i
                in range(len(self.hidden_units))]

        if len(self.hidden_units) == 0:
            self.activation_layers = []
        elif self.output_activation:
            self.activation_layers = [tf.keras.layers.Activation(self.activation) for _ in
                                      range(len(self.hidden_units) - 1)] + [
                                         tf.keras.layers.Activation(self.output_activation)]
        else:
            self.activation_layers = [tf.keras.layers.Activation(self.activation) for _ in
                                      range(len(self.hidden_units))]

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        for i in range(len(self.hidden_units)):
            x = tf.nn.bias_add(tf.tensordot(x, self.kernels[i], axes=(-1, 0)), self.bias[i])
            if self.use_bn:
                x = self.bn_layers[i](x, training=training)
            x = self.activation_layers[i](x)
            if self.use_dropout:
                x = self.dropout_layers[i](x, training=training)
        return x

    def get_config(self):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units, 'l2_reg': self.l2_reg,
                  'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate, 'output_activation': self.output_activation,
                  'seed': self.seed}
        base_config = super().get_config()
        base_config.update(config)
        return base_config


class MMOE(tf.keras.layers.Layer):
    """
      Input shape
        - 2D tensor with shape: ``(batch_size, input_dim)``.

      Output shape
        - A list of 2D tensor with shape: ``(batch_size, expert_dnn_units[-1])``.
    """

    def __init__(self, num_tasks, num_experts=2, expert_dnn_units=(32,), gate_dnn_units=(), expert_dnn_params=None,
                 gate_dnn_params=None, **kwargs):
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.expert_dnn_units = expert_dnn_units
        self.gate_dnn_units = list(gate_dnn_units) + [self.num_experts]
        self.expert_dnn_params = {}
        if expert_dnn_params:
            self.expert_dnn_params.update(expert_dnn_params)
        self.gate_dnn_params = {'output_activation': 'softmax'}
        if gate_dnn_params:
            self.gate_dnn_params.update(gate_dnn_params)

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.expert_nets = [DNN(self.expert_dnn_units, name='expert{}'.format(i), **self.expert_dnn_params) for i in
                            range(self.num_experts)]
        self.gate_nets = [DNN(self.gate_dnn_units, name='task{}_gate'.format(i), **self.gate_dnn_params) for i in
                          range(self.num_tasks)]

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        expert_outs = [net(inputs, training=training) for net in self.expert_nets]
        expert_outs = tf.stack(expert_outs, axis=-2)  # (batch_size, num_experts, expert_dim)
        task_outs = []
        for i in range(self.num_tasks):
            gate_out = self.gate_nets[i](inputs, training=training)
            gate_out = tf.expand_dims(gate_out, axis=-1)  # (batch_size, num_experts, 1)
            # for j in range(self.gate_dnn_units[-1]):
            #     self.add_metric(gate_out[...,j], name='{}_task{}_gate{}'.format(self.name, i, j))
            task_out = tf.reduce_sum(tf.multiply(expert_outs, gate_out), axis=-2)  # (batch_size, expert_dim)
            task_outs.append(task_out)
        return task_outs

    def get_config(self):
        config = {'num_tasks': self.num_tasks, 'num_experts': self.num_experts,
                  'expert_dnn_units': self.expert_dnn_units, 'gate_dnn_units': self.gate_dnn_units,
                  'expert_dnn_params': self.expert_dnn_params, 'gate_dnn_params': self.gate_dnn_params}
        base_config = super().get_config()
        base_config.update(config)
        return base_config


class PLE(tf.keras.layers.Layer):
    """
      Input shape
        - 2D tensor with shape: ``(batch_size, input_dim)``.

      Output shape
        - A list of 2D tensor with shape: ``(batch_size, expert_dnn_units[-1])``.
    """

    def __init__(self, num_tasks, num_shared_experts=2, num_specific_experts=2, expert_dnn_units=(32,),
                 gate_dnn_units=(), expert_dnn_params=None, gate_dnn_params=None, **kwargs):
        self.num_tasks = num_tasks
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.expert_dnn_units = expert_dnn_units
        self.gate_dnn_units = list(gate_dnn_units) + [self.num_shared_experts + self.num_specific_experts]
        self.expert_dnn_params = {}
        if expert_dnn_params:
            self.expert_dnn_params.update(expert_dnn_params)
        self.gate_dnn_params = {'output_activation': 'softmax'}
        if gate_dnn_params:
            self.gate_dnn_params.update(gate_dnn_params)

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.shared_expert_nets = [
            DNN(self.expert_dnn_units, name='shared_expert{}'.format(i), **self.expert_dnn_params) for i in
            range(self.num_shared_experts)]
        self.specific_expert_nets = [
            [DNN(self.expert_dnn_units, name='task{}_expert{}'.format(i, j), **self.expert_dnn_params) for j in
             range(self.num_specific_experts)] for i in range(self.num_tasks)]
        self.gate_nets = [DNN(self.gate_dnn_units, name='task{}_gate'.format(i), **self.gate_dnn_params) for i in
                          range(self.num_tasks)]

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        shared_expert_outs = [net(inputs, training=training) for net in self.shared_expert_nets]
        task_outs = []
        for i in range(self.num_tasks):
            specific_expert_outs = [net(inputs, training=training) for net in self.specific_expert_nets[i]]
            expert_outs = tf.stack(shared_expert_outs + specific_expert_outs,
                                   axis=-2)  # (batch_size, num_experts, expert_dim)
            gate_out = self.gate_nets[i](inputs, training=training)
            # for j in range(self.gate_dnn_units[-1]):
            #     self.add_metric(gate_out[...,j], name='{}_task{}_gate{}'.format(self.name, i, j))
            gate_out = tf.expand_dims(gate_out, axis=-1)  # (batch_size, num_experts, 1)
            task_out = tf.reduce_sum(tf.multiply(expert_outs, gate_out), axis=-2)  # (batch_size, expert_dim)
            task_outs.append(task_out)
        return task_outs

    def get_config(self):
        config = {'num_tasks': self.num_tasks, 'num_shared_experts': self.num_shared_experts,
                  'num_specific_experts': self.num_specific_experts, 'expert_dnn_units': self.expert_dnn_units,
                  'gate_dnn_units': self.gate_dnn_units, 'expert_dnn_params': self.expert_dnn_params,
                  'gate_dnn_params': self.gate_dnn_params}
        base_config = super().get_config()
        base_config.update(config)
        return base_config


class CrossNet(tf.keras.layers.Layer):
    def __init__(self,layer_num=2,l2_reg=0,seed=1024, **kwargs):
        self.layer_num = layer_num
        self.l2_reg = l2_reg
        self.seed = seed
        super().__init__(**kwargs)

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(dim, 1),
                                        initializer=tf.keras.initializers.GlorotNormal(seed=self.seed),
                                        regularizer=tf.keras.regularizers.L2(self.l2_reg),
                                        trainable=True) for i in range(self.layer_num)]
        self.bias = [self.add_weight(name='bias' + str(i),
                                      shape=(dim, 1),
                                      initializer=tf.keras.initializers.Zeros(),
                                      trainable=True) for i in range(self.layer_num)]
        super(CrossNet, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x0 = tf.expand_dims(inputs, axis=2)
        xl = x0
        for i in range(self.layer_num):
            xw = tf.tensordot(xl, self.kernels[i], axes=(1, 0))
            dot_ = tf.matmul(x0, xw)
            xl = dot_ + self.bias[i] + xl
        xl = tf.squeeze(xl, axis=2)
        return xl

    def get_config(self):
        config = {'layer_num': self.layer_num, 'l2_reg': self.l2_reg, 'seed': self.seed}
        base_config = super().get_config()
        base_config.update(config)
        return base_config

class KDLoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.distillation_loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        super().__init__(**kwargs)

    def call(self, student_predictions, teacher_predictions):
        loss = self.distillation_loss_fn(teacher_predictions, student_predictions)
        return loss
