# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.keras.layers import Layer

class DIN(Layer):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.layer_1 = tf.keras.layers.Dense(16, activation=tf.nn.sigmoid)
        self.layer_2 = tf.keras.layers.Dense(1, activation=None)

    def call(self, query, facts, mask):
        # mask = tf.equal(mask, tf.ones_like(mask))   #(batch_size, seq_len)
        seq_len = tf.shape(facts)[1]
        dim = tf.shape(facts)[2]
        queries = tf.tile(query, [1, seq_len])
        queries = tf.reshape(queries, tf.shape(facts))     # (batch_size, query_emb_size) -> (batch_size, seq_len, query_emb_size)
        din_all = tf.concat(
            [queries, facts, queries - facts, queries * facts], axis=-1)
        d_layer_1_all = self.layer_1(din_all)
        d_layer_2_all = self.layer_2(d_layer_1_all)
        d_layer_2_all = tf.reshape(d_layer_2_all, [-1, 1, seq_len])
        scores = d_layer_2_all
        # Mask
        if mask is not None:
            mask = tf.slice(mask, [0, 0], [tf.shape(mask)[0], seq_len])  # [B, T]
            key_masks = tf.expand_dims(mask, 1)  # (batch_size, 1, emb_size)
            paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
            # logging.info("scores is:{}".format(scores))
            scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]
        scores = tf.nn.softmax(scores)
        output = tf.matmul(scores, facts)   # (batch_size, 1, seq_len) * (batch_size, seq_len, emb_size) -> (batch_size, 1, emb_size)
        # logging.info("din output is:{}".format(output))
        #output = tf.reshape(output, (-1, dim))
        output = tf.squeeze(output, [1])
        # logging.info("final din output is:{}".format(output))
        return output


class DeepCrossLayer(Layer):
    '''
    Wang, Ruoxi, et al. "Deep & cross network for ad click predictions." Proceedings of the ADKDD'17. 2017. 1-7.
    input_shape = [batch_size, fields*emb]
    '''

    def __init__(self, num_layer=3, **kwargs):
        self.num_layer = num_layer
        super(DeepCrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = [batch_size, fields*emb]
        self.input_dim = input_shape[1]
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(
                self.add_weight(shape=[self.input_dim, 1], initializer='glorot_uniform', name='w_' + str(i),
                                trainable=True))
            self.bias.append(
                self.add_weight(shape=[self.input_dim, ], initializer='zeros', name='b_' + str(i), trainable=True))

    def call(self, inputs):
        for i in range(self.num_layer):
            if i == 0:
                cross = inputs * tf.matmul(inputs, self.W[i]) + self.bias[i] + inputs
            else:
                cross = inputs * tf.matmul(cross, self.W[i]) + self.bias[i] + cross
        return cross

    def get_config(self):
        config = super(DeepCrossLayer, self).get_config()
        config.update({
            'num_layer': self.num_layer,
        })

        return config


class FMLayer(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions without linear term and bias.
      input shape = (batch_size,field_size,embedding_size)
      output shape = (batch_size, 1)
    """

    def __init__(self, **kwargs):

        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if backend.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        square_of_sum = tf.square(tf.math.reduce_sum(
            inputs, axis=1, keepdims=True))
        sum_of_square = tf.math.reduce_sum(
            inputs* inputs, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        fm = 0.5 * tf.math.reduce_sum(cross_term, axis=-1, keepdims=False)

        return fm 

    def compute_output_shape(self, input_shape):
        return (None, 1)
