# -*- coding: utf-8 -*-
import sys
import logging
import tensorflow as tf

class DIN(tf.keras.layers.Layer):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.nn = []
        for i, unit in enumerate([16, 1]):
            deep = tf.keras.layers.Dense(unit, activation='relu', name='din_nn_{}'.format(i))
            self.nn.append(deep)

    def call(self, queries, keys, values, seq_length=None):
        queries = tf.expand_dims(queries, axis=[1])

        from_seq_len = tf.shape(queries)[1]
        to_seq_len = tf.shape(keys)[1]

        masks = tf.sequence_mask(seq_length)

        queries = tf.expand_dims(queries, axis=2) # [B, F, 1, H]
        keys = tf.expand_dims(keys, axis=1) # [B, 1, T, H]
        queries = tf.tile(queries, [1, 1, to_seq_len, 1]) # [B, F, T, H]
        keys = tf.tile(keys, [1, from_seq_len, 1, 1]) # [B, F, T, H]

        deep = tf.concat([queries, keys, queries * keys], axis=-1) # [B, F, T, H]

        for nn in self.nn:
            deep = nn(deep)

        # after dnn # [B, F, T, 1]
        deep = tf.squeeze(deep, axis=-1) # [B, F, T]

        if masks is not None:
            masks = tf.expand_dims(masks, axis=1) # [B, 1, T]
            masks = tf.tile(masks, [1, from_seq_len, 1]) # [B, F, T]
            deep = tf.where(masks, deep, tf.zeros_like(deep)) # [B, F, T]

        output = tf.matmul(deep, values)  # [B, F, H]
        output = tf.squeeze(output, [1])

        return output
