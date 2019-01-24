#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Create feed forward model for bias training
# James Kahn

import tensorflow as tf
from tensorflow.keras import layers


class NN_model():
    def __init__(
        self,
        input_shape,
    ):
        self.input_shape = input_shape
        adam = tf.keras.optimizers.Adam(lr=0.0005, amsgrad=True)
        # sgd = tf.keras.optimizers.SGD(lr=0.01, decay=0.1)
        # adam = tf.keras.optimizers.Adam(lr=0.0001)
        # nadam = optimizers.Nadam(lr=0.002)
        adagrad = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        # adadelta = optimizers.Adadelta(lr=1.0, epsilon=None, decay=0.0)
        # rmsProp = tf.keras.optimizers.RMSprop(lr=0.0005)
        self.optimizer = adam

    def build_model(self):

        l_input = layers.Input(shape=self.input_shape[1:], name='input')
        layer = l_input

        # BYO ResNet
        # layer = layers.Dense(32, kernel_initializer='uniform')(layer)
        # for i in range(6):
        #     init_l = layer
        #     layer = layers.BatchNormalization()(layer)
        #     layer = layers.LeakyReLU()(layer)
        #     layer = layers.Dense(32, kernel_initializer='uniform')(layer)
        #     layer = layers.BatchNormalization()(layer)
        #     layer = layers.LeakyReLU()(layer)
        #     layer = layers.Dense(32, kernel_initializer='uniform')(layer)
        #     # Add
        #     layer = layers.Add()([layer, init_l])

        # layer = layers.BatchNormalization()(layer)
        # layer = layers.LeakyReLU()(layer)

        for i in range(3):
            # layer = layers.Dense(1024, activation='tanh')(layer)
            layer = layers.Dense(64, kernel_initializer='uniform')(layer)
            layer = layers.BatchNormalization()(layer)
            layer = layers.LeakyReLU()(layer)
            # layer = layers.Dropout(0.2)(layer)

        # Finally, get the ouput
        l_output = layers.Dense(1, activation='sigmoid', name='output')(layer)

        # Instantiate the cnn model
        model = tf.keras.Model(
            inputs=[l_input],
            outputs=l_output,
            name='feed_forward'
        )
        # Finally compile the model
        model.compile(
            # loss='binary_crossentropy',
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy'],
        )
        model.summary()

        self.model = model
