#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Base class for cnn networks
# James Kahn

import os
from keras.utils import plot_model
from keras.models import load_model
from keras.layers import Conv1D
from keras.layers import MaxPooling1D, AveragePooling1D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import Add

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


class NNBaseClass():
    '''Base class for NNs

    Placeholder for common function.
    '''
    def __init__(
        self,
        shape_dict=None,
        num_pdg_codes=None,
    ):
        self.shape_dict = shape_dict,
        self.num_pdg_codes = num_pdg_codes,

        self.model = None

        # Set GPU memory usage to only what's needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"
        set_session(tf.Session(config=config))

    def plot_model(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        plot_model(
            self.model,
            to_file=os.path.join(out_dir, 'model_architecture.pdf'),
            show_shapes=True,
        )

    def load_model(self, model_file):
        self.model = load_model(model_file)

    def _conv1D_node(
        self,
        input_layer,
        filters=32,
        kernel_size=3,
        batchnorm=True,
        **kwargs
    ):
        ''' Build a standard conv1D node '''
        layer = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            use_bias=(not batchnorm),
            # kernel_initializer='he_normal',
            # kernel_regularizer=l1_l2(l1=0.001, l2=0.05),
            **kwargs,
        )(input_layer)
        if batchnorm:
            layer = BatchNormalization()(layer)
        layer = LeakyReLU()(layer)

        return layer

    def conv1D_avg_node(
        self,
        input_l,
        filters=64,
        kernel_size=3,
        pool=None,
        **kwargs,
    ):
        ''' Collective convolutional node '''
        particle_l = input_l
        for i in range(2):
            particle_l = self._conv1D_node(
                particle_l,
                filters=filters,
                kernel_size=kernel_size,
                **kwargs,
            )
        # Compress
        if pool == 'max':
            particle_l = MaxPooling1D(pool_size=2)(particle_l)
        elif pool == 'avg':
            particle_l = AveragePooling1D(pool_size=2)(particle_l)

        return particle_l

    def _resnet_node(
        self,
        input_layer,
        n_layers=2,
        kernels=3,
        filters=32,
        pool=None,
    ):
        ''' Builds a standard resnet block including identity hop '''
        input_filters = int(input_layer.shape[-1])

        layer = input_layer
        for i in range(n_layers - 1):
            layer = self._conv1D_node(
                layer,
                filters=filters,
                kernel_size=kernels,
                # batchnorm=False,
            )

        # Final layer in node needs to have same shape as input to do Add
        layer = self._conv1D_node(
            layer,
            filters=input_filters,
            kernel_size=kernels,
        )
        layer = Conv1D(
            filters=input_filters,
            kernel_size=kernels,
            padding='same',
            # kernel_initializer='he_normal',
            # kernel_regularizer=l1_l2(l1=0.001, l2=0.05),
            # **kwargs,
        )(layer)

        layer = Add()([layer, input_layer])
        # layer = LeakyReLU()(layer)
        if pool == 'avg':
            layer = AveragePooling1D(pool_size=2)(layer)
        elif pool == 'max':
            layer = MaxPooling1D(pool_size=2)(layer)

        return layer

    def _resnet_preact_node(
        self,
        input_layer,
        n_layers=2,
        kernels=3,
        filters=32,
        pool=None,
    ):
        ''' Builds a standard resnet block including identity hop '''
        input_filters = int(input_layer.shape[-1])

        layer = input_layer
        for i in range(n_layers - 1):
            layer = BatchNormalization()(layer)
            layer = LeakyReLU()(layer)
            layer = Conv1D(
                filters=filters,
                kernel_size=kernels,
                padding='same',
                use_bias=False,
                # kernel_initializer='he_normal',
                # kernel_regularizer=l1_l2(l1=0.001, l2=0.05),
                # **kwargs,
            )(layer)

        # Final layer in node needs to have same shape as input to do Add
        layer = BatchNormalization()(layer)
        layer = LeakyReLU()(layer)
        layer = Conv1D(
            filters=input_filters,
            kernel_size=kernels,
            padding='same',
            use_bias=False,
            # kernel_initializer='he_normal',
            # kernel_regularizer=l1_l2(l1=0.001, l2=0.05),
            # **kwargs,
        )(layer)

        layer = Add()([layer, input_layer])
        # layer = LeakyReLU()(layer)
        if pool == 'avg':
            layer = AveragePooling1D(pool_size=2)(layer)
        elif pool == 'max':
            layer = MaxPooling1D(pool_size=2)(layer)

        return layer
