#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Base class for cnn networks
# James Kahn

import os
from keras.utils import plot_model
from keras.models import load_model
from keras.layers import Conv1D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import LeakyReLU

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
        dropout=0,
        **kwargs
    ):
        ''' Build a standard conv1D node '''
        layer = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            use_bias=False,
            # kernel_initializer='he_normal',
            # kernel_regularizer=l1_l2(l1=0.001, l2=0.05),
            **kwargs,
        )(input_layer)
        layer = BatchNormalization()(layer)
        layer = LeakyReLU()(layer)
        if dropout:
            layer = Dropout(dropout)(layer)

        return layer
