#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Create CNN model to train on decay string and whole event together
# James Kahn

from smartBKG.train import NNBaseClass  # type:ignore

from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.layers import LeakyReLU
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from keras.layers import Embedding, LSTM
from keras.layers import BatchNormalization
from keras.layers import concatenate, Flatten
from keras import optimizers


class NN_model(NNBaseClass):
    def __init__(
        self,
        shape_dict,
        num_pdg_codes,
    ):
        super().__init__()
        self.shape_dict = shape_dict
        self.num_pdg_codes = num_pdg_codes

        # adam = optimizers.Adam(lr=0.001, amsgrad=True)  # best so far
        adam = optimizers.Adam(lr=0.01)  # best so far
        # nadam = optimizers.Nadam(lr=0.002)
        # adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        # adadelta = optimizers.Adadelta(lr=1.0, epsilon=None, decay=0.0)
        # rmsProp = optimizers.RMSprop(lr=0.001)
        self.optimizer = adam

    def build_model(self):
        # Create joint embedding layer (decay strings)
        decstr_embedding = Embedding(
            self.num_pdg_codes,
            8,
            input_length=self.shape_dict['decay_input'][1],
        )

        # Network to process decay string
        decay_input = Input(shape=self.shape_dict['decay_input'][1:], name='decay_input')
        decay_embed = decstr_embedding(decay_input)

        # Build wide CNN for decay string processing
        wide_layers = []
        for i in range(4, 10):
            layer_w = self._conv1D_node(
                decay_embed,
                filters=64,
                kernel_size=i,
            )
            layer_w = GlobalAveragePooling1D()(layer_w)
            wide_layers.append(layer_w)

        # Put it all together, outputs 4xfilter_size = 128
        decay_l = concatenate(wide_layers, axis=-1)

        decay_l = Dropout(0.4)(decay_l)
        decay_l = Dense(256)(decay_l)
        decay_l = LeakyReLU()(decay_l)
        decay_l = Dropout(0.4)(decay_l)
        decay_l = Dense(64)(decay_l)
        decay_l = LeakyReLU()(decay_l)
        comb_output = Dense(1, activation='sigmoid', name='y_output')(decay_l)

        # Instantiate the cnn model
        model = Model(
            inputs=[decay_input],
            outputs=comb_output,
            name='decstr-wideCNN'
        )
        # Finally compile the model
        model.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy'],
        )
        model.summary()

        self.model = model
