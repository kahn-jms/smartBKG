#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Create CNN model to train on decay string and whole event together
# James Kahn

from smartBKG.train import NNBaseClass  # type:ignore

from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.layers import LeakyReLU
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from keras.layers import Embedding
from keras.layers import BatchNormalization
from keras.layers import concatenate, Add
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

        adam = optimizers.Adam(lr=0.0005, amsgrad=True)  # best so far
        # nadam = optimizers.Nadam(lr=0.002)
        # adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        # adadelta = optimizers.Adadelta(lr=1.0, epsilon=None, decay=0.0)
        # rmsProp = optimizers.RMSprop(lr=0.001)
        self.optimizer = adam

    def build_model(self):
        # Create joint embedding layer
        pdg_embedding = Embedding(
            self.num_pdg_codes,
            8,
            input_length=self.shape_dict['pdg_input'][1],
        )

        # Network to process individual particles
        particle_input = Input(shape=self.shape_dict['particle_input'][1:], name='particle_input')

        # Embed PDG codes
        pdg_input = Input(shape=self.shape_dict['pdg_input'][1:], name='pdg_input')
        mother_pdg_input = Input(shape=self.shape_dict['mother_pdg_input'][1:], name='mother_pdg_input')

        pdg_l = pdg_embedding(pdg_input)
        mother_pdg_l = pdg_embedding(mother_pdg_input)

        # Put all the particle
        particle_l = concatenate([particle_input, pdg_l, mother_pdg_l], axis=-1)

        # for i in range(4):
        #     particle_l = self._resnet_node(particle_l, filters=64, avg_pool=2)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=64, pool='avg')
        particle_l = self._resnet_node(particle_l, kernels=3, filters=64, pool='avg')
        particle_l = self._resnet_node(particle_l, kernels=3, filters=32)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=32)

        # Flatten (not really)
        particle_output = GlobalAveragePooling1D()(particle_l)

        # particle_l = Dense(32, kernel_initializer='uniform')(particle_l)
        # particle_l = BatchNormalization()(particle_l)
        # particle_l = LeakyReLU()(particle_l)
        # particle_output = Dropout(0.4)(particle_l)

        # Finally, combine the two networks
        # comb_l = concatenate([decay_output, particle_output], axis=-1)
        comb_l = Dense(512)(particle_output)
        comb_l = LeakyReLU()(comb_l)
        comb_l = Dropout(0.4)(comb_l)
        comb_l = Dense(128)(comb_l)
        comb_l = LeakyReLU()(comb_l)
        # comb_l = Dropout(0.4)(comb_l)
        # comb_l = Dense(256)(comb_l)
        # comb_l = LeakyReLU()(comb_l)
        comb_output = Dense(1, activation='sigmoid', name='y_output')(comb_l)

        # Instantiate the cnn model
        model = Model(
            inputs=[particle_input, pdg_input, mother_pdg_input],
            outputs=comb_output,
            name='particles-ResNet'
        )
        # Finally compile the model
        model.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy'],
        )
        model.summary()

        self.model = model

    # def _resnet_node(
    #     self,
    #     input_layer,
    #     n_layers=2,
    #     kernels=3,
    #     filters=33,
    #     dropout=0,
    #     avg_pool=0,
    # ):
    #     ''' Builds a standard resnet block including identity hop '''
    #     input_filters = int(input_layer.shape[-1])

    #     layer = input_layer
    #     for i in range(n_layers - 1):
    #         layer = self._conv1D_node(
    #             layer,
    #             filters=filters,
    #             kernel_size=kernels,
    #             dropout=dropout,
    #         )

    #     # Final layer in node needs to have same shape as input to do Add
    #     layer = self._conv1D_node(
    #         layer,
    #         filters=input_filters,
    #         kernel_size=kernels,
    #         # dropout=dropout,
    #     )

    #     layer = Add()([layer, input_layer])
    #     layer = LeakyReLU()(layer)
    #     if avg_pool:
    #         layer = AveragePooling1D(pool_size=avg_pool)(layer)

    #     return layer
