#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Create CNN model to train on decay string and whole event together
# James Kahn

from smartBKG.train import NNBaseClass

from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.layers import LeakyReLU
from keras.layers import Conv1D, GlobalAveragePooling1D, AveragePooling1D
from keras.layers import Embedding
from keras.layers import BatchNormalization
from keras.layers import concatenate, Add, Flatten
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
                filters=32,
                kernel_size=i,
            )
            # layer_w = self._conv1D_node(
            #     layer_w,
            #     filters=32,
            #     kernel_size=i,
            # )
            layer_w = GlobalAveragePooling1D()(layer_w)
            wide_layers.append(layer_w)

        # Put it all together, outputs 4xfilter_size = 128
        # decay_l = concatenate([decay_4, decay_5, decay_6, decay_7, decay_8, decay_9], axis=-1)
        decay_l = concatenate(wide_layers, axis=-1)
        # decay_l = Add()(wide_layers)

        # decay_l = Dropout(0.4)(decay_l)
        decay_l = Dense(128)(decay_l)
        decay_l = LeakyReLU()(decay_l)
        # decay_l = Dropout(0.3)(decay_l)
        decay_l = Dense(8)(decay_l)
        decay_output = LeakyReLU()(decay_l)

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

        particle_l = self._resnet_node(particle_l, kernels=1, filters=64)
        particle_l = self._resnet_node(particle_l, kernels=1, filters=64)
        particle_l = self._resnet_node(particle_l, kernels=1, filters=64)
        particle_l = self._resnet_node(particle_l, kernels=1, filters=64)

        particle_l = self._resnet_node(particle_l, kernels=3, filters=128)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=128)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=128)
        # particle_l = self._resnet_node(particle_l, kernels=3, filters=128, pool='avg')
        particle_l = self._resnet_node(particle_l, kernels=3, filters=128)

        particle_l = self._resnet_node(particle_l, kernels=3, filters=64)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=64)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=64)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=64)

        # for i in range(4):
        #     particle_l = self._conv1D_node(
        #         particle_l,
        #         filters=64,
        #         kernel_size=1,
        #         # dropout=0
        #     )

        # particle_m = particle_l
        # particle_m = self.conv1D_avg_node(
        #     particle_m,
        #     filters=64,
        #     kernel_size=4,
        #     # pool='avg',
        #     # dropout=0.4
        # )
        # # for i in range(2):
        # #     # particle_m = self._conv1D_node(
        # #     particle_m = self.conv1D_avg_node(
        # #         particle_m,
        # #         filters=64,
        # #         kernel_size=4,
        # #         # dropout=0.4
        # #     )
        # for i in range(3):
        #     particle_m = self._conv1D_node(
        #         particle_m,
        #         filters=64,
        #         kernel_size=3,
        #         # dropout=0.4
        #     )

        # Add residual loopback to stop vanishing gradients
        # particle_l = Add()([particle_l, particle_m])

        # Compress
        # particle_output = AveragePooling1D(pool_size=2)(particle_l)

        # Flatten to feed to dense
        # particle_l = Flatten()(particle_3)
        particle_output = GlobalAveragePooling1D()(particle_l)

        # particle_l = Dense(256)(particle_l)
        # particle_l = LeakyReLU()(particle_l)
        # particle_l = Dropout(0.5)(particle_l)

        # particle_l = Dense(128)(particle_l)
        # particle_l = LeakyReLU()(particle_l)
        # particle_output = Dropout(0.5)(particle_l)

        # Finally, combine the two networks
        comb_l = concatenate([decay_output, particle_output], axis=-1)
        comb_l = Dense(128)(comb_l)
        comb_l = LeakyReLU()(comb_l)
        comb_l = Dropout(0.3)(comb_l)
        comb_l = Dense(64)(comb_l)
        comb_l = LeakyReLU()(comb_l)
        # comb_l = Dropout(0.3)(comb_l)
        comb_l = Dense(32)(comb_l)
        comb_l = LeakyReLU()(comb_l)
        comb_output = Dense(1, activation='sigmoid', name='y_output')(comb_l)

        # Instantiate the cnn model
        model = Model(
            inputs=[decay_input, particle_input, pdg_input, mother_pdg_input],
            outputs=comb_output,
            name='combined-1x1-ResNet'
        )
        # Finally compile the model
        model.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy'],
        )
        model.summary()

        self.model = model
