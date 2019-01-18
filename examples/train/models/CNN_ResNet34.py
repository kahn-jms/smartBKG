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
from keras.layers import concatenate
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
        sgd = optimizers.SGD(lr=0.01)
        self.optimizer = sgd

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

        # decay_l = Dropout(0.4)(decay_l)
        decay_l = Dense(128)(decay_l)
        decay_l = LeakyReLU()(decay_l)
        # decay_l = Dropout(0.1)(decay_l)
        decay_output = Dense(32)(decay_l)
        decay_l = LeakyReLU()(decay_l)

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

        # particle_l = self._conv1D_node(particle_l, filters=64, kernel_size=7)
        # particle_l = self._conv1D_node(particle_l, filters=64, kernel_size=3)

        # Block 1
        particle_l = self._resnet_node(particle_l, kernels=3, filters=64)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=64)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=64, pool='avg')

        # Block 2
        # particle_l = self.conv1D_avg_node(particle_l, filters=128, kernel_size=3)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=128)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=128)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=128)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=128, pool='avg')

        # Block 3
        # particle_l = self.conv1D_avg_node(particle_l, filters=256, kernel_size=3)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=256)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=256)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=256)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=256)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=256)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=256, pool='avg')

        # Block 4
        # particle_l = self.conv1D_avg_node(particle_l, filters=512, kernel_size=3)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=512)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=512)
        particle_l = self._resnet_node(particle_l, kernels=3, filters=512)

        # Flatten (not really)
        particle_output = GlobalAveragePooling1D()(particle_l)

        # Finally, combine the two networks
        comb_l = concatenate([decay_output, particle_output], axis=-1)
        comb_l = Dense(1024)(particle_output)
        comb_l = LeakyReLU()(comb_l)
        # comb_l = Dropout(0.1)(comb_l)
        # comb_l = Dense(64)(particle_output)
        # comb_l = LeakyReLU()(comb_l)
        # comb_l = Dropout(0.1)(comb_l)
        # comb_l = Dense(32)(comb_l)
        # comb_l = LeakyReLU()(comb_l)
        # comb_l = Dropout(0.4)(comb_l)
        # comb_l = Dense(256)(comb_l)
        # comb_l = LeakyReLU()(comb_l)
        comb_output = Dense(1, activation='sigmoid', name='y_output')(comb_l)

        # Instantiate the cnn model
        model = Model(
            inputs=[decay_input, particle_input, pdg_input, mother_pdg_input],
            outputs=comb_output,
            name='combined-wide-ResNet34'
        )
        # Finally compile the model
        model.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy'],
        )
        model.summary()

        self.model = model
