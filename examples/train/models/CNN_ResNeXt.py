#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Create CNN model to train on decay string and whole event together
# James Kahn

from smartBKG.train import NNBaseClass  # type:ignore

from keras.models import Model
from keras.layers import Dense, Dropout, Input, Lambda
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
        cardinality=16,
    ):
        super().__init__()
        self.shape_dict = shape_dict
        self.num_pdg_codes = num_pdg_codes
        self.cardinality = cardinality

        adam = optimizers.Adam(lr=0.001, amsgrad=True)  # best so far
        # adam = optimizers.Adam(lr=0.001)
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

        # decay_l = Dropout(0.5)(decay_l)
        decay_l = Dense(128)(decay_l)
        decay_l = LeakyReLU()(decay_l)
        decay_l = Dropout(0.3)(decay_l)
        decay_l = Dense(64)(decay_l)
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

        # Create initial convolutional layer
        initial_filters = 64
        particle_l = self._conv1D_node(particle_l, filters=initial_filters, kernel_size=3, strides=1)
        # particle_l = self._conv1D_node(
        #     particle_l,
        #     filters=initial_filters,
        #     kernel_size=3,
        #     dropout=0.3,
        # )
        particle_l = MaxPooling1D(pool_size=2)(particle_l)

        # Build wide ResNeXt blocks
        # ######
        # Output shape = cardinality x output shape
        particle_r = self._grouped_convolution_block(
            input_layer=particle_l,
            cardinality=self.cardinality,
            grouped_channels=int(initial_filters / self.cardinality),
            # Args for expanded block
            kernels=[1, 3, 1],
            filters=4,
        )

        # Here allow input from before ResNeXt block too
        particle_l = Add()([particle_l, particle_r])
        # This needs a nonlinear activation too for some reason?
        particle_l = LeakyReLU()(particle_l)

        # ######
        # Output shape = cardinality x output shape
        particle_r = self._grouped_convolution_block(
            input_layer=particle_l,
            cardinality=self.cardinality,
            grouped_channels=int(initial_filters / self.cardinality),
            # Args for expanded block
            kernels=[1, 3, 1],
            filters=4
        )

        # Here allow input from before ResNeXt block too
        particle_l = Add()([particle_l, particle_r])
        # This needs a nonlinear activation too for some reason?
        particle_l = LeakyReLU()(particle_l)

        # particle_l = Dense(32, kernel_initializer='uniform')(particle_l)
        # particle_l = BatchNormalization()(particle_l)
        # particle_l = LeakyReLU()(particle_l)
        # particle_output = Dropout(0.4)(particle_l)
        particle_output = GlobalAveragePooling1D()(particle_l)

        # Finally, combine the two networks
        comb_l = concatenate([decay_output, particle_output], axis=-1)
        comb_l = Dense(128)(comb_l)
        comb_l = LeakyReLU()(comb_l)
        comb_l = Dropout(0.4)(comb_l)
        comb_l = Dense(64)(comb_l)
        comb_l = LeakyReLU()(comb_l)
        comb_l = Dropout(0.2)(comb_l)
        comb_l = Dense(32)(comb_l)
        comb_l = LeakyReLU()(comb_l)
        comb_output = Dense(1, activation='sigmoid', name='y_output')(comb_l)

        # Instantiate the cnn model
        model = Model(
            inputs=[decay_input, particle_input, pdg_input, mother_pdg_input],
            outputs=comb_output,
            name='ResNeXt-wideCNN'
        )
        # Finally compile the model
        model.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy'],
        )
        model.summary()

        self.model = model

    def _grouped_convolution_block(self, input_layer, cardinality, grouped_channels, **kwargs):
        ''' Build expanded blocks with grouped input channels '''

        resnext_l = []
        for c in range(self.cardinality):
            x = Lambda(
                lambda z: z[:, :, c * grouped_channels:(c + 1) * grouped_channels]
            )(input_layer)

            resnext_l.append(
                self._expanded_block(
                    layer=x,
                    output_shape=int(x.shape[-1]),
                    **kwargs,
                )
            )

        layer = concatenate(resnext_l, axis=-1)
        return layer

    def _expanded_block(self, layer, output_shape, kernels=[1, 3, 1], filters=4):
        ''' Convenient function to add a single ResNeXt block

        Produces conv1D layers with 1, 3, 1 sized kernels
        '''
        for kernel in kernels[:-1]:
            layer = self._conv1D_node(
                layer,
                filters=filters,
                kernel_size=kernel,
            )

        # Last layer needs to have requested output shape
        return self._conv1D_node(
            layer,
            filters=output_shape,
            kernel_size=kernels[-1],
        )
