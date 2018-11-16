#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wrapper for tensorboard callback to make it work with fit_generator

from keras import callbacks
import numpy as np


class TensorBoardWrapper(callbacks.TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen  # The generator.
        self.nb_steps = nb_steps     # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        x_val_dict, y_val_dict = None, None
        for step in range(self.nb_steps):
            x_dict, y_dict = next(self.batch_gen)

            if x_val_dict is None:
                x_val_dict = self._init_val_dicts(x_dict, self.nb_steps)
            if y_val_dict is None:
                y_val_dict = self._init_val_dicts(y_dict, self.nb_steps)

            x_val_dict = self._copy_batch_vals(x_val_dict, x_dict, step)
            y_val_dict = self._copy_batch_vals(y_val_dict, y_dict, step)

        sample_weights = np.ones(list(y_val_dict.values())[0].shape[0])

        # Hackish way to add the validation data in the right order
        self.validation_data = [x_val_dict[k.name.split(':')[0]] for k in self.model.inputs]
        self.validation_data += list(y_val_dict.values()) + [sample_weights, 0.0]
        print('len valdata:', len(self.validation_data))

        return super().on_epoch_end(epoch, logs)

    def _copy_batch_vals(self, val_dict, batch_dict, step):
        ''' Copy the validation data from the current batch into the approriate place in the validation data '''
        for key in val_dict.keys():
            val_dict[key][
                step * batch_dict[key].shape[0]:(step + 1) * batch_dict[key].shape[0]
            ] = batch_dict[key]
        return val_dict

    def _init_val_dicts(self, batch_dict, steps):
        ''' Initialise the validation keys with zero np array '''
        val_dict = {}
        for key in batch_dict.keys():
            val_dict[key] = np.zeros(
                (steps * batch_dict[key].shape[0], *batch_dict[key].shape[1:]),
                dtype=batch_dict[key].dtype
            )
        return val_dict
