#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Load memmaps and prepare for input to NN training
# James Kahn

import numpy as np
import os
import pickle
from sklearn.utils.class_weight import compute_class_weight
# from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences


class LoadMemmapData():
    def __init__(self, in_files, padding_dict=None, train_frac=0.9):
        self.in_files = in_files
        self.padding_dict = padding_dict
        self.train_frac = train_frac

        if self.train_frac < 0.5:
            print('Training fraction for data ({}) too low, setting to 0.5'.format(self.train_frac))
            self.train_frac = 0.5

        self.memmap_dict = {}
        self.x_train = {}
        self.x_test = {}
        self.y_train = {}
        self.y_test = {}

        self.shape_dict = {}

    def populate_train_test(self):
        ''' Load the memmaps and return x/y train/test split '''
        self._load_memmaps()

        self._balance_shapes()

        # if self.padding_dict:
        #     print('Padding data')
        #     self._pad_data()

        # print('Splitting data into train/test')
        # self._split_data()

        print('Calculating training weights')
        self._weight_data()

        print('Populating shapes')
        self._populate_shapes()

    def _load_memmaps(self):
        ''' Load the in_files as memmaps into memmap dict '''
        for key in self.in_files.keys():
            if self.in_files[key] is not None:
                print('Loading memmap {} from {}'.format(key, self.in_files[key]))
                # First load the memmap metadata
                shape = pickle.load(open('{}.shape'.format(self.in_files[key]), 'rb'))
                dtype = pickle.load(open('{}.dtype'.format(self.in_files[key]), 'rb'))
                print('Expected memmap shape/dtype:', shape, dtype)

                # if key == 'particle_input':
                #     dtype = np.uint8
                self.memmap_dict[key] = np.memmap(
                    filename=self.in_files[key],
                    mode='r',
                    dtype=dtype,
                    shape=shape,
                )
                print('Loaded memmap shape:', self.memmap_dict[key].shape)
                print('Loaded memmap dtype:', self.memmap_dict[key].dtype)

    def _balance_shapes(self):
        ''' Enforce same number of columns in all inputs/outputs '''
        # Find the array with least number of events/columns
        min_evts = min([a.shape[0] for a in self.memmap_dict.values()])
        print('Cropping array columns to:', min_evts)

        for k in self.memmap_dict.keys():
            self.memmap_dict[k] = self.memmap_dict[k][:min_evts]
            print('New shape for {}:'.format(k), self.memmap_dict[k].shape)

    def _pad_data(self):
        ''' Pad data to requested lengths '''
        for key in self.padding_dict.keys():
            if self.memmap_dict[key] is None:
                print('Key {} not found in loaded memmaps, skipping padding'.format(key))
                continue
            self.memmap_dict[key] = pad_sequences(
                self.memmap_dict[key],
                maxlen=self.padding_dict[key],
                padding='post',
                truncating='post',
            )

    def _split_data(self):
        ''' Split data into train and test portions '''
        # seed = 42
        total_evts = list(self.memmap_dict.values())[0].shape[0]
        train_events = int(total_evts * self.train_frac)
        test_events = int(total_evts - train_events)

        test_step_size = int(1 / (1 - self.train_frac))

        # Here we'll use basic slicing to split the data into two separate memmaps for test and train
        for k in self.memmap_dict.keys():
            print('Splitting array:', k)

            # Create train and test memmaps for array
            cache_dir = os.path.dirname(self.in_files[k])
            train_memmap = np.memmap(
                os.path.join(cache_dir, '{}_train.memmap'.format(k)),
                mode='w+',
                shape=(train_events, *self.memmap_dict[k].shape[1:]),
                dtype=self.memmap_dict[k].dtype,
            )
            test_memmap = np.memmap(
                os.path.join(cache_dir, '{}_test.memmap'.format(k)),
                mode='w+',
                shape=(test_events, *self.memmap_dict[k].shape[1:]),
                dtype=self.memmap_dict[k].dtype,
            )

            for sl in range(test_step_size):
                if sl % (test_step_size - 1) == 0:
                    test_memmap[:] = self.memmap_dict[k][sl:-1:test_step_size]
                else:
                    train_memmap[
                        (sl * test_events):((sl + 1) * test_events)
                    ] = self.memmap_dict[k][sl:-(test_step_size - sl):test_step_size]

            if k == 'y_output':
                self.y_test[k] = test_memmap
                self.y_train[k] = train_memmap
            else:
                self.x_test[k] = test_memmap
                self.x_train[k] = train_memmap

    def _weight_data(self):
        '''Set class weights to ensure an effective balance of training label occurances '''
        assert self.memmap_dict['y_output'] is not None, 'Training labels empty when trying to weight data'

        # Use sklearn to do this for us
        unique = np.unique(self.memmap_dict['y_output'])
        # Returns list with weights of i-th class
        class_weights = compute_class_weight('balanced', unique, self.memmap_dict['y_output'])
        self.class_weights = dict(zip(unique, class_weights))

        print('Class weights set to:', self.class_weights)
        return

    def _populate_shapes(self):
        ''' Need the input shapes for first layer of models during building '''
        for key in self.memmap_dict.keys():
            self.shape_dict[key] = self.memmap_dict[key].shape

    def batch_chunk_generator(
        self,
        batch_size,
        train_queue_size,
        test_queue_size
    ):
        '''Generator to load next chunk of batches into memory for fitting

        We are making the dangerous assumption here that the test data can be split the same as the train data
        '''
        train_chunk_size = batch_size * train_queue_size
        test_chunk_size = batch_size * test_queue_size
        total_evts = list(self.memmap_dict.values())[0].shape[0]
        train_step_size = int(total_evts / train_chunk_size)
        test_step_size = int(total_evts / test_chunk_size)
        print('train_step_size:', train_step_size)
        print('test_step_size:', test_step_size)

        while True:
            for i in range(int(total_evts / train_chunk_size)):
                # Select random indices for this training chunk
                # NOTE: This is incredibly slow, use basic slicing instead
                # train_idx = np.random.randint(total_evts, size=(train_chunk_size * batch_size))
                # test_idx = np.random.randint(total_evts, size=(test_chunk_size * batch_size))

                # Initialise training dicts
                batch_train_x_dict = {}
                batch_train_y_dict = {}
                batch_test_x_dict = {}
                batch_test_y_dict = {}

                for k in self.memmap_dict.keys():
                    if k == 'y_output':
                        print('Loading y_train')
                        batch_train_y_dict[k] = np.array(self.memmap_dict[k][i:-(train_step_size - i):train_step_size])
                        print('Loading y_test')
                        batch_test_y_dict[k] = np.array(self.memmap_dict[k][i + 1:-(test_step_size - i + 1):test_step_size])
                    else:
                        print('Loading x_train for {}'.format(k))
                        batch_train_x_dict[k] = np.array(self.memmap_dict[k][i:-(train_step_size - i):train_step_size])
                        print('Loading x_test for {}'.format(k))
                        batch_test_x_dict[k] = np.array(self.memmap_dict[k][i + 1:-(test_step_size - i + 1):test_step_size])

                # Return the training dicts for this batch chunk
                # print('Returning train_chunk:', batch_train_x_dict)
                yield (
                    batch_train_x_dict,
                    batch_train_y_dict,
                    batch_test_x_dict,
                    batch_test_y_dict,
                )

    def batch_generator(
        self,
        batch_size,
    ):
        total_evts = list(self.memmap_dict.values())[0].shape[0]
        step_size = int(total_evts / batch_size)

        # Initialise fixed memory holders
        batch_x_dict = {}
        batch_y_dict = {}
        for k in self.memmap_dict.keys():
            if k == 'y_output':
                batch_y_dict[k] = np.zeros(
                    (batch_size, *self.memmap_dict[k].shape[1:]),
                    dtype=self.memmap_dict[k].dtype,
                )
            else:
                batch_x_dict[k] = np.zeros(
                    (batch_size, *self.memmap_dict[k].shape[1:]),
                    dtype=self.memmap_dict[k].dtype,
                )

        # Now run the actual generator
        while True:
            for i in range(step_size):
                for k in self.memmap_dict.keys():
                    if k == 'y_output':
                        batch_y_dict[k][:] = self.memmap_dict[k][i:-(step_size - i):step_size]
                    else:
                        batch_x_dict[k][:] = self.memmap_dict[k][i:-(step_size - i):step_size]

                # Return the training dicts for this batch chunk
                yield (
                    batch_x_dict,
                    batch_y_dict,
                )

    #     for k in self.memmap_dict.keys():
    #     for i in range(int(self.memmap_dict.x_train[list(train_data.x_train.values())[0]].shape[0] / train_chunk_size)):
    #         batch_train_x_dict = {}
    #         batch_train_y_dict = {}
    #         batch_test_x_dict = {}
    #         batch_test_y_dict = {}
    #         for k in train_data.x_train.keys():
    #             batch_train_x_dict[k] = train_data.x_train[k][(i * train_chunk_size):(i + 1) * train_chunk_size]
    #             batch_test_x_dict[k] = train_data.x_test[k][(i * test_chunk_size):(i + 1) * test_chunk_size]

    #         for k in train_data.y_train.keys():
    #             batch_train_y_dict[k] = train_data.y_train[k][(i * train_chunk_size):(i + 1) * train_chunk_size]
    #             batch_test_y_dict[k] = train_data.y_test[k][(i * test_chunk_size):(i + 1) * test_chunk_size]

    #         yield (
    #             batch_train_x_dict,
    #             batch_train_y_dict,
    #             batch_test_x_dict,
    #             batch_test_y_dict,
    #         )
