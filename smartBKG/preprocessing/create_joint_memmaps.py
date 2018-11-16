#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Create joint memory maps of numpy preprocessed files
# James Kahn

import numpy as np
import pickle


class CreateJointMemmap():
    ''' Create single memory map file of many npy files '''
    def __init__(self, memmap=None, memmap_mode=None):
        self.memmap = memmap
        self.memmap_mode = memmap_mode

    def grow_memmap(self, in_files, out_file):
        ''' Grow the class memmap by all input files '''
        for idx, f in enumerate(in_files):
            print('Loading file [{}/{}]:'.format(idx + 1, len(in_files)), f)
            arr = self._load_npy(f)
            self.memmap = self._append_ndarray(self.memmap, arr, out_file)
            del arr

    def flush_memmap(self):
        ''' Flush the current memmap to disk and delete the object (start fresh)

        Also saves the memmap metadata to out_file.shape and out_file.dtype
        '''
        with open('{}.shape'.format(self.memmap.filename), 'wb') as f:
            pickle.dump(self.memmap.shape, f)
        with open('{}.dtype'.format(self.memmap.filename), 'wb') as f:
            pickle.dump(str(self.memmap.dtype), f)

        self.memmap.flush()
        self.memmap = None

    def _load_npy(self, npy_file):
        ''' Load a single numpy file as a memmap '''
        return np.load(
            npy_file,
            mmap_mode=self.memmap_mode,
        )

    def _append_ndarray(self, memmap, array, out_file):
        ''' Append input to array to memmap '''
        # Need to create the memmap if this is the first array added
        if memmap is None:
            print('Creating memmap:', out_file)
            memmap = np.memmap(
                out_file,
                mode='w+',
                shape=array.shape,
                dtype=array.dtype,
            )
            memmap[:] = array[:]
        else:
            memmap = np.memmap(
                out_file,
                mode='r+',
                shape=(
                    memmap.shape[0] + array.shape[0],
                    *(memmap.shape[1:])
                ),
                dtype=array.dtype,
            )
            memmap[-array.shape[0]:] = array[:]
        return memmap
