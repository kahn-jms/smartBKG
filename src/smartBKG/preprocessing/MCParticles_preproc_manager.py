#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Preprocessing class with functions to call when applying CNN
# James Kahn

import pandas as pd
import numpy as np
import os
import argparse
from keras.preprocessing import sequence
from MCParticles_preproc_pandas import MCParticlesPreprocPandas


class MCParticlesPreprocManager():
    def __init__(self, max_decstr_len=150, max_FSPs=100):
        self.max_decstr_len = max_decstr_len
        self.max_FSPs = max_FSPs
        self.ppp = MCParticlesPreprocPandas()

        self.cont_vars = ['energy', 'prodTime', 'x', 'y', 'z', 'px', 'py', 'pz']
        self.cont_min = [0.0, 0.0, -700., -700., -700., -3.0, -3.0, -2.0]
        self.cont_max = [11.0, 150.0, 700., 700., 700., 3.0, 3.0, 4.0]

        self.cont_min_series = pd.Series(self.cont_min, index=self.cont_vars)
        self.cont_max_series = pd.Series(self.cont_max, index=self.cont_vars)

        self.disc_vars = ['charge', 'PDG', 'motherPDG']  # , 'nDaughters']

    def build_decay_string(self, particle):
        """Build particle decay string from given particle down

        Need to recode this without recursion.
        """
        dec_string = ' {}'.format(particle.getPDG())

        # Check at least one primary particle daughter exists before diving down a layer
        if (
            particle.getNDaughters() > 0 and
            [(d.isPrimaryParticle() and self.check_status_bit(d.getStatus())) for d in particle.getDaughters()].count(True)
        ):
            dec_string += ' (-->'
            for daughter in particle.getDaughters():
                if daughter.isPrimaryParticle() and self.check_status_bit(daughter.getStatus()):
                    dec_string += self.build_decay_string(daughter)
            dec_string += ' <--)'
        return dec_string

    def _preproc_cont_vars(self, df):
        ''' Perform necessary preprocessing of self.cont_vars '''
        # Normalise continuous variables
        return (df - self.cont_min_series) / (self.cont_max_series - self.cont_min_series)

    def _preproc_disc_vars(self, df):
        ''' Perform necessary preprocessing of self.disc_vars '''
        # One-hot encode discrete variables (only charge)
        # Have to force label ranges since we're processing one file at a time
        dummy_cols = ['{}_{}'.format(self.disc_vars[0], float(c)) for c in range(-2, 3)]
        dummy_df = pd.get_dummies(
            df[self.disc_vars[0]],
            columns=[self.disc_vars[0]],
            prefix=self.disc_vars[0]
        )
        dummy_df = dummy_df.T.reindex(dummy_cols).T.fillna(0)
        df = pd.concat([df, dummy_df], axis=1)
        df = df.drop(self.disc_vars[0], axis=1)

        # Hashing-trick encode PDG and mother PDG
        # df = self._hash_PDG(df, self.disc_vars[1], n_dims=10)
        # df = self._hash_PDG(df, self.disc_vars[2], n_dims=10)
        df[self.disc_vars[1]] = pd.to_numeric(df[self.disc_vars[1]].apply(self.ppp.tokenize_PDG_code))
        df[self.disc_vars[2]] = pd.to_numeric(df[self.disc_vars[2]].apply(self.ppp.tokenize_PDG_code))

        return df

    def preproc_single_whole_decay(self, df):
        ''' Keeping this separate for the time speedup in application '''
        # Combine the preprocessed discrete and continuous dataframe chunks
        df = pd.concat(
            [
                self._preproc_cont_vars(df[self.cont_vars]),
                self._preproc_disc_vars(df[self.disc_vars])
            ],
            axis=1
        )

        # Extract particle_input
        x_arr = df.drop(self.disc_vars[1:3], axis=1).values
        x_arr = np.reshape(x_arr, (1, x_arr.shape[0], x_arr.shape[1]))
        # Not sure where to put this, maybe should be in preproc function?
        x_arr = sequence.pad_sequences(
            x_arr,
            maxlen=self.max_FSPs,
            padding='post',
            truncating='post'
        )

        # Extract pdg_input
        pdg_arr = df[self.disc_vars[1]].values
        pdg_arr = np.reshape(pdg_arr, (1, pdg_arr.shape[0]))
        # Not sure where to put this, maybe should be in preproc function?
        pdg_arr = sequence.pad_sequences(
            pdg_arr,
            maxlen=self.max_FSPs,
            padding='post',
            truncating='post'
        )

        # Extract mother_pdg_input
        mother_pdg_arr = df[self.disc_vars[2]].values
        mother_pdg_arr = np.reshape(mother_pdg_arr, (1, mother_pdg_arr.shape[0]))
        # Not sure where to put this, maybe should be in preproc function?
        mother_pdg_arr = sequence.pad_sequences(
            mother_pdg_arr,
            maxlen=self.max_FSPs,
            padding='post',
            truncating='post'
        )

        return x_arr, pdg_arr, mother_pdg_arr

    def preproc_whole_decay(self, df):
        ''' Keeping this separate for the time speedup in application '''
        # Combine the preprocessed discrete and continuous dataframe chunks
        df = pd.concat(
            [
                self._preproc_cont_vars(df[self.cont_vars]),
                self._preproc_disc_vars(df[self.disc_vars])
            ],
            axis=1
        )

        # Need to convert whole dataframe to float64 for memmapping later
        # Should really use structures numpy arrays for space saving
        df = df.astype(np.float64)

        # Remove label and arrayIndex indexes
        df.reset_index(level=['label', 'arrayIndex'], drop=True, inplace=True)

        # Then populate a new arrayIndex column, need for pivoting
        df['newIndex'] = df.groupby('evtNum', sort=False).cumcount()

        # If index=None, uses existing index, in this case that's evtNum
        pivot = df.pivot(columns='newIndex')
        pivot = pivot.fillna(0.)

        # Reshape and swap axis to get (event, particle, var)
        # The pivot table has two levels for column names so that's what we're spliting into 2 dims
        # PDG and motherPDG will go into separate arrays
        x_arr = pivot.drop(self.disc_vars[1:3], axis=1).values.reshape(
            -1,
            pivot.columns.levels[0].shape[0] - 2,
            pivot.columns.levels[1].shape[0],
        ).swapaxes(1, 2)
        pdg_arr = pivot[self.disc_vars[1]].values.reshape(
            -1,
            pivot.columns.levels[1].shape[0],
        )
        mother_pdg_arr = pivot[self.disc_vars[2]].values.reshape(
            -1,
            pivot.columns.levels[1].shape[0],
        )

        # Pad output arrays
        # Should put these in separate function
        x_arr = sequence.pad_sequences(
            x_arr,
            maxlen=self.max_FSPs,
            padding='post',
            truncating='post',
            dtype=x_arr.dtype,
        )
        pdg_arr = sequence.pad_sequences(
            pdg_arr,
            maxlen=self.max_FSPs,
            padding='post',
            truncating='post',
            dtype=pdg_arr.dtype,
        )
        mother_pdg_arr = sequence.pad_sequences(
            mother_pdg_arr,
            maxlen=self.max_FSPs,
            padding='post',
            truncating='post',
            dtype=mother_pdg_arr.dtype,
        )

        return x_arr, pdg_arr, mother_pdg_arr

    def preproc_single_decay_string(self, decay_string, LSTM_flag=False):

        # Tokenize the decay string
        tok_decstr = self.ppp.tokenize_decay_string(decay_string)

        # Change to numpy array of shape (1, )
        tok_decstr = np.array(tok_decstr)
        tok_decstr = np.reshape(tok_decstr, (1, -1))

        # Pad out the decay string
        # Need to put tok_decstr in list to pad correct dimension
        tok_decstr = sequence.pad_sequences(tok_decstr, maxlen=self.max_decstr_len)

        # If inputting to LSTM reshape to include time dim
        if LSTM_flag:
            tok_decstr = np.reshape(tok_decstr, (1, -1, 1))

        return tok_decstr

    def preproc_decay_string(self, df, LSTM_flag=False):
        # Tokenize the decay string
        token_df = df['decay_str'].apply(self.ppp.tokenize_decay_string)
        # token_df = df['decay_str'].apply(self._hash_decay_string)
        df['decay_str_tok'] = token_df

        # Change to numpy array of shape (1, )
        tok_decstr = df['decay_str_tok'].values
        print('tok_decstr shape:', tok_decstr.shape)
        # tok_decstr = np.reshape(tok_decstr, (-1, 1))

        # Pad out the decay string
        # Need to put tok_decstr in list to pad correct dimension
        tok_decstr = sequence.pad_sequences(
            tok_decstr,
            maxlen=self.max_decstr_len,
            padding='post',
            truncating='post',
            dtype=tok_decstr.dtype,
        )

        # Need to convert for memmaps later
        tok_decstr = tok_decstr.astype(int)

        # If inputting to LSTM reshape to include time dim
        if LSTM_flag:
            tok_decstr = np.reshape(tok_decstr, (1, -1, 1))

        return tok_decstr

    def preproc_y_output(self, df, key):
        ''' Return training labels as numpy array '''
        if key == 'train_events':
            # Want just one label per event, don't care about arrayIndex
            df.reset_index(level=['label', 'arrayIndex'], inplace=True)
            return df.groupby('evtNum', sort=False).first()['label'].values
        elif key == 'decay_strings':
            return df['label'].values

    def check_status_bit(self, status_bit):
        '''Returns True if conditions are satisfied (not an unusable particle)

        Move this method to preprocessPandas
        '''
        return (
            (status_bit & 1 << 4 == 0) &  # IsVirtual
            (status_bit & 1 << 5 == 0) &  # Initial
            (status_bit & 1 << 6 == 0) &  # ISRPhoton
            (status_bit & 1 << 7 == 0)  # FSRPhoton
        )

    def save_npy_preprocd(self, arr, filename):
        ''' Saves the numpy array to file '''
        np.save(
            filename,
            arr,
        )


def getCmdArgs():
    parser = argparse.ArgumentParser(
        description='''Preprocess data for NN input and save.''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', type=str, required=True,
                        help="Path to pandas h5 input.", metavar="INPUT",
                        dest='in_file')
    parser.add_argument('-k', type=str, required=False, choices=['train_events', 'decay_strings'],
                        help="Key name of pandas table in input file.", metavar="KEY",
                        dest='key')
    parser.add_argument('-o', type=str, required=True,
                        help="Directory to save numpy output (will have same filename as input + _KEY.np", metavar="OUTPUT",
                        dest='save_dir')
    return parser.parse_args()


if __name__ == '__main__':

    args = getCmdArgs()
    os.makedirs(args.save_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(args.in_file))[0]
    basefile = os.path.join(args.save_dir, basename)

    preproc = MCParticlesPreprocManager()

    print('Preprocessing file {}'.format(args.in_file))
    # Should really populate a dict and use keys from that
    if not args.key or args.key == 'train_events':
        print('Running particle preprocessing')
        df = preproc.ppp.load_hdf(args.in_file, 'train_events')
        particle_input, pdg_input, mother_pdg_input = preproc.preproc_whole_decay(df)
        y_output = preproc.preproc_y_output(df, key='train_events')

        print('Particle preprocessing finished, saving to {}'.format(args.save_dir))
        preproc.save_npy_preprocd(
            particle_input,
            '{}_particle_input.npy'.format(basefile)
        )
        preproc.save_npy_preprocd(
            pdg_input,
            '{}_pdg_input.npy'.format(basefile)
        )
        preproc.save_npy_preprocd(
            mother_pdg_input,
            '{}_mother_pdg_input.npy'.format(basefile)
        )
    if not args.key or args.key == 'decay_strings':
        print('Running decay string preprocessing')
        df = preproc.ppp.load_hdf(args.in_file, 'decay_strings')
        decay_input = preproc.preproc_decay_string(df)
        y_output = preproc.preproc_y_output(df, key='decay_strings')

        print('Decay string preprocessing finished, saving to {}'.format(args.save_dir))
        preproc.save_npy_preprocd(
            decay_input,
            '{}_decay_input.npy'.format(basefile)
        )

    print('Saving training labels'.format(args.save_dir))
    preproc.save_npy_preprocd(
        y_output,
        '{}_y_output.npy'.format(basefile)
    )
