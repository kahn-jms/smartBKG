#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Load and prepare data for input to NN all in single numpy array
# James Kahn

import os
import numpy as np
import keras
import pandas as pd
from tensorflow.keras.preprocessing import text
from smartBKG import evtPdl  # type:ignore


class MCParticlesPreprocPandas():
    def __init__(self, train_frac=0.9):
        self.train_frac = train_frac
        # pd.set_option('display.expand_frame_repr', False)

        # Set maximum length of decay string elements
        self.num_pdg_codes = len(evtPdl.pdgTokens)

        # Set arrays needed for each mode
        # Maybe should put this in loadTraining.py?
        self.arr_dict = {
            'single': ['x', 'y'],
            'whole': ['x', 'pdg', 'mother_pdg', 'y'],
            'decstr': ['decstr', 'y'],
            'combined': ['x', 'pdg', 'mother_pdg', 'decstr', 'y'],
        }

        # Set particle variables we will use and there normalisation constants
        self.cont_vars = ['energy', 'prodTime', 'x', 'y', 'z', 'px', 'py', 'pz']
        self.cont_min = [0.0, 0.0, -700., -700., -700., -3.0, -3.0, -2.0]
        self.cont_max = [11.0, 150.0, 700., 700., 700., 3.0, 3.0, 4.0]

        self.cont_min_series = pd.Series(self.cont_min, index=self.cont_vars)
        self.cont_max_series = pd.Series(self.cont_max, index=self.cont_vars)

        self.disc_vars = ['charge', 'PDG', 'motherPDG']  # , 'nDaughters']

        # Set up our tokenizer for decay string
        self.tokenize = self._init_tokenizer()

        # These will be filled in by the appropriate method
        self.init_train_data()

    def init_train_data(self):
        self.event_subsample = None
        self.data_dict = {}

    def load_hdf(self, f, key):
        '''Generic hdf5 loader to return one single DF'''
        return pd.read_hdf(f, key)

    def _load_hdf_files(self, files, key):
        '''Generic hdf5 loader to return one single DF'''
        in_data = []
        for f in files:
            in_data.append(pd.read_hdf(f, key))

        return pd.concat(in_data)

    def load_npy_data(self, cache_dir, basename, key):
        '''Load previously saved arrays, there must be a better way'''
        self.init_train_data()
        # for f in glob.glob(os.path.join(cache_dir, '{}_{}_*.npy'.format(basename, key)))
        for arr in self.arr_dict[key]:
            # Use some regex to get the array name, always at end: file_arr.npy
            # arr = re.split('_|\.', f)[-2]
            mmap_mode = None
            if arr != 'decstr':
                mmap_mode = 'r'

            print('Loading preprocessed array:', os.path.join(cache_dir, '{}_{}_{}.npy'.format(basename, key, arr)))
            self.data_dict[arr] = np.load(
                os.path.join(cache_dir, '{}_{}_{}.npy'.format(basename, key, arr)),
                mmap_mode=mmap_mode,
            )

    def save_npy_data(self, cache_dir, basename, key):
        '''Save train and test data.'''
        os.makedirs(cache_dir, exist_ok=True)

        for arr in self.data_dict.keys():
            np.save(
                os.path.join(cache_dir, '{}_{}_{}.npy'.format(basename, key, arr)),
                self.data_dict[arr],
            )

    def load_vae_decay_strings(self, files):
        '''Load decay string training data'''
        print('Preprocessing decay strings')
        df = self.load_hdf(files, 'decay_strings')

        # Tokenize the decay string, apply is significantly faster when not appending column inline too
        token_df = df['decay_str'].apply(self._tokenize_decay_string)
        # token_df = df['decay_str'].apply(self._hash_decay_string)
        df['decay_str_tok'] = token_df

        y_arr = df['label'].values
        x_arr = df['decay_str_tok'].values

        self.data_dict['decstr'] = x_arr
        self.data_dict['y'] = y_arr

        # Return list of event numbers as is
        self.evt_subsample = df.index.get_level_values(0)

    def load_vae_whole_decay(self, files, use_evt_subsample=False):
        ''' Load training data for vae whole decay classification

        This might be a lot easier to do in numpy, the balancing and shuffling.
        '''
        print('Preprocessing event particles (whole decay)')
        if use_evt_subsample:
            assert self.evt_subsample is not None, 'Requested event number subsample without subsample populated'

        df = self.load_hdf(files, 'train_events')

        # Split into continuous and discrete variables
        df_cont = df[self.cont_vars]
        df_disc = df[self.disc_vars]

        # Normalise continuous variables
        df_cont = (df_cont - self.cont_min_series) / (self.cont_max_series - self.cont_min_series)

        # One-hot encode discrete variables (only charge)
        # Have to force label ranges since we're processing one file at a time
        dummy_cols = ['{}_{}'.format(self.disc_vars[0], float(c)) for c in range(-2, 3)]
        # Should put this in a fucntion
        dummy_df = pd.get_dummies(df_disc[self.disc_vars[0]], columns=[self.disc_vars[0]], prefix=[self.disc_vars[0]])
        dummy_df = dummy_df.T.reindex(dummy_cols).T.fillna(0)
        df_disc = pd.concat([df_disc, dummy_df], axis=1)
        df_disc = df_disc.drop(self.disc_vars[0], axis=1)

        # # Hashing-trick encode PDG and mother PDG
        # df_disc = self._hash_PDG(df_disc, self.disc_vars[1], n_dims=10)
        # df_disc = self._hash_PDG(df_disc, self.disc_vars[2], n_dims=10)
        df_disc[self.disc_vars[1]] = pd.to_numeric(df_disc[self.disc_vars[1]].apply(self.tokenize_PDG_code))
        # df_disc[self.disc_vars[1]] = df_disc[self.disc_vars[1]].apply(pd.to_numeric)
        df_disc[self.disc_vars[2]] = pd.to_numeric(df_disc[self.disc_vars[2]].apply(self.tokenize_PDG_code))
        # df_disc[self.disc_vars[2]] = df_disc[self.disc_vars[2]].apply(pd.to_numeric)

        # Combine the two back together
        df = pd.concat([df_cont, df_disc], axis=1)

        # Temporarily remove label and arrayIndex indexes to make .loc work
        df.reset_index(level=['label', 'arrayIndex'], inplace=True)

        # Slice and dice baby
        if use_evt_subsample:
            df = df.loc[self.evt_subsample, :]

        # Want just one label per event
        y_arr = df.groupby('evtNum', sort=False).first()['label'].values

        # Throw away the labels
        df = df.drop('label', axis=1)

        # Need to do away with array index, he useless
        df = df.drop('arrayIndex', axis=1)
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

        # Update event count
        self.data_dict['x'] = x_arr
        self.data_dict['pdg'] = pdg_arr
        self.data_dict['mother_pdg'] = mother_pdg_arr
        self.data_dict['y'] = y_arr

    def load_vae_single_particle(self, files):
        ''' Load training data for vae single particle classification'''
        df = self.load_hdf(files, 'train_events')

        # To remove by status bit
        df = df[
            (df['status'] & 1 << 4 == 0) &  # IsVirtual
            (df['status'] & 1 << 5 == 0)  # Initial
        ]

        # Drop columns we don't use in vae training
        # df = df.drop(['motherIndex', 'status'], axis=1)
        cont_vars = ['energy', 'prodTime', 'x', 'y', 'z', 'px', 'py', 'pz']
        disc_vars = ['charge', 'PDG', 'motherPDG']
        df_cont = df[cont_vars]
        df_disc = df[disc_vars]

        # Remove outliers before normalising (could also do with quantiles)
        # df = df[np.abs(df - df.mean()) <= 5 * df.std()]
        # df.dropna(inplace=True)
        # Same with quantiles
        # q = df.quantile((0.01, 0.99))
        # Do something with this like (needs fixing)
        # df = df[df < q]

        # Normalise continuous variables
        df_cont = (df_cont - df_cont.mean()) / (df_cont.std())
        # And set these for saving
        self.mean = df_cont.mean()
        self.std = df_cont.std()

        # One-hot encode discrete variables (charge only)
        df_disc = pd.get_dummies(df_disc, columns=['charge'], prefix=['charge'])

        # Hashing-trick encode PDG and mother PDG
        df_disc = self._hash_PDG(df_disc, disc_vars[1], n_dims=10, drop_column=False)
        df_disc = self._hash_PDG(df_disc, disc_vars[2], n_dims=10)

        # Combine the two back together
        df = pd.concat([df_cont, df_disc], axis=1)

        # Mix that shit up
        subsample = df.sample(frac=1.0)

        # Return train, test
        n_rows = subsample.shape[0]
        # Need these for plotting and classification later
        # self.y_train = self.x_train.index.get_level_values(level='label')
        # self.y_test = self.x_test.index.get_level_values(level='label')
        self.y_train = subsample[:int(self.train_frac * n_rows)][disc_vars[1]]
        self.y_test = subsample[int(self.train_frac * n_rows):][disc_vars[1]]

        # Throw away original PDG
        subsample = subsample.drop(disc_vars[1], axis=1)

        self.x_train = subsample[:int(self.train_frac * n_rows)]
        self.x_test = subsample[int(self.train_frac * n_rows):]
        # self.x_train = self.x_train.drop('PDG', axis=1)
        # self.x_test = self.x_test.drop('PDG', axis=1)

    def _sort_data(self):
        '''sort data according to key'''
        for event in self.in_data:
            event[1].sort(key=event[1][13])

    def _extract_data(self, in_data):
        '''extract the different particle properties from the
        input and put them in seperate arrays'''
        max_fsps = max(map(len, [e[1] for e in in_data]))
        print('Max FSPs:', max_fsps)

        labels = []    # event labels
        data_list = []      # list of full data
        # decay_str = []

        # Collect our data
        # For now in a flattened array, should eventually split by event for the test data or include event numbers
        # for latent space plotting
        for event in in_data:
            for a in event[1]:
                labels.append(event[0])
                data_list.append(
                    np.concatenate(
                        (
                            a[0:2],  # PDG, Mass
                            a[3:4],  # Energy, Production time
                            a[5:11],  # x, y, z, px, py, pz
                            (keras.utils.to_categorical(a[2], num_classes=3)).flatten(),  # Charge (one-hot)
                            [a[11]],  # NDaughters
                            [a[14]],  # motherPDG
                        )
                    )
                )
            # decay_str.append(event[2][3])
        return max_fsps, labels, data_list  # , decay_str

    def _normalize_data(self, data):
        # Normalize data
        # Eventually move to gen_var_saver when I've decided on best scheme
        # [0,1] norm across each channel individually
        # Not sure if the padding will mess this up, probably?
        # Need to figure out how to avoid that

        mean = np.mean(data, axis=(0, 1), keepdims=True)
        std = np.std(data, axis=(0, 1), keepdims=True)
        norm = (data - mean) / std
        return norm

    def _separate_data(self, labels, data_list):
        '''separate the extracted data into test and training sets

        Should really use the sklearn train_test_split function
        '''
        # How many events we'll process
        num_data = len(labels)
        print('Num data:', num_data)

        train_labels = keras.utils.to_categorical(
            labels[:int(self.train_frac * num_data)], num_classes=2)
        test_labels = keras.utils.to_categorical(
            labels[int(self.train_frac * num_data):], num_classes=2)

        self._normalize_data(data_list)
        train_data_list = np.array(data_list[:int(self.train_frac * num_data)])
        test_data_list = np.array(data_list[int(self.train_frac * num_data):])

        # train_decay_str = decay_str[:int(self.train_frac * num_data)]
        # test_decay_str = decay_str[int(self.train_frac * num_data):]

        # return (train_labels, train_data_list, train_decay_str), (test_labels, test_data_list, test_decay_str)
        return (train_labels, train_data_list), (test_labels, test_data_list)

    def _PDGdecompose(self, index):
        '''decomposition of pdg index'''
        # ANTI-/PARTICLE
        if np.sign(index) > 0:
            decompose = '01'
        elif np.sign(index) < 0:
            decompose = '10'
        else:
            print('FormatError: PDG index error')
            exit()
        for i in range(0, 7 - len(str(np.absolute(index)))):
            decompose += '0'
        for i in str(np.absolute(index)):
            decompose += i
        print(index, decompose)
        return decompose

    def _hash_PDG(self, df, column, n_dims=10, drop_column=True):
        '''Hash given PDG column to n dimensions by first tokenizing then converting to binary'''
        # First create a separate series of the hashes
        # Process: convert PDG code into its token (int from 1-541),
        # then convert to binary and left pad so all will be same length,
        # then convert them all to int.
        hash_series = df[column].apply(
            lambda x: [
                int(i) for i in list(
                    bin(
                        self.tokenize_decay_string(str(x))[0]
                    )[2:].rjust(n_dims, '0')
                )
            ]
        )
        # Now split the series of lists into a df with n_dims columns
        col_names = ['{}_{}'.format(column, x) for x in range(n_dims)]
        hash_df = pd.DataFrame(
            hash_series.values.tolist(),
            index=hash_series.index,
            columns=col_names
        )

        # Delete the original column
        if drop_column:
            df = df.drop(column, axis=1)

        # Return the merged df
        return pd.merge(df, hash_df, left_index=True, right_index=True)

    def _hash_decay_string(self, decay_str):
        '''Convert decay string into list of hashes?'''

        hash_list = []
        hash_dims = 10  # should be global var
        for el in decay_str.split():
            el_hash = [
                int(i) for i in list(
                    bin(
                        self.tokenize_decay_string(str(el))[0]
                    )[2:].rjust(hash_dims, '0')
                )
            ]
            hash_list.append(el_hash)

        # Now df entry will be a list of lists, each sub list the binary hash of the PDG code
        return hash_list

    def tokenize_decay_string(self, decay_str):
        '''Lambda method to call as DataFrame apply during decay string tokenization

        Converts individual decay string into their tokens.
        '''
        # Should make this call tokenize_PDG_code on the split
        tok = self.tokenize.texts_to_sequences(decay_str.split())
        # Convert to single list of tokens
        return [i for sub in tok for i in sub]

    def tokenize_PDG_code(self, pdg_code):
        '''Lambda method to call as DataFrame apply during PDG tokenization
        '''
        tok = self.tokenize.texts_to_sequences([str(pdg_code)])
        # Convert to single token
        return tok[0][0]

    def _init_tokenizer(self):
        '''Initialise tokenizer for processing decay strings'''

        tokenize = text.Tokenizer(
            num_words=self.num_pdg_codes,
            filters='!"#$%&*+,./:;=?@[\]^_`{|}~'
        )
        tokenize.fit_on_texts(evtPdl.pdgTokens)
        return tokenize


# def getCmdArgs():
#     parser = argparse.ArgumentParser(
#         description='''Preprocess data for NN input and save.''',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument('-i', type=str, required=True, nargs='+',
#                         help="Path to training files.", metavar="INPUT",
#                         dest='in_files')
#     parser.add_argument('-o', type=str, required=True,
#                         help="Directory to save numpy arrays", metavar="OUTPUT",
#                         dest='save_dir')
#     return parser.parse_args()


# if __name__ == '__main__':

#     args = getCmdArgs()
#     os.makedirs(args.save_dir, exist_ok=True)
#     preproc = MCParticlesPreprocPandas()
#     preproc.preproc_sequence(args.in_files)
#     preproc.save_data(args.save_dir)
