#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module to save MCParticles to Pandas df or ROOT file
# James Kahn 2018

import basf2 as b2
from ROOT import Belle2
import pandas as pd
import numpy as np
import root_pandas  # noqa
# from keras.preprocessing import sequence, text
# import evtPdl  # type:ignore


class TrainDataSaver(b2.Module):
    '''Saves pickle of MC information for NN training input.

    '''
    def __init__(
        self,
        evt_num_file,
        output_file,
        keep_only=-1,
    ):
        '''Class constructor.

        Args:
            evt_num_file (str): Path to h5 file containing pandas df of pass event numbers, \
                    will be created if it doesn't exist.
            output_file (str): Filename to save training data to.
            keeponly (int): Flag to signal whether to save information for only interesting events (1), \
                    fail events (0), or all events (-1).
        '''
        super().__init__()
        self.evt_num_file = evt_num_file
        self.output_file = output_file
        self.keep_only = keep_only

        # self.num_pdg_codes = len(evtPdl.pdgTokens)

    def initialize(self):
        '''Create a member to access event info StoreArray'''
        self.eventinfo = Belle2.PyStoreObj('EventMetaData')

        self.index_names = ['label', 'evtNum', 'arrayIndex']
        # Dataframe to hold training data, format: (label, Dataframe of event particles, event level vars)
        self.columns = [
            'PDG',
            'mass',
            'charge',
            'energy',
            'prodTime',
            'x',
            'y',
            'z',
            'px',
            'py',
            'pz',
            'nDaughters',
            'status',
            'motherPDG',
            'motherIndex',
        ]
        self.events_list = []
        self.decay_str_list = []

        # If the evtNum file does not exist we need to create it
        self.create_evtNum_file = False
        try:
            # Eventually want to have series of LFN: [eventNums]
            # self.evtNum_series = pd.read_hdf(self.evt_num_file, 'evtNum')
            self.evtNum_arr = np.load(self.evt_num_file).tolist()
            b2.B2INFO('Event file {} loaded'.format(self.evt_num_file))
        # except FileNotFoundError:  # For pd series
        except IOError:
            b2.B2INFO('Event file {} not found, creating'.format(self.evt_num_file))
            self.create_evtNum_file = True
            self.evtNum_arr = []

    def event(self):
        '''Return match of event number to input list'''
        mcplist = Belle2.PyStoreArray("MCParticles")

        # Get event number, need for DF index
        evtNum = self.eventinfo.getEvent()
        # parentLFN = self.eventinfo.getParentLFN()  # for pd series

        # Get training label, need for DF index
        useful = True
        # If we already have list of event numbers then only keep requested events
        if not self.create_evtNum_file:
            useful = evtNum in self.evtNum_arr
            if self.keep_only != -1 and self.keep_only != useful:
                return
        else:
            self.evtNum_arr.append(evtNum)

        event_dict = {}

        # Create particle vars
        for mcp in mcplist:
            if mcp.isPrimaryParticle() and self._check_status_bit(mcp.getStatus()):

                # Load particle's data
                arrayIndex = mcp.getArrayIndex()
                four_vec = mcp.get4Vector()
                prod_vec = mcp.getProductionVertex()
                mother = mcp.getMother()
                motherPDG = 0
                motherArrayIndex = 0
                if mother:
                    motherPDG = mother.getPDG()
                    motherArrayIndex = mother.getArrayIndex()

                # Append to dict for making dataframe later
                event_dict[(useful, evtNum, arrayIndex)] = {
                    'PDG': mcp.getPDG(),
                    'mass': mcp.getMass(),
                    'charge': mcp.getCharge(),
                    'energy': mcp.getEnergy(),
                    'prodTime': mcp.getProductionTime(),
                    'x': prod_vec.x(),
                    'y': prod_vec.y(),
                    'z': prod_vec.z(),
                    'px': four_vec.Px(),
                    'py': four_vec.Py(),
                    'pz': four_vec.Pz(),
                    'nDaughters': mcp.getNDaughters(),
                    'status': mcp.getStatus(),
                    'motherPDG': motherPDG,
                    'motherIndex': motherArrayIndex,
                }

        # Create event wide feedforward vars
        # First particle is always the top of the decay chain
        if len(event_dict) > 0:
            MCdecay_string = self._build_decay_string(mcplist[0])

        # data_dict['decay_input'] = self.cap.preproc_single_decay_string(MCdecay_string, self.LSTM_flag)
        decay_str_df = pd.DataFrame(
            data=[{
                'label': useful,
                'decay_str': MCdecay_string
            }],
            index=[evtNum],
        )
        self.decay_str_list.append(decay_str_df)

        # If I tokenize the decay string here then root_pandas should work
        event_df = pd.DataFrame.from_dict(event_dict, orient='index')
        event_df.index.names = self.index_names
        self.events_list.append(event_df)

    def terminate(self):
        '''called once after all the processing is complete'''
        # Put all the data together into a dataframe
        self.event_df = pd.concat(self.events_list)
        self.decay_str_df = pd.concat(self.decay_str_list)
        self.decay_str_df.index.names = ['evtNum']

        # Save our df
        if self.output_file.endswith('.h5'):
            self.event_df.to_hdf(self.output_file, key='train_events')
            self.decay_str_df.to_hdf(self.output_file, key='decay_strings')
        elif self.output_file.endswith('.root'):
            self.event_df.to_root(self.output_file, key='train_events')
            # Export dtype=object to ROOT not yet supported
            # self.decay_str_df.to_root(self.output_file, key='decay_strings', mode='a')

        # And save the event numbers if needed
        if self.create_evtNum_file:
            np.save(self.evt_num_file, self.evtNum_arr)

    # All functions below here are copies of expert MCParticles preproc manager
    # Need to convert everything to a package and can directly call those funcs
    def _build_decay_string(self, particle):
        '''Build particle decay string from given particle down

        Need to recode this without recursion.
        '''
        dec_string = ' {}'.format(particle.getPDG())

        # Check at least one primary particle daughter exists before diving down a layer
        if (
            particle.getNDaughters() > 0 and
            [(d.isPrimaryParticle() and self._check_status_bit(d.getStatus())) for d in particle.getDaughters()].count(True)
        ):
            dec_string += ' (-->'
            for daughter in particle.getDaughters():
                if daughter.isPrimaryParticle() and self._check_status_bit(daughter.getStatus()):
                    dec_string += self._build_decay_string(daughter)
            dec_string += ' <--)'
        return dec_string

    def _check_status_bit(self, status_bit):
        '''Returns True if conditions are satisfied (not an unusable particle)

        Move this method to preprocessPandas
        '''
        return (
            (status_bit & 1 << 4 == 0) &  # IsVirtual
            (status_bit & 1 << 5 == 0) &  # Initial
            (status_bit & 1 << 6 == 0) &  # ISRPhoton
            (status_bit & 1 << 7 == 0)  # FSRPhoton
        )

    """
    # For grid, needs dev/testing
    def _preproc_decay_string(self, df, LSTM_flag=False):
        # Tokenize the decay string
        token_df = df['decay_str'].apply(self._tokenize_decay_string)
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

    def _tokenize_decay_string(self, decay_str):
        '''Lambda method to call as DataFrame apply during decay string tokenization

        Converts individual decay string into their tokens.
        '''
        # Should make this call tokenize_PDG_code on the split
        tok = self.tokenize.texts_to_sequences(decay_str.split())
        # Convert to single list of tokens
        return [i for sub in tok for i in sub]

    def _init_tokenizer(self):
        '''Initialise tokenizer for processing decay strings'''

        tokenize = text.Tokenizer(
            num_words=self.num_pdg_codes,
            filters='!"#$%&*+,./:;=?@[\]^_`{|}~'
        )
        tokenize.fit_on_texts(evtPdl.pdgTokens)
        return tokenize
    """
