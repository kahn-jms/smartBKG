#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module to apply keras training to MCParticles
# Returns prediction value and saves as EventExtraInfo
# James Kahn 2018

import os
import pandas as pd
import basf2 as b2

# make the Belle2 namespace available
from ROOT import Belle2
from smartBKG.preprocessing import MCParticlesPreprocManager

# config = tf.ConfigProto(intra_op_parallelism_threads=1,
#                         inter_op_parallelism_threads=1,
#                         allow_soft_placement=True,
#                         device_count = {'CPU': 1})
# session = tf.Session(config=config)
# K.set_session(session)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class NNApplyModule(b2.Module):
    ''' Apply the given keras model to MCParticles

    Supports a limited set of model inputs and only binary output.
    Setting the extra_info_var saves the output prediction to EventExtraInfo,
    users then need to explicitly add this branch to the mdst output to save this.
    '''
    def __init__(
        self,
        model_file,
        model_type,
        threshold,
        extra_info_var=None,
    ):
        super().__init__()
        self.model_file = model_file
        self.model_type = model_type
        self.threshold = threshold
        self.extra_info_var = extra_info_var

        # Require model type to be one of supported types
        supported_models = [
            'combined-LSTM',
            'combined-wideCNN',
            'particles',
            'decstr-LSTM'
            'decstr-wideCNN'
        ]
        # Eventually want to calculate this from loaded model inputs
        assert model_type in supported_models, b2.B2ERROR(
            'model_type {} not supported, must be one of: {}'.format(self.model_type, supported_models)
        )

        # Initialise preprocessing class
        self.cap = MCParticlesPreprocManager()

        # Set other flags for preproc calls
        self.LSTM_flag = self.model_type in ['combined-LSTM', 'decstr-LSTM']

    def initialize(self):
        """ Initialise module before any events are processed"""
        # Create a StoreArray to save predictions to
        self.e_e_info = Belle2.PyStoreObj('EventExtraInfo')
        self.e_e_info.registerInDataStore()

        # Apparently needed to make thread safe
        self.model = None

    def event(self):
        """Return match of event number to input list"""
        if self.model is None:
            from keras.models import load_model
            self.model = load_model(self.model_file)
            # Required to be initialised before multithreading
            self.model._make_predict_function()
            # And if needed infer the name of the extra info var
            if self.extra_info_var is None:
                self.extra_info_var = self.model.name
                print('EventExtraInfo variable set to: {}'.format(self.extra_info_var))
            print('Initialised model')

        # Need to create the eventExtraInfo entry for each event
        if self.extra_info_var:
            self.e_e_info.create()

        mcplist = Belle2.PyStoreArray("MCParticles")

        # Can get away with list because we don't use arrayIndex
        event_list = []
        data_dict = {}

        if self.model_type in ['combined-LSTM', 'combined-wideCNN', 'particles']:
            # Create particle vars
            for mcp in mcplist:
                if mcp.isPrimaryParticle():
                    # Check mc particle is useable
                    if not self.cap.check_status_bit(mcp.getStatus()):
                        continue

                    four_vec = mcp.get4Vector()
                    prod_vec = mcp.getProductionVertex()
                    mother = mcp.getMother()
                    motherPDG = 0
                    if mother:
                        motherPDG = mother.getPDG()

                    event_list.append({
                        'PDG': mcp.getPDG(),
                        # 'mass': mcp.getMass(),
                        'charge': mcp.getCharge(),
                        'energy': mcp.getEnergy(),
                        'prodTime': mcp.getProductionTime(),
                        'x': prod_vec.x(),
                        'y': prod_vec.y(),
                        'z': prod_vec.z(),
                        'px': four_vec.Px(),
                        'py': four_vec.Py(),
                        'pz': four_vec.Pz(),
                        'motherPDG': motherPDG,
                    })
            # Convert to a dataframe for preprocessing
            event_df = pd.DataFrame(event_list)

            # Perform event preprocessing and get back the numpy array of particles
            data_dict['particle_input'], data_dict['pdg_input'], data_dict['mother_pdg_input'] = self.cap.preproc_single_whole_decay(event_df)
            # Need to do reshaping here I think
            # x_arr = np.reshape(x_arr, (1, x_arr.shape[0], x_arr.shape[1]))
            # pdg_arr = np.reshape(pdg_arr, (1, pdg_arr.shape[0]))
            # mother_pdg_arr = np.reshape(mother_pdg_arr, (1, mother_pdg_arr.shape[0]))

        # Build decay string
        if self.model_type in ['combined-LSTM', 'combined-wideCNN', 'decstr-LSTM', 'decstr-wideCNN']:
            # First particle is always the top of the decay chain
            if len(mcplist) > 0:
                MCdecay_string = self.cap.build_decay_string(mcplist[0])

            data_dict['decay_input'] = self.cap.preproc_single_decay_string(MCdecay_string, self.LSTM_flag)

        # Outputs pass probability
        pred = self.model.predict(data_dict)

        b2.B2INFO('Pass probability:\t{}'.format(pred[0][0]))
        b2.B2INFO('Passes threshold:\t{}'.format(int(pred[0][0] >= self.threshold)))

        # Save the pass probability to EventExtraInfo
        if self.extra_info_var:
            self.e_e_info.addExtraInfo(self.extra_info_var, pred[0][0])

        # Module returns bool of whether prediciton passes threshold for use in basf2 path flow control
        self.return_value(int(pred[0][0] >= self.threshold))
