#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Steering file to read a skim file and save the B meson lists with the NN model predictions to ntuple
# James Kahn 2018

import os
import basf2 as b2

from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1
import argparse  # noqa
import modularAnalysis as ma  # noqa
from smartBKG.apply import NNApplyModule  # noqa


def GetCmdArgs():
    parser = argparse.ArgumentParser(
        description='''Create Ntuple of MC B meson list for bias training''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-m', type=str, required=True,
                        help="Path to trained model", metavar="MODEL",
                        dest='model')
    parser.add_argument('-o', type=str, required=True,
                        help="Output filename", metavar="OUTPUT",
                        dest='out_file')
    return parser.parse_args()


particle_list = 'B+:generic'
# particle_list = 'B0:generic'

# Need to be event level vars as the smartBKG NN cut throws away events only
B_vars = [
    'nTracks',
    'nCleanedTracks(dr<2 and abs(dz)<4)',
    'nECLClusters',
    'nKLMClusters',
    # 'R2EventLevel',
    'isSignal',
    'charge',
    'Mbc',
    'deltaE',
    # 'missingMass',
    'M',
    'useCMSFrame(p)',
    'extraInfo(SignalProbability)',
    # Training labels
    'eventExtraInfo(smartBKG)',
]


if __name__ == '__main__':

    args = GetCmdArgs()
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    # Load the input skim file
    path = ma.create_path()
    ma.inputMdstList('MC9', filelist=[], path=path)
    # ma.inputMdstList('default', filelist=[], path=path)

    ma.applyCuts(particle_list, 'nCleanedTracks(dr<2 and abs(dz)<4) <= 12', path=path)

    # Apply the smartBKG NN model
    # Will use extraInfo saved as training labels later,
    # need to be flattened before training to 0 or 1
    NNApplyModule_m = NNApplyModule(
        model_file=args.model,
        model_type='combined-wideCNN',
        threshold=0.,
        # threshold=args.threshold,
        extra_info_var='smartBKG'

    )
    # dead_path = b2.create_path()
    # NNApplyModule_m.if_false(dead_path)

    path.add_module(NNApplyModule_m)

    # We'll keep just one Btag per event and train on that
    ma.rankByHighest(
        particle_list,
        'extraInfo(SignalProbability)',
        outputVariable='FEIProbabilityRank',
        numBest=1,
        path=path
    )

    # Write output
    # ma.variablesToNTuple(
    ma.variablesToNtuple(
        decayString=particle_list,
        variables=B_vars,
        filename=args.out_file,
        path=path,
    )

    b2.process(path)
    print(b2.statistics)
