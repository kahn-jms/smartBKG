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
        description='''Create Ntuple of skimmed B meson list for bias training''',
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

# ROE filters
IPtrack_cut = 'dr < 2 and abs(dz) < 4'  # From stdCharged
gamma_cut = (
    'clusterHypothesis == 5'
    ' and theta > 0.296706 and theta < 2.61799'
    ' and clusterErrorTiming < 1e6'
    ' and [clusterE1E9 > 0.4 or E > 0.075]'
    ' and [[clusterReg == 1 and E > 0.05] or [clusterReg == 2 and E > 0.05] or [clusterReg == 3 and E > 0.075]]'
    ' and abs(clusterTiming) < formula(0.5 * clusterErrorTiming)'
)

# Need to be event level vars as the smartBKG NN cut throws away events only
B_vars = [
    'evtNum',
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
    'extraInfo(FEIProbabilityRank)',
    # Training labels
    'eventExtraInfo(smartBKG)',
]

ROE_vars = [
    # ROE mask vars
    'ROE_E(ROE)',
    'ROE_eextra(ROE)',
    'ROE_neextra(ROE)',
    'WE_MissP(ROE, 0)',
    'WE_MissE(ROE, 0)',
    # Continuum suppression
    'R2',
    'thrustBm',
    'thrustOm',
    'cosTBTO',
    'cosTBz',
    'KSFWVariables(et)',
    'KSFWVariables(mm2)',
    'KSFWVariables(hso00)',
    'KSFWVariables(hso02)',
    'KSFWVariables(hso04)',
    'KSFWVariables(hso10)',
    'KSFWVariables(hso12)',
    'KSFWVariables(hso14)',
    'KSFWVariables(hso20)',
    'KSFWVariables(hso22)',
    'KSFWVariables(hso24)',
    'KSFWVariables(hoo0)',
    'KSFWVariables(hoo1)',
    'KSFWVariables(hoo2)',
    'KSFWVariables(hoo3)',
    'KSFWVariables(hoo4)',
    'CleoConeCS(1, ROE)',
    'CleoConeCS(2, ROE)',
    'CleoConeCS(3, ROE)',
    'CleoConeCS(4, ROE)',
    'CleoConeCS(5, ROE)',
    'CleoConeCS(6, ROE)',
    'CleoConeCS(7, ROE)',
    'CleoConeCS(8, ROE)',
    'CleoConeCS(9, ROE)',
]
B_vars += ROE_vars


if __name__ == '__main__':

    args = GetCmdArgs()
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    # Load the input skim file
    path = ma.create_path()
    ma.inputMdstList('MC9', filelist=[], path=path)
    # ma.inputMdstList('default', filelist=[], path=path)

    ma.applyCuts(particle_list, 'nCleanedTracks(dr<2 and abs(dz)<4) <= 12', path=path)

    # Build some event specific ROE and continuum vars
    ma.buildRestOfEvent(particle_list, path=path)
    ROEMask = ('ROE', IPtrack_cut, gamma_cut)
    ma.appendROEMasks(particle_list, [ROEMask], path=path)
    ma.buildContinuumSuppression(particle_list, roe_mask='ROE', path=path)

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
