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


def GetCmdArgs():
    parser = argparse.ArgumentParser(
        description='''Create Ntuple of skimmed B meson list''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-m', type=str, required=True, nargs='+',
                        help="Model predictions to save from EventExtraInfo", metavar="MODELS",
                        dest='models')
    parser.add_argument('-o', type=str, required=True,
                        help="Output filename", metavar="OUTPUT",
                        dest='out_file')
    return parser.parse_args()


particle_list = 'B+:generic'
# particle_list = 'B0:generic'

B_vars = [
    # Event wide vars
    'evtNum',
    'nTracks',
    'nCleanedTracks(dr<2 and abs(dz)<4)',
    'Ecms',
    'EventType',
    'IPX',
    'IPY',
    'IPZ',
    # B meson vars
    'isSignal',
    'charge',
    'Mbc',
    'deltaE',
    # 'missingMass',
    'M',
    'useCMSFrame(p)',
    'extraInfo(SignalProbability)',
    # MC variables
    'mcE',
    'mcP',
    'mcPX',
    'mcPY',
    'mcPZ',
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

# Rest of event filters
IPtrack_cut = 'dr < 2 and abs(dz) < 4'  # From stdCharged
gamma_cut = (
    'clusterHypothesis == 5'
    ' and theta > 0.296706 and theta < 2.61799'
    ' and clusterErrorTiming < 1e6'
    ' and [clusterE1E9 > 0.4 or E > 0.075]'
    ' and [[clusterReg == 1 and E > 0.05] or [clusterReg == 2 and E > 0.05] or [clusterReg == 3 and E > 0.075]]'
    ' and abs(clusterTiming) < formula(0.5 * clusterErrorTiming)'
)


if __name__ == '__main__':

    args = GetCmdArgs()
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    # Add model specific vars to output Ntuple
    for model in args.models:
        B_vars.append('eventExtraInfo({})'.format(model))

    # Load the input skim file
    path = ma.create_path()
    ma.inputMdstList('MC9', filelist=[], path=path)

    # Build some event specific ROE and continuum vars
    ma.buildRestOfEvent(particle_list, path=path)
    ROEMask = ('ROE', IPtrack_cut, gamma_cut)
    ma.appendROEMasks(particle_list, [ROEMask], path=path)
    ma.buildContinuumSuppression(particle_list, roe_mask='ROE', path=path)

    B_vars += ROE_vars
    print(B_vars)

    # Then choose one candidate per event
    # Dont' need to, want to view changes to candidates overall
    # But let's try for fun
    ma.rankByHighest(particle_list, 'extraInfo(SignalProbability)',
                     outputVariable='FEIProbabilityRank', path=path)
    ma.applyCuts(particle_list, 'extraInfo(FEIProbabilityRank) == 1', path=path)

    # Write output
    ma.variablesToNtuple(
        particle_list,
        B_vars,
        filename=args.out_file,
        path=path,
    )

    b2.process(path)
    print(b2.statistics)
