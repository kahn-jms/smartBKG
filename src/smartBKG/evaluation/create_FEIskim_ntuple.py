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


B_vars = [
    # Event wide vars
    'evtNum',
    'nTracks',
    'nCleanedTracks(dr<2 and abs(dz)<4)',
    'extraInfo(SignalProbability)',
    # B meson vars
    'isSignal',
    'charge',
    'Mbc',
    'deltaE'
]


if __name__ == '__main__':

    args = GetCmdArgs()
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    # Add model specific vars to output Ntuple
    for model in args.models:
        B_vars.append('eventExtraInfo({})'.format(model))

    path = ma.create_path()
    ma.inputMdstList('default', filelist=[], path=path)

    ma.variablesToNTuple(
        'B+:generic',
        B_vars,
        filename=args.out_file,
        path=path,
    )

    b2.process(path)
    print(b2.statistics)
