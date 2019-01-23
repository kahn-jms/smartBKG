#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import importlib
import pandas as pd
from smartBKG.evaluate import makeRootCompatible

from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1
import argparse  # noqa
import root_pandas  # noqa


def getCmdArgs():
    parser = argparse.ArgumentParser(
        description='''Preprocess data for NN input and save.''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', type=str, required=True, nargs='+',
                        help="Path to pandas h5 input.", metavar="INPUT",
                        dest='in_file')
    parser.add_argument('-t', type=float, required=False, default=0.5,
                        help="Threshold for NN output", metavar="THRESHOLD",
                        dest='threshold')
    parser.add_argument('-o', type=str, required=True,
                        help="Directory to save training output", metavar="OUTPUT",
                        dest='out_dir')
    return parser.parse_args()


gamma_cut = (
    'clusterHypothesis == 5'
    ' and theta > 0.296706 and theta < 2.61799'
    ' and clusterErrorTiming < 1e6'
    ' and [clusterE1E9 > 0.4 or E > 0.075]'
    ' and [[clusterReg == 1 and E > 0.05] or [clusterReg == 2 and E > 0.05] or [clusterReg == 3 and E > 0.075]]'
    ' and abs(clusterTiming) < formula(0.5 * clusterErrorTiming)'
)
train_vars = [
    'nTracks',
    'nCleanedTracks(dr<2 and abs(dz)<4)',
    'nCleanedECLClusters({})'.format(gamma_cut),
    'nECLClusters',
    'nKLMClusters',
    'Ecms',
    'Eher',
    'Eler',
]
root_train_vars = []
for s in train_vars:
    root_train_vars.append(makeRootCompatible(s))

label_var = 'eventExtraInfo(smartBKG)'
root_label_var = makeRootCompatible(label_var)


if __name__ == '__main__':

    args = getCmdArgs()
    os.makedirs(args.out_dir, exist_ok=True)

    df_list = []
    for f in args.in_file:
        df_list.append(root_pandas.read_root(f))
    df = pd.concat(df_list)

    # Preprocess
    X = df[root_train_vars]
    y = df[root_label_var]

    # Threshold labels to 0 or one depending on input threshold
    # should be a builtin to do this but I don't know it
    y = y.assign(label=0)
    # Drop the original labels
    y = y['label'].mask(y[root_label_var] > args.threshold, 1)

    
