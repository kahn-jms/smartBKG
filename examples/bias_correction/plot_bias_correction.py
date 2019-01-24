#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Plot various bias metrics for the given Ntuple
# James Kahn

import argparse
import os
# from smartBKG.evaluate import PlotPull  # type:ignore
# from smartBKG.evaluate import PlotAsymmetry  # type:ignore
from smartBKG.evaluate import PlotBinomBiasCorr  # type:ignore


def GetCmdArgs():
    parser = argparse.ArgumentParser(
        description='''Compare variables in an Ntuple with threshold cut applied to NN prediciton''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-i', type=str, required=True, nargs='+',
                        help="Path to events to plot", metavar="NTUPLES",
                        dest='in_files')
    parser.add_argument('-m', type=str, required=False,
                        help="Model to plot", metavar="MODEL",
                        dest='model')
    parser.add_argument('-t', type=float, required=False,
                        default=0.5,
                        help="Threshold to apply", metavar="THRESHOLD",
                        dest='threshold')
    parser.add_argument('-b', type=float, required=False, nargs='+',
                        default=[x * 0.1 for x in range(10)],
                        help="Bias thresholds to apply", metavar="BIAS_T",
                        dest='bias_thresholds')
    parser.add_argument('-v', type=str, required=True,
                        help="""
                        Variables file to plot (CSV).
                        """,
                        metavar="VAR", dest='var')
    parser.add_argument('-o', type=str, required=True,
                        help="Output directory to save plots", metavar="OUTPUT",
                        dest='out_dir')
    return parser.parse_args()


if __name__ == '__main__':

    args = GetCmdArgs()
    os.makedirs(args.out_dir, exist_ok=True)

    # Initialise
    plot_binom = PlotBinomBiasCorr(
        files=args.in_files,
        model=args.model,
        var_file=args.var,
        tree='s',
    )

    # Custom cut
    plot_binom.df = plot_binom.df.query('nCleanedTracks__bodr__st2__spand__spabs__bodz__bc__st4__bc <= 12')
    # plot_binom.df = plot_binom.df.query('biasNN_pred <= 0.4')
    # plot_asym.df = plot_asym.df.query('isSignal == 1')

    for b in args.bias_thresholds:
        plot_binom.plot_threshold(args.threshold, b, args.out_dir)
