#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Plot various bias metrics for the given Ntuple
# James Kahn

import argparse
import os
from smartBKG.evaluate import PlotPull  # type:ignore


def GetCmdArgs():
    parser = argparse.ArgumentParser(
        description='''Compare variables in an Ntuple with threshold cut applied to NN prediciton''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-i', type=str, required=True, nargs='+',
                        help="Path to Ntuples to plot", metavar="NTUPLES",
                        dest='in_files')
    parser.add_argument('-m', type=str, required=False,
                        help="Model to plot", metavar="MODEL",
                        dest='model')
    parser.add_argument('-t', type=float, required=False, nargs='+',
                        default=[x * 0.1 for x in range(10)],
                        help="Thresholds to apply", metavar="THRESHOLDS",
                        dest='thresholds')
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
    plot_pull = PlotPull(
        files=args.in_files,
        model=args.model,
        var_file=args.var,
    )
    for t in args.thresholds:
        plot_pull.plot_threshold(t, args.out_dir)
