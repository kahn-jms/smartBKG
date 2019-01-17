#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Plot pull distribution of threshold for Ntuple variables
# James Kahn 2018

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .make_root_compatible import makeRootCompatible

from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1
import argparse  # noqa
import root_pandas  # noqa


class PlotPull():
    ''' Class to load Ntuple and plot pulls for threshold cut '''
    def __init__(self, files, var_file, model=None, tree='variables'):
        self.files = files
        self.model = model
        self.var_file = var_file
        self.tree = tree

        # For multiple models in one files
        if self.model:
            self.threshold_var = 'eventExtraInfo({})'.format(self.model)
        else:
            self.threshold_var = 'eventExtraInfo(smartBKG)'
        self.threshold_var = makeRootCompatible(self.threshold_var)

        self.vars_df = self._load_var_csv(self.var_file)

        self.df = self._load_ntuples(self.files, self.tree)
        self.df = self._bin_vars(self.df)

    def plot_threshold(self, threshold):
        ''' Try plotting cut then pull '''
        for var in self.vars_df['variable']:
            cut_df = self._create_var_counts(self.df, var, threshold)
            cut_df = self._norm_df(cut_df)
            # Could move this call into create_var_counts or move create_var_counts
            # call into calc_diff_metrics is better
            cut_df = self._calc_diff_metrics(cut_df, var)
            self._plot_metrics(cut_df, var, threshold)

    def _norm_df(self, df):
        ''' Normalise df, taking care of errors '''
        # Create error df to normalise first
        err_df = np.sqrt(df)
        # Normalise
        err_df = err_df / (df.max() - df.min())
        df = (df - df.min()) / (df.max() - df.min())

        return pd.concat([df, err_df.add_suffix('_err')], axis=1)

    def _plot_metrics(self, df, var, threshold):
        ''' Plot metrics '''
        plt.figure(figsize=(5, 3.5))
        # First draw variable
        # ax1 = plt.subplot(411, nrows=3)
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        df[[
            '{}_nocut'.format(var),
            '{}_cut'.format(var),
        ]].plot(
            kind='area',
            # bins=30,
            grid=True,
            stacked=False,
            ax=ax1,
            alpha=0.5,
        )
        ax1.legend(['Without cut', 'NN prediction > {}'.format(threshold)])
        # ax1.legend(loc='best')

        # Draw the normed difference
        # ax2 = plt.subplot(414, sharex=ax1)
        ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
        df['{}_diff'.format(var)].plot(
            grid=True,
            ax=ax2
            # sharex=True,
        )

        # Decorate
        ax2.set_xticklabels([
            '{:.4f}'.format(c.mid) for c in df.index.values.categories
        ])
        plt.xlabel(
            self.vars_df.loc[self.vars_df['variable'] == var]['title'].values[0]
        )
        plt.ylabel('Error')
        # plt.axhline(
        #     y=0.,
        #     color='tab:green'
        # )
        plt.show()

    def _calc_diff_metrics(self, df, var):
        ''' Calculate cut metrics '''
        # Normalised difference
        df['{}_diff'.format(var)] = (
            df['{}_nocut'.format(var)] - df['{}_cut'.format(var)]
        ) / (
            df['{}_nocut'.format(var)] + df['{}_cut'.format(var)]
        )
        # Need to scale this now

        return df

    def _create_var_counts(self, df, var, threshold):
        ''' Create df with bin counts before and after threshold cut '''
        # Create bin counts with and without cut
        no_cut = df['{}_bins'.format(var)].value_counts()
        with_cut = df.query(
            '{} > {}'.format(self.threshold_var, threshold)
        )['{}_bins'.format(var)].value_counts()
        # Rename before merging
        no_cut = no_cut.rename('{}_nocut'.format(var))
        with_cut = with_cut.rename('{}_cut'.format(var))

        return pd.concat([no_cut, with_cut], axis=1).fillna(0).sort_index()

    def _bin_vars(self, df):
        ''' First crerate bin counts for each variable '''
        for var, bins in zip(self.vars_df['variable'], self.vars_df['bins']):
            df['{}_bins'.format(var)] = pd.cut(df[var], bins)
        return df

    def _load_ntuples(self, ntuples, tree='variable'):
        ''' Load ntuples into single dataframe '''
        return root_pandas.read_root(ntuples, key=tree)

    def _load_var_csv(self, var_csv):
        ''' Load CSV containing plot variables '''
        return pd.read_csv(var_csv, comment='#')


def GetCmdArgs():
    parser = argparse.ArgumentParser(
        description='''Compare variables in an Ntuple with threshold cut applied to NN prediciton''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-i', type=str, required=True, nargs='+',
                        help="Path to Ntuples to plot", metavar="NTUPLES",
                        dest='in_files')
    parser.add_argument('-m', type=str, required=True,
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
    plot_pull.plot_threshold(0.5)
