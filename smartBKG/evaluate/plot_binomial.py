#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Plot binomial distribution of threshold for Ntuple variables
# James Kahn 2018

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
from .make_root_compatible import makeRootCompatible

from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1
import argparse  # noqa
import root_pandas  # noqa


class PlotBinomial():
    ''' Class to load Ntuple and plot pulls for threshold cut '''
    def __init__(self, files, var_file, model=None, tree='variables'):
        self.files = files
        self.model = model
        self.var_file = var_file
        self.tree = tree

        self.default_bins = 30

        # For multiple models in one files
        if self.model:
            self.threshold_var = 'eventExtraInfo({})'.format(self.model)
        else:
            self.threshold_var = 'eventExtraInfo(smartBKG)'
        self.threshold_var = makeRootCompatible(self.threshold_var)

        self.vars_df = self._load_var_csv(self.var_file)

        self.df = self._load_ntuples(self.files, self.tree)
        # self.df = self._crop_outliers(self.df)
        # Could put this in plot function, then can apply plotting
        # range restriction (query) there
        self.df = self._bin_vars(self.df)

    def plot_threshold(self, threshold, out_dir):
        ''' Try plotting cut then pull '''
        for var in self.vars_df['variable']:
            # First create df with counts of each bin: A (no cut), B (cut)
            cut_df, pre_evts, post_evts = self._create_var_counts(self.df, var, threshold)
            efficiency = post_evts / pre_evts
            # Next add binomial mean, stddev, p-value for A
            cut_df = self._append_binom_stats(cut_df, efficiency)

            self._plot_metrics(
                cut_df,
                var,
                threshold,
                out_dir,
                pre=pre_evts,
                post=post_evts,
            )

    def _append_binom_stats(self, df, eff):
        ''' Add binomial metrics to df '''

        # Manual-ish
        # df['binom_mean'] = df['nocut'] * eff
        # df['binom_stddev'] = np.sqrt(df['nocut'] * eff * (1 - eff))
        # # I don't think this is right, it's a two-sided test
        # df['cut_binom_pval'] = df.apply(
        #     lambda x: binom_test(x['cut'], x['nocut'], eff),
        #     axis=1,
        # )
        # Purely using scipy binom class?
        df['binom_mean'] = binom.mean(df['nocut'], eff)
        df['binom_stddev'] = binom.std(df['nocut'], eff)
        df['cut_binom_pval'] = df.apply(
            lambda x: binom.cdf(x['cut'], x['nocut'], eff),
            axis=1,
        )

        return df

    def _plot_metrics(self, df, var, threshold, out_dir, pre, post):
        ''' Plot metrics '''
        # As a first step need to create a label column from the index
        # Need this for x-axis
        df = df.reset_index()
        if df['index'].dtype.name == 'category':
            df['labels'] = df['index'].apply(lambda x: x.mid).astype(float)
        else:
            df['labels'] = df['index']

        # Cut to the plotting range requested (if requested)
        # Would be nice to put in binning function but then reduces stats for everything
        if not np.isnan(self.vars_df.loc[self.vars_df['variable'] == var]['low_x'].values[0]):
            df = df.query('labels > {}'.format(
                self.vars_df.loc[self.vars_df['variable'] == var]['low_x'].values[0]
            ))
            df = df.query('labels < {}'.format(
                self.vars_df.loc[self.vars_df['variable'] == var]['high_x'].values[0]
            ))

        plt.figure(
            figsize=(6, 5),
            linewidth=1,
        )
        # First draw variable
        # ax1 = plt.subplot(411, nrows=3)
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        df.plot(
            x='labels',
            y=[
                'binom_mean',
                'cut',
            ],
            kind='bar' if (
                self.vars_df.loc[self.vars_df['variable'] == var]['bins'].values[0] == -1
            ) else 'line',
            grid=True,
            stacked=False,
            alpha=0.9,
            ax=ax1,
            yerr={
                'binom_mean': df['binom_stddev'],
            } if (
                self.vars_df.loc[self.vars_df['variable'] == var]['bins'].values[0] == -1
            ) else None,
        )
        # Draw error bands
        if not (
            self.vars_df.loc[self.vars_df['variable'] == var]['bins'].values[0] == -1
        ):
            ax1.fill_between(
                df['labels'],
                df['binom_mean'] + df['binom_stddev'],
                df['binom_mean'] - df['binom_stddev'],
                facecolor='blue',
                alpha=0.3,
            )
            # ax1.fill_between(
            #     df['labels'],
            #     df['{}_cut'.format(var)] + df['{}_cut_err'.format(var)],
            #     df['{}_cut'.format(var)] - df['{}_cut_err'.format(var)],
            #     facecolor='orange',
            #     alpha=0.3,
            # )
        ax1.legend([
            'Expected (binomial)',
            'Observed (NN prediction > {:.2f})'.format(threshold)
            # 'Without cut ({:d})'.format(pre),
            # 'NN prediction > {:.2f} ({:d})'.format(threshold, post)
        ])
        # ax1.legend(loc='best')

        # Draw the p-values now
        # ax2 = plt.subplot(414, sharex=ax1)
        ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax1)
        # ymax = (df['{}_diff'.format(var)] + df['{}_diff_err'.format(var)]).max()
        # ymin = (df['{}_diff'.format(var)] - df['{}_diff_err'.format(var)]).min()
        # ymax = ymax if ymax < 1. else 1.01
        # ymin = ymin if ymin > -1. else -1.01
        df.plot(
            x='labels',
            y='cut_binom_pval',
            grid=True,
            kind='bar' if (
                self.vars_df.loc[self.vars_df['variable'] == var]['bins'].values[0] == -1
            ) else 'line',
            color='red',
            alpha=0.9,
            ax=ax2,
        )
        # Draw error bands
        # if not (
        #     self.vars_df.loc[self.vars_df['variable'] == var]['bins'].values[0] == -1
        # ):
        #     ax2.fill_between(
        #         df['labels'],
        #         df['{}_diff'.format(var)] + df['{}_diff_err'.format(var)],
        #         df['{}_diff'.format(var)] - df['{}_diff_err'.format(var)],
        #         facecolor='red',
        #         alpha=0.3,
        #     )

        # Decorate
        plt.xlabel(
            self.vars_df.loc[self.vars_df['variable'] == var]['title'].values[0]
        )
        ax1.set(ylabel='No. events')
        ax2.set(ylabel='p-value')
        ax2.get_legend().remove()
        ax2.axhline(0.05, color='orange')
        # Use log axes if necessary
        if self.vars_df.loc[
            self.vars_df['variable'] == var
        ]['logscale'].values[0] is True:
            ax1.set_yscale('log')
            ax1.set_xscale('log')
            ax2.set_xscale('log')

        ax2.set_yscale('log')

        # Save output
        plt.savefig(
            os.path.join(out_dir, '{}_{}'.format(var, threshold)) + '.pdf',
            bbox_inches='tight',
            transparent=True,
        )

    def _create_var_counts(self, df, var, threshold):
        ''' Create df with bin counts before and after threshold cut '''
        # Create bin counts with and without cut
        no_cut = df['{}_bins'.format(var)].value_counts()
        with_cut = df.query(
            '{} > {}'.format(self.threshold_var, threshold)
        )['{}_bins'.format(var)].value_counts()
        # Rename before merging
        no_cut = no_cut.rename('nocut')
        with_cut = with_cut.rename('cut')

        return (
            pd.concat([no_cut, with_cut], axis=1).fillna(0).sort_index(),
            no_cut.sum(),
            with_cut.sum(),
        )

    def _bin_vars(self, df):
        ''' Create bin counts for each variable '''
        for var, bins in zip(self.vars_df['variable'], self.vars_df['bins']):
            if bins == -1:
                df['{}_bins'.format(var)] = df[var]
            else:
                df['{}_bins'.format(var)] = pd.cut(
                    df[var],
                    bins if not np.isnan(bins) else self.default_bins,
                )

        return df

    def _crop_outliers(self, df):
        ''' Crop outlying events according to quantiles '''
        low_quant = 0.0001
        high_quant = 0.9999
        for var in self.vars_df['variable']:
            outliers = df[var].quantile([low_quant, high_quant])
            df = df.ix[
                (df[var] < outliers.loc[high_quant]) &
                (df[var] > outliers.loc[low_quant])
            ]
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
    plot_binom = PlotBinomial(
        files=args.in_files,
        model=args.model,
        var_file=args.var,
    )
    plot_binom.plot_threshold(0.5, args.out_dir)
