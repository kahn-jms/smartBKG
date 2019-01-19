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


class PlotAsymmetry():
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
        self.df = self._bin_vars(self.df)

    def plot_threshold(self, threshold, out_dir):
        ''' Try plotting cut then pull '''
        for var in self.vars_df['variable']:
            cut_df, pre_evts, post_evts = self._create_var_counts(self.df, var, threshold)
            # Could move this call into create_var_counts or move create_var_counts
            # call into calc_diff_metrics is better
            cut_df = self._add_stat_err(cut_df)
            # comb_df = self._norm_df(cut_df, err_df)
            cut_df = self._calc_diff_metrics(cut_df, var)
            print(cut_df.head())
            self._plot_metrics(
                cut_df,
                var,
                threshold,
                out_dir,
                pre=pre_evts,
                post=post_evts,
            )

    def _add_stat_err(self, df):
        ''' Calcualte statistical uncertainty '''
        err_df = np.sqrt(df)
        return pd.concat([df, err_df.add_suffix('_err')], axis=1)

    # def _norm_df(self, df, err_df):
    #     ''' Normalise df, taking care of errors '''
    #     # Normalise
    #     # err_df = df.filter(regex=('.*_err'))
    #     err_df = err_df / (df.max() - df.min())

    #     # non_err_df = df.select(lambda col: not col.endswith('_err'), axis=1)
    #     df = (df - df.min()) / (df.max() - df.min())

    #     # return pd.concat([df, err_df], axis=1)
    #     return pd.concat([df, err_df.add_suffix('_err')], axis=1)

    def _plot_metrics(self, df, var, threshold, out_dir, pre, post):
        ''' Plot metrics '''
        # As a first step need to create a label column from the index
        df = df.reset_index()
        if df['index'].dtype.name == 'category':
            df['labels'] = df['index'].apply(lambda x: x.mid).astype(float)
        else:
            df['labels'] = df['index']

        # Cut to the plotting range requested (if requested)
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
                '{}_nocut'.format(var),
                '{}_cut'.format(var),
            ],
            kind='bar' if (
                self.vars_df.loc[self.vars_df['variable'] == var]['bins'].values[0] == -1
            ) else 'line',
            # bins=30,
            grid=True,
            stacked=False,
            alpha=0.9,
            ax=ax1,
            # xlim=(
            #     self.vars_df.loc[self.vars_df['variable'] == var]['low_x'].values[0],
            #     self.vars_df.loc[self.vars_df['variable'] == var]['high_x'].values[0],
            # ) if not np.isnan(self.vars_df.loc[self.vars_df['variable'] == var]['low_x'].values[0])
            # else None,
            yerr=df[[
                '{}_nocut_err'.format(var),
                '{}_cut_err'.format(var),
            ]].values.T if (
                self.vars_df.loc[self.vars_df['variable'] == var]['bins'].values[0] == -1
            ) else None,
        )
        # Draw error bands
        if not (
            self.vars_df.loc[self.vars_df['variable'] == var]['bins'].values[0] == -1
        ):
            ax1.fill_between(
                df['labels'],
                df['{}_nocut'.format(var)] + df['{}_nocut_err'.format(var)],
                df['{}_nocut'.format(var)] - df['{}_nocut_err'.format(var)],
                facecolor='blue',
                alpha=0.3,
            )
            ax1.fill_between(
                df['labels'],
                df['{}_cut'.format(var)] + df['{}_cut_err'.format(var)],
                df['{}_cut'.format(var)] - df['{}_cut_err'.format(var)],
                facecolor='orange',
                alpha=0.3,
            )
        ax1.legend([
            'Without cut ({:d})'.format(pre),
            'NN prediction > {:.2f} ({:d})'.format(threshold, post)
        ])
        # ax1.legend(loc='best')

        # Draw the normed difference
        # ax2 = plt.subplot(414, sharex=ax1)
        ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax1)
        ymax = (df['{}_diff'.format(var)] + df['{}_diff_err'.format(var)]).max()
        ymin = (df['{}_diff'.format(var)] - df['{}_diff_err'.format(var)]).min()
        ymax = ymax if ymax < 1. else 1.01
        ymin = ymin if ymin > -1. else -1.01
        df.plot(
            x='labels',
            y='{}_diff'.format(var),
            grid=True,
            kind='bar' if (
                self.vars_df.loc[self.vars_df['variable'] == var]['bins'].values[0] == -1
            ) else 'line',
            # sharex=True,
            color='red',
            alpha=0.9,
            ax=ax2,
            # xlim=(
            #     self.vars_df.loc[self.vars_df['variable'] == var]['low_x'].values[0],
            #     self.vars_df.loc[self.vars_df['variable'] == var]['high_x'].values[0],
            # ) if not np.isnan(
            #     self.vars_df.loc[self.vars_df['variable'] == var]['low_x'].values[0]
            # )
            # else None,
            # ylim=(-1, 1),
            # ylim=(ymin, ymax),
            yerr=df[[
                '{}_diff_err'.format(var),
            ]].values.T if (
                self.vars_df.loc[self.vars_df['variable'] == var]['bins'].values[0] == -1
            ) else None,
        )
        # Draw error bands
        if not (
            self.vars_df.loc[self.vars_df['variable'] == var]['bins'].values[0] == -1
        ):
            ax2.fill_between(
                df['labels'],
                df['{}_diff'.format(var)] + df['{}_diff_err'.format(var)],
                df['{}_diff'.format(var)] - df['{}_diff_err'.format(var)],
                facecolor='red',
                alpha=0.3,
            )

        # Decorate
        # ax2.set_xticklabels([
        #     '{:.4f}'.format(c.mid) for c in df.index.values.categories
        #     # '{:d}'.format(int(c.left)) for c in df.index.values.categories  # For discrete vars
        # ])
        plt.xlabel(
            self.vars_df.loc[self.vars_df['variable'] == var]['title'].values[0]
        )
        ax1.set(ylabel='No. events (normalised)')
        ax2.set(ylabel='Asymmetry')
        ax2.get_legend().remove()
        # Use log axes if necessary
        if self.vars_df.loc[
            self.vars_df['variable'] == var
        ]['logscale'].values[0] is True:
            ax1.set_yscale('log')
            ax1.set_xscale('log')
            ax2.set_xscale('log')

        # Save output
        plt.savefig(
            os.path.join(out_dir, '{}_{}'.format(var, threshold)) + '.pdf',
            # os.path.join(out_dir, var) + '.pdf',
            bbox_inches='tight',
            transparent=True,
        )

    def _calc_diff_metrics(self, df, var, normed=False):
        ''' Calculate cut metrics
        Input:
            Dataframe of two distributions and their errors
        Output:
            Dataframe of asymmetry (a - b) / (a + b) - c
            where c is the adjustment constant to account for a and b
            not being normalised
        '''
        # Normalised asymmetry
        num = df['{}_nocut'.format(var)] - df['{}_cut'.format(var)]
        den = df['{}_nocut'.format(var)] + df['{}_cut'.format(var)]
        df['{}_diff'.format(var)] = num / den

        # Now subtract adjustment due to a and b not being normalised
        if not normed:
            adj = df['{}_cut'.format(var)].sum() / df['{}_nocut'.format(var)].sum()
            adj = ((1 - adj) / (1 + adj))
            print(adj)
            df['{}_diff'.format(var)] = df['{}_diff'.format(var)] - adj
        # Now need to propagate error (wrong)
        # First get error for numerator and denominator (same, quadrature)
        # df['{}_diff_err'.format(var)] = np.sqrt(
        #     df['{}_nocut_err'.format(var)]**2 + df['{}_cut_err'.format(var)]**2
        # )
        # # Then handle the division
        # df['{}_diff_err'.format(var)] = np.abs(df['{}_diff'.format(var)]) * np.sqrt(
        #     (df['{}_diff_err'.format(var)] / num)**2 +
        #     (df['{}_diff_err'.format(var)] / den)**2
        # )
        df['{}_diff_err'.format(var)] = self._calc_variance(
            df['{}_nocut'.format(var)],
            df['{}_nocut_err'.format(var)],
            df['{}_cut'.format(var)],
            df['{}_cut_err'.format(var)],
        )

        return df

    def _calc_variance(self, a, a_err, b, b_err):
        ''' Very specific calculation for standard error

        Input:
            Series of two distributions and their errors
        Output:
            Series of error of the asymmetry (a - b) / (a + b)
        '''
        asym_err = np.sqrt(
            (
                ((2. * b) / (a + b)**2)**2 * a_err**2
            ) + (
                ((-2. * a) / (a + b)**2)**2 * b_err**2
            )
        )

        return asym_err

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
    plot_asym = PlotAsymmetry(
        files=args.in_files,
        model=args.model,
        var_file=args.var,
    )
    plot_asym.plot_threshold(0.5, args.out_dir)
