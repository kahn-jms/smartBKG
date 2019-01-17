#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Plot Ntuple variables for different NN threshold cuts
# James Kahn 2018

import os
import make_root_compatible as mrc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1
import argparse  # noqa
import root_pandas  # noqa


class PlotBias():
    ''' Class to load Ntuple and plot vars for different threshold cuts '''
    def __init__(self, in_files, model, tree='variables'):
        self.in_files = in_files
        self.model = model
        self.tree = tree

        self.color_args = dict(
            histtype='stepfilled',
            density=False,
            # alpha=0.1,
            # bins='auto',
        )

        self.threshold_var = 'eventExtraInfo({})'.format(self.model)
        self.threshold_var = mrc.makeRootCompatible(self.threshold_var)

        self.in_df = self._load_ntuples(self.in_files, self.tree)
        # We can straight away drop the dead __weight__
        self.in_df = self.in_df.drop(
            ['__weight__', 'charge', 'evtNum'],
            axis=1
        )
        print(self.in_df.head())

    def plot_comparisons(
        self,
        thresholds,
        out_dir,
    ):
        ''' Plot all comparisons and save to file '''
        for col in self.in_df.drop(self.threshold_var, axis=1).columns:
            print('plotting column:', col)
            out_file = os.path.join(out_dir, '{}.pdf'.format(col))
            self._plot_single_var_thresholds(self.in_df, col, self.threshold_var, thresholds, out_file)

        # Also plot the thrshold var for comparison
        # out_file = os.path.join(out_dir, '{}.pdf'.format(mrc.invertMakeRootCompatible(self.threshold_var)))
        # self._plot_single_var(self.in_df, self.threshold_var, out_file)

    # def _plot_single_var(self, df, var, out_file):
    #     ''' Plot single variable from df '''

    def _plot_single_var_thresholds(self, df, x, cut_var, thresholds, out_file):
        ''' Plot x, y columns with different threshold cuts applied

        Should really plot all vars per cut, and in plot_comparisons keep dict of
        different plots, then save all.
        '''
        # Remove outliers
        # q = df[x].quantile(0.99)
        # df = df[df[x] < q]
        plt.figure()
        df['binned'] = pd.cut(df[x], 100)
        counts = df['binned'].value_counts()
        norm_counts = (counts - counts.mean()) / (counts.max() - counts.min())
        # sns.distplot(
        #     df[x],
        #     # hist=False,
        #     # rug=True,
        #     # kde_kws={'shade': True},
        #     # label='> {:.2f} ({:.2f})'.format(cut, (cut_df.shape[0] / df.shape[0])),
        #     label='> {:.2f} ({:.2f})'.format(0.0, 1.0),
        # )
        for cut in thresholds:
            cut_df = df.query('{} > {}'.format(cut_var, cut))
            # cut_df = df.query('{} < {}'.format(cut_var, cut))
            # Bin the data for comparison
            # cut_df['binned'] = pd.cut(cut_df[x], 100)
            uni = cut_df['binned'].value_counts()
            # uni = pd.concat([uni, df['binned'].value_counts().rename('orig')], axis=1)
            uni = (uni - uni.mean()) / (uni.max() - uni.min())
            print(uni.head())
            # diff = uni.divide(np.sqrt(counts))
            # diff = uni.divide(counts)
            diff = uni.sub(norm_counts).divide(np.sqrt(np.abs(norm_counts)))
            diff = diff.dropna()
            print(diff.head())
            ax = diff.plot()

            # uni['cut_df'] = cut_df['binned'].value_counts()
            # sns.distplot(
            #     cut_df[x],
            #     # hist=False,
            #     # rug=True,
            #     # kde_kws={'shade': True},
            #     # label='> {:.2f} ({:.2f})'.format(cut, (cut_df.shape[0] / df.shape[0])),
            #     label='> {:.2f} ({:.2f})'.format(cut, 1 - (cut_df.shape[0] / df.shape[0])),
            # )
            # plt.plot(
            #     uni['df'],
            #     uni['cut_df'],
            #     # label='{} > {}'.format(self.model, cut),
            #     # label='> {:.1f}'.format(cut),
            #     # **self.color_args,
            # )
            # plt.hist(
            #     diff,
            #     # cut_df['binned'].value_counts(),
            #     # label='{} > {}'.format(self.model, cut),
            #     label='> {:.1f}'.format(cut),
            #     bins=100,
            #     **self.color_args,
            # )

        ax.set_xticklabels(['{:.4f}'.format(c.mid) for c in cut_df['binned'].cat.categories])
        plt.legend(loc='best')
        plt.xlabel(mrc.invertMakeRootCompatible(x))
        # if self.color_args['density']:
        #     plt.ylabel('nEvents (normed)')
        # else:
        #     plt.ylabel('nEvents')
        plt.ylabel('Pull')

        plt.savefig(
            out_file,
            bbox_inches='tight',
            transparent=True,
        )

    def _load_ntuples(self, ntuples, tree='variables'):
        ''' Load ntuples into single dataframe '''
        return root_pandas.read_root(ntuples, key=tree)


def GetCmdArgs():
    parser = argparse.ArgumentParser(
        description='''Compare variables in an Ntuple with different threshold cuts applied to NN prediciton''',
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
    parser.add_argument('-o', type=str, required=True,
                        help="Output directory to save plots", metavar="OUTPUT",
                        dest='out_dir')
    return parser.parse_args()


if __name__ == '__main__':

    args = GetCmdArgs()
    os.makedirs(args.out_dir, exist_ok=True)

    plot_bias = PlotBias(
        in_files=args.in_files,
        model=args.model,
    )
    plot_bias.plot_comparisons(
        args.thresholds,
        args.out_dir,
    )
