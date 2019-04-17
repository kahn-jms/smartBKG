#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Plot binomial distribution of threshold for Ntuple variables with bias correction
# James Kahn 2019

import pandas as pd
from smartBKG.evaluate import PlotBinomial


class PlotBinomReweight(PlotBinomial):
    ''' Class to plot binomial dist of bias corrections '''
    def __init__(
        self,
        files,
        var_file,
        model=None,
        tree=None,
        biasNN_var='biasNN_pred'
    ):
        super().__init__(files, var_file, model, tree)
        self.biasNN_var = biasNN_var

    def plot_threshold(self, threshold, out_dir):
        ''' Plot cut '''
        bin_weights = self._collect_bin_weights(self.df, threshold)
        for var in self.vars_df['variable']:
            # First create df with counts of each bin: A (no cut), B (cut)
            cut_df, pre_evts, post_evts = self._create_var_counts(
                self.df,
                var,
                threshold,
                bin_weights,
            )
            efficiency = post_evts / pre_evts
            # Collect bin weights for that var
            # bin_weights = self._collect_weights(
            #     self.df,
            #     cut_df,
            #     var,
            #     threshold,
            #     efficiency,
            # )
            # Weight the cut bins
            # print(cut_df.head())
            # print(bin_weights.head())
            # cut_df['cut'] = cut_df['cut'] * bin_weights['weights']
            # print(cut_df.head())

            # Update efficiency
            # post_evts = cut_df['cut'].sum()
            # efficiency = post_evts / pre_evts

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

    def _collect_bin_weights(
        self,
        df,
        threshold,
        biasNN_var='biasNN_pred'
    ):
        ''' Rewieght dataframe based on biasNN output '''
        bin_var = '{}_bins'.format(biasNN_var)
        # total_nocut = df.shape[0]
        # total_withcut = df.query(
        #     '{} > {}'.format(self.threshold_var, threshold)
        # ).shape[0]
        # total_efficiency = total_withcut / total_nocut
        # Can't get this to work atm
        # bin_weights = pd.DataFrame(index=df[bin_var].unique())
        # bin_weights = pd.DataFrame(index=cut_df.index)
        # bin_weights['weights'] = 1.

        pass_cut = df.query(
            '{} > {}'.format(self.threshold_var, threshold)
        )[bin_var].value_counts()
        fail_cut = df.query(
            '{} <= {}'.format(self.threshold_var, threshold)
        )[bin_var].value_counts()

        # Normalise both
        # Ideally do this to retain as many events as possible
        # Cant think of a nice way at this hour
        # pass_cut = pass_cut / pass_cut.sum()
        # fail_cut = fail_cut / fail_cut.sum()

        # Calculate weight factor
        # bin_weights = fail_cut / pass_cut
        bin_weights = (fail_cut + pass_cut) / pass_cut
        # Rescale to be between 0 and 1
        bin_weights = (bin_weights - bin_weights.min()) / (bin_weights.max() - bin_weights.min())
        print('Mean:', bin_weights.mean())
        print(bin_weights)
        return bin_weights

    def _create_var_counts(self, df, var, threshold, bin_weights):
        ''' Create df with bin counts before and after threshold cut '''
        # Create bin counts with and without cut
        no_cut = df['{}_bins'.format(var)].value_counts()
        with_cut = df.query(
            '{} > {}'.format(self.threshold_var, threshold)
        )
        # Reweight first
        with_cut = self._reweight_biasNN(with_cut, bin_weights)
        with_cut = with_cut['{}_bins'.format(var)].value_counts()

        # Rename before merging
        no_cut = no_cut.rename('nocut')
        with_cut = with_cut.rename('cut')

        return (
            pd.concat([no_cut, with_cut], axis=1).fillna(0).sort_index(),
            no_cut.sum(),
            with_cut.sum(),
        )

    def _reweight_biasNN(self, df, bin_weights, biasNN_var='biasNN_pred'):
        bin_var = '{}_bins'.format(biasNN_var)
        weighted_list = []
        for b in bin_weights.index:
            weighted_list.append(
                df[df[bin_var] == b].sample(frac=bin_weights.loc[b])
            )

        return pd.concat(weighted_list)
        # for b in df[bin_var].unique():
        #     pred_pass_count = df[
        #         (df[bin_var] == b) &
        #         (df[self.threshold_var] > threshold)
        #     ][biasNN_var].sum()
        #     fail_pass_count = df[
        #         (df[bin_var] == b) &
        #         (df[self.threshold_var] <= threshold)
        #     ][biasNN_var].sum()
        #     bin_weights.loc[b]['weights'] = (
        #         float(fail_pass_count) / (efficiency * pred_pass_count)
        #     )

        # return pd.DataFrame.from_dict(
        #     bin_weights,
        #     orient='index',
        #     columns=['weights']
        # )
