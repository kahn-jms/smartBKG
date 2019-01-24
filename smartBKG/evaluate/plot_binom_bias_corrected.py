#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Plot binomial distribution of threshold for Ntuple variables with bias correction
# James Kahn 2019

import pandas as pd
from smartBKG.evaluate import PlotBinomial


class PlotBinomBiasCorr(PlotBinomial):
    ''' Class to plot binomial dist of bias corrections '''
    def __init__(
        self,
        files,
        var_file,
        model=None,
        tree='variables',
        biasNN_var='biasNN_pred'
    ):
        super().__init__(files, var_file, model, tree)
        self.biasNN_var = biasNN_var

    def plot_threshold(self, threshold, bias_threshold, out_dir):
        ''' Plot cut '''
        for var in self.vars_df['variable']:
            # First create df with counts of each bin: A (no cut), B (cut)
            cut_df, pre_evts, post_evts = self._create_var_counts(
                self.df,
                var,
                threshold,
                bias_threshold,
            )
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

    def _create_var_counts(self, df, var, threshold, bias_threshold=None):
        ''' Create df with bin counts before and after threshold cut '''
        # Create bin counts with and without cut
        no_cut = df['{}_bins'.format(var)].value_counts()
        with_cut = df.query(
            '{} > {}'.format(self.threshold_var, threshold)
        )

        if bias_threshold:
            with_cut = with_cut.query(
                '{} < {}'.format(self.biasNN_var, bias_threshold)
            )
        with_cut = with_cut['{}_bins'.format(var)].value_counts()

        # Rename before merging
        no_cut = no_cut.rename('nocut')
        with_cut = with_cut.rename('cut')

        return (
            pd.concat([no_cut, with_cut], axis=1).fillna(0).sort_index(),
            no_cut.sum(),
            with_cut.sum(),
        )
