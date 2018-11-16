#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Create plots to compare metrics between multiple trainings
# James Kahn

import argparse
import pandas as pd
from matplotlib import pyplot as plt
import os


class CompareTrainingMetrics():
    ''' Class to compare multiple training's metrics '''
    def __init__(self, metric_files):
        self.metric_files = metric_files

        self.metrics_df = self._load_metrics(self.metric_files)

    def _load_metrics(self, files):
        ''' Load the metrics into a single dataframe.

        The index will be [filename, epoch]
        '''
        file_dict = {}
        for f in files:
            f_base = os.path.splitext(os.path.basename(f))[0]
            try:
                f_df = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                print('File contains no data, skipping ({})'.format(f))
            if f_df.shape[0] < 2:
                'Training {} contains one or less epochs, skipping'.format(f_base)
            else:
                file_dict[f_base] = f_df

        df = pd.concat(file_dict)
        # Drop epoch column, don't need
        return df.drop('epoch', axis=1)

    def plot_all_metrics(self, out_dir):
        ''' Plot all available metrics in single plot and separately'''
        # First plot all metrics together
        self.plot_metrics(
            self.metrics_df.columns,
            os.path.join(out_dir, 'all_metrics.pdf')
        )

        for col in self.metrics_df.columns:
            self.plot_metrics(
                col,
                os.path.join(out_dir, '{}.pdf'.format(col))
            )

    def plot_metrics(self, metrics, out_file):
        ''' Plot the given metric(s) to using matplotlib

        Arguments:
            metrics (str, list): The metrics in the dataframe to plot
            out_file (str): Path to file for saving plot
        '''
        plt.figure()
        self.metrics_df.unstack(level=0)[metrics].plot()
        plt.legend(loc='best', prop={'size': 6})
        plt.grid(True, which='both', axis='y')
        plt.xlabel('Epoch')
        if isinstance(metrics, str) or len(metrics) == 1:
            plt.ylabel(metrics)
        plt.minorticks_on()
        # Turn minor ticks back on for y-axis
        # ml = MultipleLocator(5)
        # plt.axes().yaxis.set_minor_locator(ml)
        # plt.ylim(0., 1.0)
        plt.savefig(
            out_file,
            bbox_inches='tight',
            transparent=True,
        )
        # plt.show()


def GetCmdArgs():
    parser = argparse.ArgumentParser(
        description='''Plot training metrics from input files''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-i', type=str, required=True, nargs='+',
                        help="Path to input training csv files.", metavar="INPUT",
                        dest='in_files')
    parser.add_argument('-o', type=str, required=True,
                        help="Output directory for plot files", metavar="OUTPUT",
                        dest='out_dir')
    return parser.parse_args()


if __name__ == '__main__':

    args = GetCmdArgs()
    os.makedirs(args.out_dir, exist_ok=True)

    ctm = CompareTrainingMetrics(args.in_files)
    ctm.plot_all_metrics(args.out_dir)
