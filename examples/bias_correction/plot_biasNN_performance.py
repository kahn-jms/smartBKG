#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Read nTuples from analysis output files and plot requested variable.
# Same as vars compare but splitting up modes more
# James Kahn 2018


# Import libraries
################################################################################
import os
import sys
import matplotlib.pyplot as plt
import importlib

from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1
import argparse # noqa
import root_pandas # noqa
import pandas as pd # noqa

# Custom imports
sys.path.append(sys.path[0] + '/../lib/')
from makeRootCompatible import invertMakeRootCompatible  # noqa


# Read settings/inputs for module params
################################################################################
parser = argparse.ArgumentParser(
    description="Plot single variable from input",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('-i', type=str, required=True, nargs='+',
                    help="Path to input ntuples", metavar="INPUT",
                    dest='in_file')
parser.add_argument('-o', type=str, required=True,
                    help="Directory for output files.", metavar="OUTPUT_DIR",
                    dest='out_dir')
parser.add_argument('-t', type=str, required=False, default='variables',
                    help="Tree/Branch to make plot.", metavar="TREE",
                    dest='tree')
parser.add_argument('-v', type=str, required=True,
                    help="""
                    Variables file to plot (CSV).
                    """,
                    metavar="VAR", dest='var')
parser.add_argument('-r', type=str, required=True,
                    help="File with extra ROOT cuts to apply.", metavar="CUT_FILE",
                    dest='cut_file')
parser.add_argument('-d', type=str, required=False, nargs='+',
                    help="Cut dict keys to apply (requires cut_dict in root file input with -r)",
                    metavar="CUT_KEYS", dest='cut_keys')
parser.add_argument('--bcs', action='store_true',
                    help="Perform best candidate selection.",
                    dest='bcs')
parser.add_argument('--evt-var', type=str, required=False, default='evtNum',
                    help="Unique event identifier to use during best cand. selection for signal file (always evtNum in background)", metavar="EVT_VAR",
                    dest='evt_var')
parser.add_argument('-b', required=False, type=int,  # default='auto',
                    help="Binning level.", metavar="BINNING",
                    dest='binning')
args = parser.parse_args()

# if not (args.sig_file or args.bkg_file or args.cnt_file):
#     print("You must specify at least one input file.\n")
#     parser.print_help()
#     sys.exit()

# Handle input and output files
################################################################################
# Make sure output directory exists
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

# Load the Root cuts to apply
sys.path.append(os.path.dirname(args.cut_file))
cut_module = importlib.import_module(
    os.path.splitext(
        os.path.basename(args.cut_file)
    )[0]
)

applied_cuts = []
if args.cut_keys:
    for k in args.cut_keys:
        # Convert to matplotlib compatible
        cut_str = cut_module.cut_dict[k]
        cut_str = [a.replace('||', '|') for a in cut_str]
        cut_str = [a.replace('&&', '&') for a in cut_str]
        applied_cuts += cut_str

print("Cuts applied:", applied_cuts)

channel_dict = {
    'True Btag': ('tab:blue', (31. / 255, 119. / 255, 180. / 255, 0.1), args.in_file),
    'False Btag': ('tab:red', (214. / 255, 39. / 255, 40. / 255, 0.1), args.in_file),
    # 'True $B_{sig}$': ('tab:blue', (31. / 255, 119. / 255, 180. / 255, 0.1), args.in_file),
    # 'False $B_{sig}$': ('tab:red', (214. / 255, 39. / 255, 40. / 255, 0.1), args.in_file),
    # 'Signal': ('tab:blue', (31. / 255, 119. / 255, 180. / 255, 0.1), args.sig_file),
    # 'SCF': ('tab:orange', (255. / 255, 127. / 255, 14 / 255, 0.1), args.sig_file),
    # # Combined backgrounds
    # 'BB Background': ('tab:green', (44. / 255, 160. / 255., 44. / 255, 0.1), args.bkg_file),
    # 'Continuum': ('tab:purple', (148. / 255, 103. / 255, 189. / 255, 0.1), args.cnt_file),
    # # Decomposed backgrounds
    # 'B+B-': ('tab:green', (44. / 255, 160. / 255., 44. / 255, 0.1), args.bcharged),
    # 'B0B0': ('tab:red', (214. / 255, 39. / 255, 40. / 255, 0.1), args.bneutral),
    # 'uubar': ('tab:purple', (148. / 255, 103. / 255, 189. / 255, 0.1), args.uubar),
    # 'ddbar': ('tab:brown', (140. / 255, 86. / 255, 75. / 255, 0.1), args.ddbar),
    # 'ccbar': ('tab:pink', (227. / 255, 119. / 255, 194. / 255, 0.1), args.ccbar),
    # 'ssbar': ('tab:olive', (188. / 255, 189. / 255, 34. / 255, 0.1), args.ssbar),
}
df_dict = {}

# Load the ntuples as pandas dataframes
for key in channel_dict.keys():
    if channel_dict[key][2] is not None:
        df = root_pandas.read_root(channel_dict[key][2], key=args.tree)
        for col in df:
            df[col] = df[col].astype(float)

        if len(applied_cuts) > 0:
            df = df.query(' & '.join(applied_cuts))

        if args.bcs:
            df = df.groupby(args.evt_var).apply(lambda g: g.nsmallest(1, "extraInfo__boFEIProbabilityRank__bc"))

        # Doing this means we have to load the signal files twice, but oh well
        if key == 'True Btag':
        # if key == 'True $B_{sig}$':
            df = df.query(' & '.join(cut_module.sig_cut))
        elif df.shape[0] > 0:  # Will error if query with no events
            df = df.query(' & '.join(cut_module.bkg_cut))

        print('N {}:'.format(key), df.shape[0])
        # Won't add channels with no remaining events
        if df.shape[0] == 0:
            continue

        df_dict[key] = df

# Load variables file
var_file_df = pd.read_csv(args.var, comment='#')
# Fix stuff
var_file_df['variable_nice'] = var_file_df['variable'].apply(invertMakeRootCompatible)
var_file_df['logscale'] = var_file_df['logscale'].fillna(False)
# Fill remaining empty values with None because NaNs suck when working with strings and floats together
var_file_df = var_file_df.where((pd.notnull(var_file_df)), None)

# Vars to plot with (should put in seperate config file as input)
for index, row in var_file_df.iterrows():

    # For plotting purposes only, ignore outliers
    if row['low_x'] is None:
        low_quant = 0.001
        high_quant = 0.999
        plot_outliers = pd.DataFrame()
        for key in df_dict.keys():
            plot_outliers = plot_outliers.append(df_dict[key][row['variable']].quantile([low_quant, high_quant]))
        row['low_x'] = plot_outliers[low_quant].min()
        row['high_x'] = plot_outliers[high_quant].max()
    # # print("Range to plot:", row['low_x'], row['high_x'])

    plt.figure(figsize=(4, 3))

    color_args = dict(
        # histtype='stepfilled',
        # histtype='step',
        histtype='bar',
        # density=True,
        density=False,
        # normed=True,
        # normed=False,
        stacked=False,
        # alpha=0.5,
        # fill=True,
        # edgecolor='auto',
    )

    # for key in df_dict.keys():
    #     # if key in ['Signal', 'SCF']:
    #     #     color_args['normed'] = True
    #     # else:
    #     #     color_args['normed'] = False
    #     if df_dict[key].shape[0] == 1:
    #         continue

    #     plt.hist(
    #         df_dict[key][row['variable']],
    #         bins=args.binning,
    #         range=(row['low_x'], row['high_x']),
    #         label='{} ({})'.format(key, df_dict[key].shape[0]),
    #         **color_args,
    #         edgecolor=channel_dict[key][0],
    #         facecolor=channel_dict[key][1],
    #     )
    plt.hist(
        [df_dict[key][row['variable']] for key in df_dict.keys()],
        bins=args.binning if args.binning else 'auto',
        range=(row['low_x'], row['high_x']),
        label=['{}'.format(key) for key in df_dict.keys()],
        **color_args,
        # edgecolor=[channel_dict[key][0] for key in df_dict.keys()],
        # facecolor=[channel_dict[key][1] for key in df_dict.keys()],
    )

    # if row['units']:
    #     plt.xlabel(r'{} ({})'.format(row['variable_nice'], row['units']))
    # else:
    #     plt.xlabel(r'{}'.format(row['variable_nice']))
    plt.xlabel(r'{}'.format(row['title']))

    # if color_args['normed']:
    if 'density' in color_args.keys() and color_args['density']:
        plt.ylabel('nEvents (normalised)')
    else:
        plt.ylabel('nEvents')

    plt.legend(loc='best')

    if row['logscale']:
        plt.yscale('log')
        # plt.xscale('log')

    plt.savefig(
        os.path.join(args.out_dir, row['variable']) + '.pdf',
        bbox_inches='tight',
        transparent=True,
    )
