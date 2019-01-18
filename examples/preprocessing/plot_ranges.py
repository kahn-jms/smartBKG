''' Plot distributions of MCParitcles and decay strings fori calculating padding

    Reads pandas dataframes before preprocessing and plots number of entries
    in MCParticles and decay strings.
    This can be used to find the optimal padding values.
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import matplotlib.pyplot as plt


PARSER = argparse.ArgumentParser()
PARSER.add_argument('-i', type=str, required=True, nargs='+',
                    help="Path to pandas dataframes (before preprocessing).", metavar="INPUT",
                    dest='input')
ARGS = PARSER.parse_args()


load_dfs = []
load_dec_dfs = []
for f in ARGS.input:
    load_dfs.append(pd.read_hdf(f, key='train_events'))
    load_dec_dfs.append(pd.read_hdf(f, key='decay_strings'))
df = pd.concat(load_dfs, axis=1)
dec_df = pd.concat(load_dec_dfs, axis=1)

dec_df['Decay string'] = dec_df['decay_str'].apply(lambda x: len(x.split(' ')))

# df[:1000]['x'].apply(abs).plot.hist(
#     range=(0,7),
#     histtype='step',
#     bins=10000,
#     )

# plt.xscale('log')
# plt.show()

comp_df = df.reset_index().groupby('evtNum')['arrayIndex'].nunique()
print(comp_df.head())
print(dec_df.head())

# concat_df = pd.concat([comp_df, dec_df['len']])
# print(concat_df.head())
dec_df['MCParticles'] = comp_df
print(dec_df.head())

# plt.figure(figsize=(4, 3))
# ax = plt.gca()
# df.reset_index().groupby('evtNum')['arrayIndex'].nunique().plot.hist(
#     bins=100
# )
dec_df[['Decay string', 'MCParticles']].plot.hist(
    histtype='stepfilled',
    alpha=0.5,
    # bins=int(dec_df['Decay string'].max() / 3),
    range=(10, 160),
    bins=150,
    density=True,
    figsize=(5, 3.5),
    # ax=ax,
)

plt.xlabel('No. components')
# plt.show()
plt.savefig(
    'MCParticle_counts.pdf',
    bbox_inches='tight',
    transparent=True,
)
