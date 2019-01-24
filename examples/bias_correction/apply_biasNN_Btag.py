#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import pandas as pd
import math

from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1
import argparse  # noqa
import root_pandas  # noqa
# This has to go last or we get a segfault
from tensorflow.keras.models import load_model  # noqa
from smartBKG.evaluate import makeRootCompatible  # noqa


def getCmdArgs():
    parser = argparse.ArgumentParser(
        description='''Preprocess data for NN input and save.''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', type=str, required=True, nargs='+',
                        help="Path to input Ntuple.", metavar="INPUT",
                        dest='in_file')
    parser.add_argument('-m', type=str, required=True,
                        help="Path to trained NN model to apply.", metavar="MODEL",
                        dest='model')
    parser.add_argument('-t', type=float, required=False, default=0.5,
                        help="Threshold for NN output", metavar="THRESHOLD",
                        dest='threshold')
    parser.add_argument('-o', type=str, required=True,
                        help="Directory to save output plots", metavar="OUTPUT",
                        dest='out_dir')
    return parser.parse_args()


# Var: nCategories
disc_train_vars = {
    # 'nTracks': 24,
    'nCleanedTracks(dr<2 and abs(dz)<4)': 12,
    # 'nKLMClusters': 15,
    # 'R2EventLevel',
    # 'isSignal': 2,
    # 'charge': 5,
}
# Var: (mean, std)
cont_train_vars = {
    # 'nECLClusters': (185, 25),  # Discrete but let's cheat a little (should really embed this)
    'Mbc': (5.265, 0.0138),
    'deltaE': (0., 0.065),
    # 'missingMass',
    # 'M': (5.252, 0.066),
    # 'useCMSFrame(p)': (0.47, 0.15),
    'extraInfo(SignalProbability)': (0.0213, 0.0816),
    # 'ROE_E__boROE__bc': (4.278, 1.029),
    'thrustBm': (0.797, 0.077),
    'thrustOm': (0.728, 0.075),
    # 'WE_MissP__boROE__cm__sp0__bc': (0.86, 0.96),
    # 'WE_MissE__boROE__cm__sp0__bc': (),
    'R2': (0.166, 0.092),
    'cosTBTO': (0.56, 0.29),
    'cosTBz': (0.38, 0.25),
}
label_var = 'eventExtraInfo(smartBKG)'

# Convert to ROOT strings
disc_train_vars = {makeRootCompatible(s): v for (s, v) in disc_train_vars.items()}
cont_train_vars = {makeRootCompatible(s): v for (s, v) in cont_train_vars.items()}
root_label_var = makeRootCompatible(label_var)

if __name__ == '__main__':

    args = getCmdArgs()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load Ntuples to process
    df_list = []
    for f in args.in_file:
        df_list.append(root_pandas.read_root(f))
    df = pd.concat(df_list)

    # Preprocess
    X = df[list(cont_train_vars.keys()) + list(disc_train_vars.keys())]
    y = df[[root_label_var]]

    # Threshold labels to 0 or one depending on input threshold
    # should be a builtin to do this but I don't know it
    y = y.assign(label=0)
    # Drop the original labels
    y = y[['label']].mask(y[root_label_var] > args.threshold, 1)

    # tanh estimator (magic)
    X[list(cont_train_vars.keys())] = X[list(cont_train_vars.keys())].apply(
        lambda x:
        # 0.5 * (np.tanh(0.01 * ((x - cont_train_vars[x.name][0]) / cont_train_vars[x.name][1]) + 1))
        (x - cont_train_vars[x.name][0]) / cont_train_vars[x.name][1]
    )
    # One-hot encode
    for v in disc_train_vars.keys():
        dummy_cols = ['{}_{}'.format(v, c) for c in range(disc_train_vars[v])]
        dummy_df = pd.get_dummies(
            X[v],
            columns=[v],
            prefix=v
        )
        dummy_df = dummy_df.T.reindex(dummy_cols).T.fillna(0)
        # Here we append the one-hot encoded column(s) and drop the original
        X = pd.concat([X, dummy_df], axis=1)
        X = X.drop(v, axis=1)
    # print(X.columns)
    # X = X[:10]
    # y = y[:10]
    # print(X.shape)
    # print(X.columns)
    # print(X.head())
    # sys.exit()

    # Load the model
    model = load_model(args.model)
    pred = model.predict(X)

    # Add the inference results to the true labels
    results = y.assign(pred=pred)
    # results['pred'] = results['pred'].apply(
    #     lambda x: math.log((x) / (1.0 - x))
    # )

    # Save plots of the performance
    plt.figure(figsize=(7, 4))
    gp = results.groupby('label')['pred']
    plt.hist(
        [g[1] for g in gp],
        # label=[g[0] for g in gp],
        label=[
            'Predicted fail (NN $\leq 0.85$)',
            'Predicted pass (NN $> 0.85$)'
        ],
        histtype='stepfilled',
        bins='auto',
        alpha=0.5,
        density=True,
    )
    plt.legend(loc='best')
    plt.xlabel('Bias NN prediction')
    # plt.show()
    plt.savefig(
        os.path.join(args.out_dir, 'biasNN_predictions') + '.pdf',
        bbox_inches='tight',
        transparent=True,
    )

    # Finally, append the results to the original input and save
    # Doing CSV for now, hdf segfaults? Need to figure out why
    df = df.assign(biasNN_pred=results['pred'])
    # df.to_csv(
    #     os.path.join(args.out_dir, 'bias_predictions') + '.csv',
    # )
    df.to_hdf(
        os.path.join(args.out_dir, 'bias_predictions') + '.h5',
        key='s',
        # mode='w',
    )
