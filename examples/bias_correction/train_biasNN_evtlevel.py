#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import importlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from smartBKG.evaluate import makeRootCompatible

from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1
import argparse  # noqa
import root_pandas  # noqa


def getCmdArgs():
    parser = argparse.ArgumentParser(
        description='''Preprocess data for NN input and train.''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', type=str, required=True, nargs='+',
                        help="Path to pandas h5 input.", metavar="INPUT",
                        dest='in_file')
    parser.add_argument('-m', type=str, required=True,
                        help="Path to NN model to train.", metavar="MODEL",
                        dest='model')
    parser.add_argument('-t', type=float, required=False, default=0.5,
                        help="Threshold for NN output", metavar="THRESHOLD",
                        dest='threshold')
    parser.add_argument('-o', type=str, required=True,
                        help="Directory to save training output", metavar="OUTPUT",
                        dest='out_dir')
    return parser.parse_args()


batch_size = 4
epochs = 100

# Var: nCategories
disc_train_vars = {
    # 'nTracks': 30,
    'nCleanedTracks(dr<2 and abs(dz)<4)': 23,
    'nCleanedClusters': 33,
    # 'nKLMClusters': 20,
    # 'R2EventLevel',
    # 'isSignal': 2,
    # 'charge': 5,
}
# Var: (mean, std)
cont_train_vars = {
    # 'nECLClusters': (185, 25),  # Discrete but let's cheat a little (should really embed this)
    # 'Eher',
    # 'Eler',
    # 'Mbc': (5.265, 0.0138),
    # 'deltaE': (0., 0.065),
    # 'missingMass',
    # 'M': (5.252, 0.066),
    # 'useCMSFrame(p)': (0.5, 0.15),
    # 'extraInfo(SignalProbability)': (0.886, 0.183),
    # 'eventExtraInfo(smartBKG)': (0.9, 0.1),
}
label_var = 'eventExtraInfo(smartBKG)'

# Need these to be fixed

# Convert to ROOT strings
disc_train_vars = {makeRootCompatible(s): v for (s, v) in disc_train_vars.items()}
cont_train_vars = {makeRootCompatible(s): v for (s, v) in cont_train_vars.items()}
root_label_var = makeRootCompatible(label_var)


if __name__ == '__main__':

    args = getCmdArgs()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load the NN model to build/train
    sys.path.append(os.path.dirname(args.model))
    model_module = importlib.import_module(
        os.path.splitext(
            os.path.basename(args.model)
        )[0]
    )

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
    y = y['label'].mask(y[root_label_var] > args.threshold, 1)

    # tanh estimator (magic)
    # X[list(cont_train_vars.keys())] = X[list(cont_train_vars.keys())].apply(
    #     lambda x:
    #     # 0.5 * (np.tanh(0.01 * ((x - cont_train_vars[x.name][0]) / cont_train_vars[x.name][1]) + 1))
    #     (x - cont_train_vars[x.name][0]) / cont_train_vars[x.name][1]
    # )
    # One-hot encode
    # for v in disc_train_vars.keys():
    #     dummy_cols = ['{}_{}'.format(v, c) for c in range(disc_train_vars[v])]
    #     dummy_df = pd.get_dummies(
    #         X[v],
    #         columns=[v],
    #         prefix=v
    #     )
    #     dummy_df = dummy_df.T.reindex(dummy_cols).T.fillna(0)
    #     # Here we append the one-hot encoded column(s) and drop the original
    #     X = pd.concat([X, dummy_df], axis=1)
    #     X = X.drop(v, axis=1)

    print('Split/shuffle data')
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        shuffle=True,
    )
    # X_train = X_train[:100]
    # y_train = y_train[:100]
    # print(X_train)

    print('Setting class weights')
    unique = np.unique(y_train)
    class_weights = compute_class_weight('balanced', unique, y_train)
    class_weights = dict(zip(unique, class_weights))
    print('Class weights:', class_weights)

    print('Building model')
    NN_model = model_module.NN_model(X_train.shape)
    NN_model.build_model()

    NN_model.model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        class_weight=class_weights
    )
