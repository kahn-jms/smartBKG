#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import importlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import smartBKG.evtPdl as evtPdl  # type:ignore


parser = argparse.ArgumentParser()
parser.add_argument('--particle_input', type=str, required=False, default=None, nargs='+',
                    help="Path to particle_input npy.", metavar="PARTICLE_INPUT",
                    dest='particle_input')
parser.add_argument('--pdg_input', type=str, required=False, default=None, nargs='+',
                    help="Path to pdg_input npy.", metavar="PDG_INPUT",
                    dest='pdg_input')
parser.add_argument('--mother_pdg_input', type=str, required=False, default=None, nargs='+',
                    help="Path to mother_pdg_input npy.", metavar="MOTHER_PDG_INPUT",
                    dest='mother_pdg_input')
parser.add_argument('--decay_input', type=str, required=False, default=None, nargs='+',
                    help="Path to decay_input npy.", metavar="DECAY_INPUT",
                    dest='decay_input')
parser.add_argument('--y_output', type=str, required=False, default=None, nargs='+',
                    help="Path to y_outpu npy.", metavar="Y_OUTPUT",
                    dest='y_output')
parser.add_argument('-m', type=str, required=True,
                    help="Path to model class to train. Should always have name NN_model and inherit NN_base_model.",
                    metavar="MODEL", dest='model')
parser.add_argument('--batch-size', type=int, required=False, default=128,
                    help="Batch sizes to train with", metavar="BATCH_SIZE",
                    dest='batch_size')
parser.add_argument('--epochs', type=int, required=False, default=30,
                    help="Number of major epochs to train for", metavar="EPOCHS",
                    dest='epochs')
args = parser.parse_args()

def load_multi_npy(arg):
    return np.concatenate([np.load(f) for f in arg])

in_files = {
    'particle_input': load_multi_npy(args.particle_input),
    'pdg_input': load_multi_npy(args.pdg_input),
    'mother_pdg_input': load_multi_npy(args.mother_pdg_input),
    # 'decay_input': load_multi_npy(args.decay_input),
    'y_output': load_multi_npy(args.y_output),
}

# Load the NN model to build/train
sys.path.append(os.path.dirname(args.model))
model_module = importlib.import_module(
    os.path.splitext(
        os.path.basename(args.model)
    )[0]
)

shape_dict = {}
for key in in_files.keys():
    shape_dict[key] = (None, *in_files[key].shape[1:])
print(shape_dict)

print('Split/shuffle data')
splits = train_test_split(
    *tuple(in_files[k] for k in in_files.keys()),
    test_size=0.1,
    shuffle=True,
)

X_train = {}
X_test = {}
y_train = {}
y_test = {}

for idx, k in enumerate(in_files.keys()):
    if k == 'y_output':
        y_train['{}'.format(k)] = splits[idx * 2]
        y_test['{}'.format(k)] = splits[(idx * 2) + 1]
    else:
        X_train['{}'.format(k)] = splits[idx * 2]
        X_test['{}'.format(k)] = splits[(idx * 2) + 1]

print(y_train)

print('Setting class weights')
unique = np.unique(y_train['y_output'])
class_weights = compute_class_weight('balanced', unique, y_train['y_output'])
class_weights = dict(zip(unique, class_weights))


print('Building model')
NN_model = model_module.NN_model(
    shape_dict=shape_dict,
    num_pdg_codes=len(evtPdl.pdgTokens),
)
NN_model.build_model()


NN_model.model.fit(
    x=X_train,
    y=y_train,
    batch_size=args.batch_size,
    epochs=args.epochs,
    validation_data=(X_test, y_test),
    class_weight=class_weights
)
