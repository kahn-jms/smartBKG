#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import argparse
import pandas as pd

from smartBKG.preprocessing import MCParticlesPreprocManager  # type:ignore


def getCmdArgs():
    parser = argparse.ArgumentParser(
        description='''Preprocess data for NN input and save.''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', type=str, required=True,
                        help="Path to pandas h5 input.", metavar="INPUT",
                        dest='in_file')
    parser.add_argument('-k', type=str, required=False, choices=['train_events', 'decay_strings'],
                        help="Key name of pandas table in input file.", metavar="KEY",
                        dest='key')
    parser.add_argument('-o', type=str, required=True,
                        help="Directory to save numpy output (will have same filename as input + _KEY.np", metavar="OUTPUT",
                        dest='save_dir')
    return parser.parse_args()


if __name__ == '__main__':

    args = getCmdArgs()
    os.makedirs(args.save_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(args.in_file))[0]
    basefile = os.path.join(args.save_dir, basename)

    preproc = MCParticlesPreprocManager()

    print('Preprocessing file {}'.format(args.in_file))
    # Should really populate a dict and use keys from that
    if not args.key or args.key == 'train_events':
        print('Running particle preprocessing')
        df = preproc.ppp.load_hdf(args.in_file, 'train_events')
        particle_input, pdg_input, mother_pdg_input = preproc.preproc_whole_decay(df)
        y_output = preproc.preproc_y_output(df, key='train_events')

        print('Particle preprocessing finished, saving to {}'.format(args.save_dir))
        preproc.save_npy_preprocd(
            particle_input,
            '{}_particle_input.npy'.format(basefile)
        )
        preproc.save_npy_preprocd(
            pdg_input,
            '{}_pdg_input.npy'.format(basefile)
        )
        preproc.save_npy_preprocd(
            mother_pdg_input,
            '{}_mother_pdg_input.npy'.format(basefile)
        )
    if not args.key or args.key == 'decay_strings':
        print('Running decay string preprocessing')
        df = preproc.ppp.load_hdf(args.in_file, 'decay_strings')
        decay_input = preproc.preproc_decay_string(df)
        y_output = preproc.preproc_y_output(df, key='decay_strings')

        print('Decay string preprocessing finished, saving to {}'.format(args.save_dir))
        preproc.save_npy_preprocd(
            decay_input,
            '{}_decay_input.npy'.format(basefile)
        )

    print('Saving training labels'.format(args.save_dir))
    preproc.save_npy_preprocd(
        y_output,
        '{}_y_output.npy'.format(basefile)
    )
