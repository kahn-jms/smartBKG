#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Steering file to save training data on grid
# James Kahn 2018

# Import libraries
################################################################################
import os

# # BASF2/BelleII libraries
from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1

import argparse # noqa
import basf2 as b2 # noqa
import modularAnalysis as ma # noqa

# Local module import
from train_data_module import TrainDataSaver  # noqa

# Read settings/inputs for module params
################################################################################
parser = argparse.ArgumentParser(
    description='''Generator variables saver.''',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--input', type=str, required=False, default='',
                    help="Input file. If given via basf2 -i flag output file defaults to train_data.h5", metavar="IN_FILE",
                    dest='in_file')
parser.add_argument('-e', type=str, required=False, default='evtNum.npy',
                    help="Event number file. If it doesn't exist it will be created from input file", metavar="EVTNUM_FILE",
                    dest='evtnum_file')
parser.add_argument('-k', type=int, required=False, action='store',
                    default=-1, choices=[-1, 0, 1],
                    help="Keep only pass (1), fail (0), or all (-1) events.", metavar="KEEP_ONLY",
                    dest='keep_only')
parser.add_argument('-o', type=str, required=False, default='./',
                    help="Directory for output files", metavar="OUTPUT_DIR",
                    dest='out_dir')
args = parser.parse_args()

# Set up output file (not integrated into basf2 command line yet)
if len(args.in_file) > 0:
    out_filename = '{}.h5'.format(os.path.splitext(os.path.basename(args.in_file))[0])
else:
    out_filename = 'train_data.h5'
os.makedirs(args.out_dir, exist_ok=True)
outfile = os.path.join(args.out_dir, out_filename)
print("Output file set to:\t" + outfile)

# Main processing
################################################################################
path = ma.create_path()

# Should change to RootInput with keppParents=True
# Don't need the other junk added by inputMdst since we're only dealing with MCParticles
ma.inputMdstList('default', filelist=[args.in_file], path=path)

TrainDataSaver_module = TrainDataSaver(
    evt_num_file=args.evtnum_file,
    output_file=outfile,
    keep_only=args.keep_only,
)
path.add_module(TrainDataSaver_module)

b2.process(path)
print(b2.statistics)
