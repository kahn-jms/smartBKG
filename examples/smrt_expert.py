#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Apply training during simulation and save list of pass events
# James Kahn 2018

# Import libraries
################################################################################
import argparse
import glob

# BASF2/BelleII libraries
# ROOT is shit, will override our argparse without this
from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1
# from ROOT import Belle2
import reconstruction  # noqa
import simulation  # noqa
from generators import add_evtgen_generator  # noqa
import modularAnalysis as ma  # noqa
import basf2 as b2  # noqa

# Custom functions
from NN_apply_module import NNApplyModule  # noqa


# Read settings/inputs for module params
################################################################################
supported_models = ['combined-LSTM', 'combined-wideCNN', 'particles', 'decstr-LSTM', 'decstr-wideCNN']
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Event generation, simulation, and reconstruction."
)
parser.add_argument('-d', type=str, required=False,
                    help="Path to decay file to use.", metavar="DECAY_FILE",
                    dest='decay_file')
parser.add_argument('-b', type=str, required=True,
                    help="Path to background files' directory.", metavar="BKG_DIR",
                    dest='bkg_dir')
parser.add_argument('-n', type=int, required=True,
                    help="Number of events to generate.", metavar="NUM_EVENTS",
                    dest='num_events')
parser.add_argument('-m', type=str, required=True, nargs='+',
                    help="Path to trained model", metavar="MODEL",
                    dest='model')
parser.add_argument('--type', type=str, required=True, nargs='+',
                    choices=supported_models,
                    help='Type of input models (order matches order of model files): {}'.format(supported_models),
                    metavar="TYPE", dest='model_type')
parser.add_argument('-t', type=float, required=False, default=0.0,
                    help="Threshold to cut evtgen events on.", metavar="THRESHOLD",
                    dest='threshold')
args = parser.parse_args()

assert (len(args.model) == len(args.model_type)), 'MODEL files must have exactly on TYPE each'

# User output
print("\nSettings:")
if args.decay_file:
    print("Input file set to:\t" + args.decay_file)
print("Trained model loaded:\t" + str(args.model))
print("Number of events to generate:\t" + str(args.num_events))

# Need to fetch output directory from basf2 -o flag and create it

main = b2.create_path()

# ma.setupEventInfo(args.num_events, path=main)
# Need to input decay file with Y4S -> B0 B0 if I want mixed bkg?
if args.decay_file:
    # add_evtgen_generator(main, 'signal', args.decay_file)
    # Deprecated
    ma.generateY4S(noEvents=args.num_events, decayTable=args.decay_file, path=main)
else:
    # Annoying, need to add flag for this shit now
    # add_evtgen_generator(main, 'mixed')
    ma.generateY4S(noEvents=args.num_events, path=main)

# Override event number -- use skip-events flag for basf2 instead -- easy in htcondor
# set_module_parameters(path=main, name='EventInfoSetter', recursive=False, runList=[0], expList=[0])

NNApplyModule_m = NNApplyModule(
    model_file=args.model[0],
    model_type=args.model_type[0],
    threshold=args.threshold,
    extra_info_var='smartBKG'

)
dead_path = b2.create_path()
NNApplyModule_m.if_false(dead_path)

main.add_module(NNApplyModule_m)

# For multiple models
# for model, model_type in zip(args.model, args.model_type):
#     NNApplyModule_m = NNApplyModule(
#         model_file=model,
#         model_type=model_type,
#         threshold=args.threshold,
#     )
#     dead_path = b2.create_path()
#     NNApplyModule_m.if_false(dead_path)

#     main.add_module(NNApplyModule_m)
#     # Do I need this here?
#     # c1.set_property_flags(b2.ModulePropFlags.PARALLELPROCESSINGCERTIFIED)


# ma.loadGearbox(path=main)
# main.add_module('Geometry', ignoreIfPresent=True, components=['MagneticField'])
simulation.add_simulation(
    path=main,
    bkgfiles=glob.glob(args.bkg_dir + '/*.root')
)
reconstruction.add_reconstruction(path=main)

# dump generated events in MDST format to the output ROOT file
reconstruction.add_mdst_output(
    path=main,
    mc=True,
    filename='RootOutput.root',
    additionalBranches=['EventExtraInfo'],
)

# Begin processing
################################################################################
main.add_module('ProgressBar')
b2.process(path=main)

print(b2.statistics)
