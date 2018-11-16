#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Train a network and log the outputs
# James Kahn

import argparse
import os
import sys
import time
from keras import callbacks
# import tensorflow as tf
import importlib

# from tensorboard_wrapper import TensorBoardWrapper
from smartBKG.train import LoadMemmapData  # type:ignore
import smartBKG.evtPdl as evtPdl  # type:ignore


def getCmdArgs():
    parser = argparse.ArgumentParser(
        description='''Train the given model''',
        epilog='''
        The input memmaps files expect a corresponding .shape and .dtype file
        containing the memmap metadata. It's recommended to create the memmaps
        with the create_memmaps.py script in the utils dir as this generates these automatically.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--particle_input', type=str, required=False, default=None,
                        help="Path to particle_input memmap.", metavar="PARTICLE_INPUT",
                        dest='particle_input')
    parser.add_argument('--pdg_input', type=str, required=False, default=None,
                        help="Path to pdg_input memmap.", metavar="PDG_INPUT",
                        dest='pdg_input')
    parser.add_argument('--mother_pdg_input', type=str, required=False, default=None,
                        help="Path to mother_pdg_input memmap.", metavar="MOTHER_PDG_INPUT",
                        dest='mother_pdg_input')
    parser.add_argument('--decay_input', type=str, required=False, default=None,
                        help="Path to decay_input memmap.", metavar="DECAY_INPUT",
                        dest='decay_input')
    parser.add_argument('--y_output', type=str, required=False, default=None,
                        help="Path to y_outpu memmap.", metavar="Y_OUTPUT",
                        dest='y_output')
    # parser.add_argument('--cache', type=str, required=True,
    #                     help="Directory to store collective memmaps in.",
    #                     metavar="CACHE", dest='cache')
    parser.add_argument('-m', type=str, required=True,
                        help="Path to model class to train. Should always have name NN_model and inherit NN_base_model.",
                        metavar="MODEL", dest='model')
    parser.add_argument('-t', type=str, required=False, default=None,
                        help="Previously trained model to load. Will cause -m flag to be ignored", metavar="LOAD",
                        dest='load_model')
    parser.add_argument('-o', type=str, required=True,
                        help="Output directory for results", metavar="OUTPUT",
                        dest='out_dir')
    parser.add_argument('-l', type=str, required=True,
                        help="Log directory to save tensorboard output", metavar="LOGDIR",
                        dest='log_dir')
    parser.add_argument('--queue', type=int, required=False, default=1000,
                        help="Number of batches to preload into training queue (must be smaller than total number of train events)",
                        metavar="QUEUE", dest='queue')
    parser.add_argument('--epochs', type=int, required=False, default=30,
                        help="Number of major epochs to train for", metavar="EPOCHS",
                        dest='epochs')
    parser.add_argument('--sub-epochs', type=int, required=False, default=4,
                        help="Number of times per epoch to perform validation (useful for very long epoch times)",
                        metavar="SUB_EPOCHS", dest='sub_epochs')
    parser.add_argument('--batch-size', type=int, required=False, default=128,
                        help="Batch sizes to train with", metavar="BATCH_SIZE",
                        dest='batch_size')
    parser.add_argument('--cpu', action='store_true',
                        help="Run training on CPU instead of GPU",
                        dest='cpu')
    return parser.parse_args()


if __name__ == '__main__':

    args = getCmdArgs()
    epochs = args.epochs
    sub_epochs = args.sub_epochs
    batch_size = args.batch_size
    train_queue_size = args.queue
    test_queue_size = int(0.1 * train_queue_size)

    in_files = {
        'particle_input': args.particle_input,
        'pdg_input': args.pdg_input,
        'mother_pdg_input': args.mother_pdg_input,
        'decay_input': args.decay_input,
        'y_output': args.y_output,
    }
    # ADD as default in LoadMemmapData arg, user's don't need to know this
    padding_dict = {
        'particle_input': 100,
        'pdg_input': 100,
        'mother_pdg_input': 100,
        'decay_input': 150,
    }

    # ADD to NN base class, I guess when initialised, and timestamp the model later
    now = time.strftime("%Y.%m.%d.%H.%M")

    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print('Loading input data')
    data_loader = LoadMemmapData(
        in_files,
        # memmap_dir=args.cache,
        padding_dict=padding_dict,
    )
    data_loader.populate_train_test()

    # Load the NN model to build/train
    sys.path.append(os.path.dirname(args.model))
    model_module = importlib.import_module(
        os.path.splitext(
            os.path.basename(args.model)
        )[0]
    )

    print('Building model')
    NN_model = model_module.NN_model(
        shape_dict=data_loader.shape_dict,
        num_pdg_codes=len(evtPdl.pdgTokens),
    )

    if args.load_model:
        NN_model.load_model(args.load_model)
    else:
        NN_model.build_model()

    # Append training time to model name (useful when re-training a model)
    # ADD to base class
    NN_model.model.name = '{}_{}'.format(NN_model.model.name, now)

    # Create the output dirs, better to do here incase loading train data fails
    # ADD to NN base class
    out_dir = os.path.join(args.out_dir, now)
    log_dir = os.path.join(args.log_dir, now)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print('out_dir:', out_dir)
    print('log_dir:', log_dir)

    NN_model.plot_model(out_dir)

    # Setup callbacks
    # ADD callbacks to base NN class
    modelCheckpoint = callbacks.ModelCheckpoint(
        os.path.join(
            out_dir,
            str(NN_model.model.name) + '_{epoch:02d}-{val_loss:.2f}.h5'
        ),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1,
    )

    # Reduce learning rate if training is stalling
    rlrop = callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=int(sub_epochs / 2),
        verbose=1,
        mode='auto',
        min_lr=0.
    )

    # Stop training if it's not improving
    earlyStop = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=sub_epochs * 3,
        verbose=1,
        mode='auto'
    )

    # Record metrics in CSV (for later plotting with Matplotlib)
    csvLogger = callbacks.CSVLogger(
        os.path.join(out_dir, '{}_metrics.csv'.format(NN_model.model.name)),
    )

    # Need to specify in fit_generator how many times to call batch generator
    # ADD to data_loader
    train_steps = int(
        data_loader.memmap_dict['y_output'].shape[0] * data_loader.train_frac / (sub_epochs * batch_size)
    )
    validation_steps = int(
        data_loader.memmap_dict['y_output'].shape[0] * (1 - data_loader.train_frac) / (sub_epochs * batch_size)
    )

    # # Tensorboard callback (doesn't work atm)
    # Need to run validation steps twice if using tensorboard
    # validation_steps = int(validation_steps / 2)
    # # Make separate subdirs for logs, needed for run separation
    # # tbCallBack = callbacks.TensorBoard(
    # tbCallBack = TensorBoardWrapper(
    #     batch_gen=data_loader.batch_generator(batch_size),
    #     nb_steps=validation_steps,
    #     log_dir=os.path.join(log_dir),
    #     histogram_freq=1,
    #     write_graph=False,
    #     write_grads=True,
    #     batch_size=batch_size,
    #     # write_images=True,
    # )

    NN_model.model.fit_generator(
        data_loader.batch_generator(batch_size),
        steps_per_epoch=train_steps,
        epochs=int(epochs * sub_epochs),
        validation_data=data_loader.batch_generator(batch_size),
        validation_steps=validation_steps,
        # ADD this will become callbacks=model.callbacks
        callbacks=[
            modelCheckpoint,
            csvLogger,
            # tbCallBack,
            rlrop,
            earlyStop,
        ],
        max_queue_size=train_queue_size,
        class_weight=data_loader.class_weights,
        # workers=16,
        use_multiprocessing=True,
    )
