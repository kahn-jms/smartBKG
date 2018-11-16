# Selective MC generation (smartBKG)

> This program is ready for the grid, the grid is not ready for this program.

SmartBKG is a package designed to facilitate the start-to-finish production of Monte Carlo simulations concentrated on a particular
subset of topologies most relevant to the physics process being investigated.

The goal is to predict early on in the simulation process, before the computationally heavy detector simulation, whether the event being simulated
will be relevant to the study, i.e. will it pass the inital round of trivial selections performed in the beginning of most analyses.
This is achieved by training a (typically convolutional) neural network to classify each event with a usefulness probability.

There are four main steps to achieving this, with an additional final step that can be taken to validate the training's performance:
1. Training data production
2. Preprocessing
3. Training
4. Application
5. Evaluation

## Installation

The Belle II AnalysiS Software (basf2) framework should be setup before installing/running this package, otherwise you will need to install
the dependencies yourself.

1. Clone this repository
2. Install package (better to do it locally): `python3 setup.py install --user`

## Training data production

### Event numbers

Since the skims are contained in a separate MDST (and not as an index file) we 
need to first record the event numbers of all the events that passed the skim.

To do this we save as a numpy array the event numbers.
This relies on each event within an MC batch having unique event numbers (in official MC this is the case).
The arrays can then be merged for a single MC channel and used for event selection on the parent MDST files.

Everything's a lie, just need to use keepParents flag.
~~~At the moment can't seem to access parent event number during processing (overwritten on import?), but when I can:~~~
~~To do this we save a Pandas series with the index being the LFN of the original MDST file (parentLFN in the skim file) and the value is a list of event numbers.~~

## Preprocessing

The preprocessing of training data translates the inputs into a format that can be handled by a neural network.
This is done in two steps:
1. Run standard preprocessing procedures (e.g. one-hot encoding, normalisation)
2. Create a memory map of the data to allow fast processing of inputs larger than the machine's memory

### Standard preprocessing

Once you have the hdf pandas tables saved from the training data production step you can invoke the preprocessing with the
steering file available in the examples:
```bash
python3 examples/preprocessing/standard_preprocessing.py -i /path/to/input.h5 -o /path/to/output/directory/
```

The `-k` flag can also be used if you only want to preprocess some of the data, see `python3 examples/preprocessing/standard_preprocessing.py -h` for info.

### Creating memmaps

Memory maps are a means of letting Numpy quickly read large files without loading the elements into memory until they are actually accessed.
Since the training of a neural network is performed in batches of the total data, we can simply preload the next subset of batches instead of
filling the memory of the machine immediately.
This is especially useful when the total training data exceeds the machine's memory or the machine is shared with other users.

To create the memory maps run the steering file available in the examples.
This should be run separately for each type of preprocessed file output from the previous step, for example to memmap all of the
particle input files you can run:
```bash
python3 examples/preprocessing/create_memmaps.py -i /path/to/preprocessed/files/*_particle_input.npy -o /path/to/output/particle_input.memmap
```
This will create three files, the `particle_input.memmap` file and two more with the same name with `.shape` and `.dtype` appended.
The last two simply contain metadata required for reading the memory maps.
All three should be in the same directory at training time.

**Important note:** Because of the shared regex in the `pdg_input` and `mother_pdg_input` files you may need to place the different input types into different
directories or use some bash regexp, e.g. if the unique sections of filenames all end with numbers:
```bash
python3 examples/preprocessing/create_memmaps.py -i /path/to/preprocessed/files/*[0-9]_particle_input.npy -o /path/to/output/particle_input.memmap
```

## Training

Here we want to train the actual network
