.. _tutorial:

========
Tutorial
========

There are four main steps in the running of this software, each 
depending on the last, with the evaluation as an additional 
cross-check that can be applied to the resulting simulations:

1. :ref:`train_data_prod`
2. :ref:`preprocessing`
3. :ref:`training`
4. :ref:`application`
5. :ref:`evaluation`

.. highlight:: console

.. _train_data_prod:

Training data production
________________________

The first step in any ML challenge is to collect samples for training.
In our case we want the MCParticle information labelled with whether it will pass or fail the analysis selection chosen (e.g. skims).
To do this you can run the `examples/data_prod/grid_training_prod.py` steering file provided.
There are two ways to run the file, either on the grid or locally over downloaded MDSTs.

Grid data production
^^^^^^^^^^^^^^^^^^^^

This is not ready yet, need to add keepParents flag to RootInput and either tokenize decay string already or write out as TString.

Local data production
^^^^^^^^^^^^^^^^^^^^^

Should add htcondor examples too.

.. _preprocessing:

Preprocessing
_____________

The preprocessing of training data translates the inputs into a format that can be handled by a neural network.
This is done in two steps:

1. Run standard preprocessing procedures (e.g. one-hot encoding, normalisation)
2. Create a memory map of the data to allow fast processing of inputs larger than the machine's memory

Standard preprocessing
^^^^^^^^^^^^^^^^^^^^^^

Once you have the hdf pandas tables saved from the training data production step you can invoke the preprocessing with the
steering file available in the examples
::

   python3 examples/preprocessing/standard_preprocessing.py -i /path/to/input.h5 -o /path/to/output/directory/

The `-k` flag can also be used if you only want to preprocess some of the data, see `python3 examples/preprocessing/standard_preprocessing.py -h` for info.

Creating memmaps
^^^^^^^^^^^^^^^^

Memory maps are a means of letting Numpy quickly read large files without loading the elements into memory until they are actually accessed.
Since the training of a neural network is performed in batches of the total data, we can simply preload the next subset of batches instead of
filling the memory of the machine immediately.
This is especially useful when the total training data exceeds the machine's memory or the machine is shared with other users.

To create the memory maps run the steering file available in the examples.
This should be run separately for each type of preprocessed file output from the previous step, for example to memmap all of the
particle input files you can run
::

   python3 examples/preprocessing/create_memmaps.py -i /path/to/preprocessed/files/*_particle_input.npy -o /path/to/output/particle_input.memmap

This will create three files, the `particle_input.memmap` file and two more with the same name with `.shape` and `.dtype` appended.
The last two simply contain metadata required for reading the memory maps.
All three should be in the same directory at training time.

**Important note:** Because of the shared regex in the `pdg_input` and `mother_pdg_input` files you may need to place the different input types into different
directories or use some bash regexp, e.g. if the unique sections of filenames all end with numbers:
::

   python3 examples/preprocessing/create_memmaps.py -i /path/to/preprocessed/files/*[0-9]_particle_input.npy -o /path/to/output/particle_input.memmap


.. _training:

Training
________

Now that we have all the training input data prepared we can actually train the actual network.
It's *highly* recommended you perform the training on a GPU, if not it's recommended to visit `here <https://www.nvidia.com/de-de/shop/geforce/?nvid=nv-int-geo-de-shop-all>`_ first.

There is a range of models available to use in `examples/train/models`, with top-level model being for both decay string and MCparticles input.
The models in subdirectories deal with the individual inputs only as indicated by their names.
From my own tests I found that the ResNet models seem to be the most robust so these are a good start.
You are of course free to create your own models, the only requirement is that they inherit the `NNBaseClass` and you acknowledge me in your Nobel acceptance speech.

As an example, training the combined input ResNet model can be done with
::

   python3 train_network.py \
   --decay_input /path/to/decay_input.memmap \
   --particle_input path/to/particle_input.memmap \
   --pdg_input path/to/pdg_input.memmap \
   --mother_pdg_input /path/to/mother_pdg_input.memmap \
   --y_output /path/to/y_output.memmap \
   -m models/CNN_ResNet.py \
   -o training/output/ \
   -l logs/

It's recommended to not save the logs to your home directory but somewhere with large disposable storage as these
can become quite large if many trainings are performed.

.. _application:

Application
___________

.. _evaluation:

Evaluation
__________
