==================================
Selective MC generation (smartBKG)
==================================

SmartBKG is a package designed to facilitate the start-to-finish production of 
Monte Carlo simulations concentrated on a particular
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

The :ref:`tutorial` section covers each of the steps.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   tutorial


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
