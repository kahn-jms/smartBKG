# Grid Operations Guide

> This program is ready for the grid, the grid is not ready for this program.

This is a guide on how to run mass creation of training data.
Currently the grid doesn't support any (big) output files that are non-ROOT format, so we need to make use of the root\_pandas package to save/load our dataframes.

Sub-packages:
- [expert\_data\_production](expert_data_production/README.md)

## Definitions

### Event numbers

Since the skims are contained in a separate MDST (and not as an index file) we 
need to first record the event numbers of all the events that passed the skim.

To do this we save as a numpy array the event numbers.
This relies on each event within an MC batch having unique event numbers (in official MC this is the case).
The arrays can then be merged for a single MC channel and used for event selection on the parent MDST files.

At the moment can't seem to access parent event number during processing (overwritten on import?), but when I can:
~~To do this we save a Pandas series with the index being the LFN of the original MDST file (parentLFN in the skim file) and the value is a list of event numbers.~~
