#!/bin/bash

PREPROC_DIR=/path/to/preprocessed/files/

for f in decay_input particle_input pdg_input mother_pdg_input y_output
do
    python3 create_memmaps.py -i ${PREPROC_DIR}/*[0-9]_${f}.npy -o ${PREPROC_DIR}/${f}.memmap &>${f}.log &
done
