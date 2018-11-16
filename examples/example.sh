#!/bin/bash

PREPROC_DIR=/srv/data/jkahn/output/smrt_gen/FEIskim_init_rec/charged/charged/sub00/gen_vars/preprocessed/

for f in decay_input particle_input pdg_input mother_pdg_input y_output
do
    python3 create_memmaps.py -i ${PREPROC_DIR}/*[0-9]_${f}.npy -o ${PREPROC_DIR}/${f}.memmap &>${f}.log &
    #python3 create_memmaps.py -i ${PREPROC_DIR}/udst_0001[0-3]*[0-9]_${f}.npy ${PREPROC_DIR}/mdst_00*[0-9]_${f}.npy -o ${TEST_DIR}/${f}.memmap &>${f}.log &
done

#for f in particle_input pdg_input mother_pdg_input y_output
#do
#    python3 create_memmaps.py -i ${PREPROC_DIR}/*[0-9]_${f}.npy -o ${PREPROC_DIR}/${f}.memmap --npy-load-memmap &>${f}.log &
#done
