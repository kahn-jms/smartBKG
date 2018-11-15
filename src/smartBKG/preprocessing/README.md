# Utility functions

Useful tools to make mass data manipulation easier

## Memmap creation

This allows you to process more training data than fits in memory

```bash
python3 create_memmaps.py \
		-i /srv/data/jkahn/output/smrt_gen/FEIskim_init_rec/mixed/neutral/sub00/gen_vars/pandas/preprocessed/*_combined_x.npy \
		-o /srv/data/jkahn/output/smrt_gen/FEIskim_init_rec/mixed/neutral/sub00/gen_vars/pandas/preprocessed/particle_input.memmap \
		--npy-load-memmap
```


