## Steps to run
1. apply `.patch` to nemo so that nvfsdp can run
2. run `run.sh` so that the .pt files are made
3. Run `python compare_grads.py` to collapse distributed grads and compare to single process grads
