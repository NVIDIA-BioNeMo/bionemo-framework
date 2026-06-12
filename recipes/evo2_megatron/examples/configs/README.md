## Example configs

These configs are provided as examples to the user. Note that the files referenced in these configs can be downloaded from [OpenGenome2 dataset on Hugging Face](https://huggingface.co/datasets/arcinstitute/opengenome2).

- `full_pretrain_shortphase_config.yaml` was used to test full scale pre-training runs of evo2 at the 8k context length.
- `full_pretrain_longphase_config.yaml` was used to test full scale context extension phase pre-training (starting from an 8k checkpoint and continuing to train at longer context lengths).
- `test_preproc_config.yaml` was used to test our preprocessing scripts to generate .bin/.idx files that are used for pre-training from fasta file inputs.
- `test_promotors_dataset_config.yaml` is a smaller test file that can be used for pre-training but is one of the smaller tests.
