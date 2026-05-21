# Evo2 SAE Recipe

Train a sparse autoencoder on Evo2 (DNA language model) residual-stream activations.

Pipeline:

```
HF Savanna ckpt --convert--> MBridge ckpt
                              |
                      predict_evo2 --embedding-layer N (FASTA in, .pt out)
                              |
                      pt_to_parquet shim (.pt -> ActivationStore parquet shards)
                              |
                      train.py (TopK SAE)
```

The eval / dashboard stage from the esm2 recipe is intentionally not ported in v1.

## Quick start (1B model, single GPU)

```bash
bash scripts/1b.sh
```

This will:

1. Convert `arcinstitute/savanna_evo2_1b_base` to MBridge format
2. Run `predict_evo2` on the OpenGenome2 organelle FASTA, extracting layer-12 embeddings
3. Convert the .pt outputs to parquet shards
4. Train a TopK SAE (expansion=8, k=32)
