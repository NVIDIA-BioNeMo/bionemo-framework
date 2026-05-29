# Evo2 SAE Recipe

Train a sparse autoencoder on Evo2 (DNA language model) residual-stream activations.

Pipeline:

```
HF Savanna ckpt --convert--> MBridge ckpt
                              |
                      extract.py (FASTA in, ActivationStore parquet shards out)
                              |
                      train.py (TopK SAE)
```

`extract.py` monkey-patches `predict_evo2`'s writer to stream parquet shards inline,
so there is no `.pt` intermediate and no separate shim step.

The eval / dashboard stage from the esm2 recipe is intentionally not ported in v1.

## Quick start (1B model, 4 GPU)

```bash
bash scripts/1b.sh
```

This will:

1. Convert `arcinstitute/savanna_evo2_1b_base` to MBridge format
2. Run `extract.py` on the OpenGenome2 organelle FASTA, streaming layer-12
   activations directly to parquet shards (no `.pt` intermediate)
3. Train a TopK SAE (expansion=8, k=32, auxk=512)

Common overrides:

```bash
# Different layer, different FASTA, tagged output paths
LAYER=22 FASTA=/data/.../prokeuk_25M.fasta RUN_TAG=25M_prokeuk bash scripts/1b.sh

# Skip extraction (assumes parquet already exists at PARQUET_DIR)
TRAIN_ONLY=1 PARQUET_DIR=/data/.../parquet_25M_prokeuk bash scripts/1b.sh

# Sweep hyperparams
LAYER=22 RUN_TAG=auxk2048 AUXK=2048 N_EPOCHS=4 bash scripts/1b.sh
```

See the top of `scripts/1b.sh` for the full list of env-overridable variables
(`MODEL`, `LAYER`, `CHUNK_BP`, `FASTA`, `RUN_TAG`, `MAX_TOKENS`, `MICRO_BATCH`,
`DEVICES`, `EXPANSION_FACTOR`, `TOP_K`, `AUXK`, `AUXK_COEF`,
`DEAD_TOKENS_THRESHOLD`, `N_EPOCHS`, `LR`, `WANDB_API_KEY`, `WANDB_PROJECT`,
`WANDB_RUN_NAME`, `TRAIN_ONLY`).
