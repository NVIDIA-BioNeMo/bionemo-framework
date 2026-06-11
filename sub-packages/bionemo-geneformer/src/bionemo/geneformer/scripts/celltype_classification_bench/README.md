# Cell type classification benchmark

This benchmark uses a Chron's disease dataset from both healthy and unhealthy patients. We first perform inference on all models against the dataset, then using cross validation, we fit a MLP on the frozen embeddings from each sample. The resulting test accuracy in each fold is then averaged to compute the final performance.

Importantly, this benchmark is NOT equivalent to the cell type classification notebook provided in documentation. The documentation link is an example, and will not reproduce to the same results shared below.

## How to run this benchmark

1. Download the underlying h5ad file:

```bash
python download.py --base-dir path/to/result
```

(Will be saved as hs-celltype-bench.h5ad)
2\) Convert the downloaded file in to scdl by pointing `--data-path` to the parent directory containing the h5ad files to convert:

```bash
convert_h5ad_to_scdl --data-path path/to/result/ --save-path path/to/result/processed_input
```

3. Execute inference using some pretrained model and the scdl converted input:

-- NOTE: checkpoint directory should have two folders, `context` and `weights`. This can be downloaded for pretrained models on NGC, or point to your own pre-trained model. This directory structure is the default checkpoint from both NeMo and BioNeMo.

```bash
infer_geneformer \
    --data-dir path/to/result/processed_input \
    --checkpoint-path path/to/checkpoint \
    --results-path path/to/result/inference-embeddings \
    --micro-batch-size 8 \
    --seq-len 2048 \
    --num-dataset-workers 10 \
    --num-gpus 1 \
    --include-input-ids
```

4. Run the cell type classification script. This requires the original h5ad file (for metadata) and the inference embeddings.

```
python bench.py path/to/result/hs-celltype-bench.h5ad path/to/result/inference-embeddings
```

Results are then saved in a csv file (results.csv) in the same directory as inference embeddings.
