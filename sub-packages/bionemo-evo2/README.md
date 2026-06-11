# bionemo-evo2

`bionemo-evo2` is a `pip`-installable package that contains **data preprocessing**, **training**, and **inferencing** code for Evo2, a new `Hyena`-based foundation model for genome generation and understanding. Built upon `Megatron-LM` parallelism and `NeMo2` algorithms, `bionemo-evo2` provides the remaining tools necessary to effectively fine-tune the pre-trained Evo2 model checkpoint on user-provided sequences at scale, and generate state-of-the-art life-like DNA sequences from Evo2 for downstream metagenomic tasks.

## Available models in NGC

| HF Model                                                                                        | BioNeMo Resource Name                                                                                                 | Blackwell FP8 | Blackwell BF16 | Hopper FP8 | Hopper BF16 | Ampere | Notes                                                                                                                                                                                                                                                                    |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------- | -------------- | ---------- | ----------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [arcinstitute/savanna_evo2_1b_base](https://huggingface.co/arcinstitute/savanna_evo2_1b_base)   | [evo2/1b-8k:1.0](https://registry.ngc.nvidia.com/orgs/nvidia/teams/clara/models/evo2-1b-8k-nemo2)                     | ✅            | ❌             | ✅         | ❌          | ❌     | Low accuracy on bf16 (eg ampere) GPUs                                                                                                                                                                                                                                    |
|                                                                                                 | [evo2/1b-8k-bf16:1.0](https://registry.ngc.nvidia.com/orgs/nvidia/teams/clara/models/evo2-1b-8k-bf16-nemo2)           | ✅            | ✅             | ✅         | ✅          | ✅     | Fine-tuned variant of the 1b-8k that supports bf16 as well as fp8, enabling ampere as well as hopper/blackwell.                                                                                                                                                          |
| [arcinstitute/savanna_evo2_7b_base](https://huggingface.co/arcinstitute/savanna_evo2_7b_base)   | [evo2/7b-8k:1.0](https://registry.ngc.nvidia.com/orgs/nvidia/teams/clara/models/evo2-7b-8k-nemo2)                     | ✅            | ✅             | ✅         | ✅          | ✅     | The original 7b models have good accuracy across the board at bf16 and fp8 across tested hardware.                                                                                                                                                                       |
| [arcinstitute/savanna_evo2_7b](https://huggingface.co/arcinstitute/savanna_evo2_7b)             | [evo2/7b-1m:1.0](https://registry.ngc.nvidia.com/orgs/nvidia/teams/clara/models/evo2-7b-1m-nemo2)                     | ✅            | ✅             | ✅         | ✅          | ✅     | The original 7b models have good accuracy across the board at bf16 and fp8 across tested hardware.                                                                                                                                                                       |
| [arcinstitute/savanna_evo2_40b_base](https://huggingface.co/arcinstitute/savanna_evo2_40b_base) |                                                                                                                       | ?             | ?              | ?          | ?           | ?      | Unknown, likely has the same support pattern as the 40b-1m row below since this is the same model at an earlier step of training.                                                                                                                                        |
| [arcinstitute/savanna_evo2_40b](https://huggingface.co/arcinstitute/savanna_evo2_40b)           |                                                                                                                       | ❌            | ❌             | ✅         | ❌          | ❌     | The original 40b-1m context trained model only supports hpper fp8                                                                                                                                                                                                        |
|                                                                                                 | [evo2/40b-1m-fp8-bf16:1.0](https://registry.ngc.nvidia.com/orgs/nvidia/teams/clara/models/evo2-40b-1m-fp8-bf16-nemo2) | ✅            | ✅             | ✅         | ✅          | ✅     | A fine-tuned variant of [arcinstitute/savanna_evo2_40b](https://huggingface.co/arcinstitute/savanna_evo2_40b) with broad hardware support (fp8 or bf16 and ampere, hopper, and blackwell have all been tested). The original model only has good accuracy on hopper fp8. |

On the CLI you can access the resources in this table (and others) with:

```bash
CKPT_PATH=$(download_bionemo_data evo2/40b-1m-fp8-bf16:1.0)
```

In code these resources can be accessed with:

```python
from bionemo.core.data.load import load

ckpt_path = load("evo2/40b-1m-fp8-bf16:1.0")
```

Or you can follow the links in the table above to the ngc registry and follow the download links from there.

Note, in the following two sections, the model described as `ft1(step199)` is the model that was released above as `evo2/40b-1m-fp8-bf16:1.0`.

### Loss evaluation

| device | model_size | is_finetune | fine_tune_desc |                               precision | ctx_length | average_nll | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| -----: | ---------: | ----------: | -------------: | --------------------------------------: | ---------: | ----------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   a100 |         1b |       FALSE |           None |                                    bf16 |       8192 |    1.242033 | 1b base model works ok on b300, but cannot handle bf16 precision (and by extension ampere)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|   h200 |         1b |       FALSE |           None |                                     fp8 |       8192 |    1.076465 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   b300 |         1b |       FALSE |           None |                                     fp8 |       8192 |    1.084777 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   h200 |         1b |       FALSE |           None |                                    bf16 |       8192 |    1.243525 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   b300 |         1b |       FALSE |           None |                                    bf16 |       8192 |    1.243527 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   a100 |         1b |        TRUE |             ft |                                    bf16 |       8192 |    1.078681 | 1b base model fine-tuned for bf16 can handle both bf16 and b300. B300 accuracy is also more similar to H200 accuracy after fine-tuning to handle bf16. Ampere appears to work fine as well.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|   h200 |         1b |        TRUE |             ft |                                     fp8 |       8192 |    1.078623 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   b300 |         1b |        TRUE |             ft |                                     fp8 |       8192 |     1.07901 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   h200 |         1b |        TRUE |             ft |                                    bf16 |       8192 |    1.078671 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   b300 |         1b |        TRUE |             ft |                                    bf16 |       8192 |    1.078694 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   a100 |      7b-1m |       FALSE |           None |                                    bf16 |       8192 |    0.995102 | 7b model got lucky in training and generalizes well to bf16 precision as well as to blackwell and ampere.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|   h200 |      7b-1m |       FALSE |           None |                                     fp8 |       8192 |    0.995265 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   b300 |      7b-1m |       FALSE |           None |                                     fp8 |       8192 |      0.9951 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   h200 |      7b-1m |       FALSE |           None |                                    bf16 |       8192 |    0.995109 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   b300 |      7b-1m |       FALSE |           None |                                    bf16 |       8192 |     0.99535 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   a100 |     40b-1m |       FALSE |           None |                                    bf16 |       8192 |    1.702023 | 40b model got unlucky in training. It is sensitive to fp8 and within that appears to have memorized the known difference in hopper that leads to lower accuracy when using standard fp8 computations. (see Deepseek V3 paper where they point out the hopper difference in the “Increasing Accumulation Precision” sub-section where hopper uses 14 bits to accumulate partials rather than the typical 32 bits). It does not work well on bf16 and that seems to carry over to ampere as expected. Note if we set (use_split_accumulator=True) to True by setting https://github.com/NVIDIA/TransformerEngine/blob/bd55e7ba5f0235a80eaa63d49adaa8fb7c6ced50/transformer_engine/pytorch/module/base.py#L56 to True then the fp8 is more accurate which breaks fp8 on hopper, making it seem more like blackwell.                              |
|   h200 |     40b-1m |       FALSE |           None |                                     fp8 |       8192 |    0.922422 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   b300 |     40b-1m |       FALSE |           None |                                     fp8 |       8192 |       1.789 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   h200 |     40b-1m |       FALSE |           None | fp8-delayed(use_split_accumulator=True) |       8192 |    1.791161 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   h200 |     40b-1m |       FALSE |           None |                                    bf16 |       8192 |     1.70015 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   b300 |     40b-1m |       FALSE |           None |                                    bf16 |       8192 |    1.700162 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   a100 |     40b-1m |        TRUE |            ft0 |                                    bf16 |       8192 |    0.962564 | The first fine-tuning run used a global batch size of 4 rather than 16. The training loss curve was very unstable which could have lead to a lower quality fine-tune. This was successful in that every hardware and fp8 precision combination works to some degree. The accuracy sits between the 7b and 40b checkpoints. This is also reflected in a 1% AUC drop on the BRCA1 notebook. https://wandb.ai/nvidia/evo2_40b_finetune/runs/Alp3KXuC/overview. Note that the accuracy on hopper or blackwell bf16 seems to closely track with ampere bf16.                                                                                                                                                                                                                                                                                       |
|   h200 |     40b-1m |        TRUE |            ft0 |                                     fp8 |       8192 |    0.963434 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   b300 |     40b-1m |        TRUE |            ft0 |                                     fp8 |       8192 |     0.95985 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   h200 |     40b-1m |        TRUE |            ft0 | fp8-delayed(use_split_accumulator=True) |       8192 |    0.959287 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   h200 |     40b-1m |        TRUE |            ft0 |                                    bf16 |       8192 |    0.962654 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   b300 |     40b-1m |        TRUE |            ft0 |                                    bf16 |       8192 |    0.962621 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   a100 |     40b-1m |        TRUE |   ft1(step119) |                                    bf16 |       8192 |    0.955813 | The second fine-tuning run has the same accuracy in the BRCA notebook as the original model, and maintains similar accuracy on hopper at fp8 (0.926 vs 0.922). Unfortunately the accuracy drops somewhat on bf16 as well as blackwell, but it is marginally better than the previous fine-tuning run. Accuracy closely tracks between ampere, hopper, and blackwell at bf16.                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|   h200 |     40b-1m |        TRUE |   ft1(step119) |                                     fp8 |       8192 |    0.926986 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   b300 |     40b-1m |        TRUE |   ft1(step119) |                                     fp8 |       8192 |    0.954112 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   h200 |     40b-1m |        TRUE |   ft1(step119) | fp8-delayed(use_split_accumulator=True) |       8192 |    0.953928 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   h200 |     40b-1m |        TRUE |   ft1(step119) |                                    bf16 |       8192 |    0.955881 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   b300 |     40b-1m |        TRUE |   ft1(step119) |                                    bf16 |       8192 |    0.955859 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   h200 |     40b-1m |        TRUE |   ft1(step279) |                                     fp8 |       8192 |    1.379552 | Interestingly if you keep training the model, the accuracy continues to degrade on validation slightly, but note that the model has now shifted its sensitivity away from the fp8 rounding pecularity on hopper to requring the more accurate FP8 implementation on blackwell. Perhaps fine-tuning at a lower learning rate (I used the final minimal learning rate from the pretraining run), with more dropout (I used 0.1% dropout), or more weight decay (I set a very smalll value to nearly disable it rather than how the model was trained at 0.1). https://wandb.ai/nvidia/evo2_40b_finetune/runs/Ji2IRcrz/overview. Note if we set (use_split_accumulator=True) to True by setting https://github.com/NVIDIA/TransformerEngine/blob/bd55e7ba5f0235a80eaa63d49adaa8fb7c6ced50/transformer_engine/pytorch/module/base.py#L56 to True. |
|   b300 |     40b-1m |        TRUE |   ft1(step279) |                                     fp8 |       8192 |    0.958749 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   h200 |     40b-1m |        TRUE |   ft1(step279) | fp8-delayed(use_split_accumulator=True) |       8192 |    0.957551 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   h200 |     40b-1m |        TRUE |   ft1(step279) |                                    bf16 |       8192 |    0.959398 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|   b300 |     40b-1m |        TRUE |   ft1(step279) |                                    bf16 |       8192 |    0.959373 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

### AUC Evaluation

| device | model_size | is_finetune | fine_tune_desc | precision | BRCA1 SM AUC | BRCA1 Bal AUC | BRCA1 AUC |
| -----: | ---------: | ----------: | -------------: | --------: | ------------ | ------------- | --------- |
|   A100 |        40b |        TRUE |   ft1(step119) |      BF16 |              |               | 0.86      |
|   H200 |        40b |        TRUE |   ft1(step119) |      BF16 |              |               |           |
|   B300 |        40b |        TRUE |   ft1(step119) |      BF16 |              |               |           |
|   B300 |        40b |        TRUE |   ft1(step119) |       FP8 |              |               | 0.87      |
|   H200 |        40b |        TRUE |   ft1(step119) |       FP8 |              |               | 0.88      |
|   A100 |        40b |        TRUE |   ft1(step279) |      BF16 |              |               | 0.86      |
|   B300 |        40b |        TRUE |   ft1(step279) |      BF16 |              |               |           |
|   B300 |        40b |        TRUE |   ft1(step279) |       FP8 |              |               |           |
|   H200 |        40b |        TRUE |   ft1(step279) |       FP8 |              |               | 0.5       |
|   A100 |      7b-1m |       FALSE |                |      BF16 |              |               | 0.88      |
|   B300 |      7b-1m |       FALSE |                |       FP8 |              | 0.88          |           |
|   H200 |      7b-1m |       FALSE |                |       FP8 |              |               | 0.88      |
|   H200 |        40b |        TRUE |  ft0(step2600) |       FP8 |              |               | 0.47      |
|   B300 |        40b |        TRUE |   ft0(step870) |      BF16 |              |               | 0.86      |
|   B300 |        40b |        TRUE |   ft0(step870) |       FP8 |              | 0.86          |           |
|   H200 |        40b |        TRUE |   ft0(step870) |       FP8 |              | 0.86          | 0.86      |
|   H200 |        40b |       FALSE |                |       FP8 | 0.85         |               | 0.87      |
|   A100 |        40b |       FALSE |                |      BF16 |              |               |           |
|   B300 |        40b |       FALSE |                |      BF16 | 0.55         |               |           |
|   H200 |        40b |       FALSE |                |      BF16 | 0.53         |               |           |
|   B300 |        40b |       FALSE |                |       FP8 | 0.48         |               |           |

## Quickstart tutorials

Two Jupyter notebooks are available to help you get started with Evo 2: one demonstrating how to finetune the model on your own sequences, and another showing how to perform zero-shot BRCA1 variant effect prediction.

- [Finetuning](./examples/fine-tuning-tutorial.ipynb)

- [Zeroshot BRCA1 Variant Effect Prediction](./examples/zeroshot_brca1.ipynb)

## Installation

To install this package, execute the following command:

```bash
pip install -e .
```

To run unit tests, execute the following command:

```bash
pytest -v .
```

## Preprocessing

To train or fine-tune Evo2 on a custom dataset, we need to preprocess and index sequence data for training from raw FASTA files into tokenized binaries compliant with `NeMo2` / `Megatron-LM`. For more information about how to configure your data for training, refer to [data/README.md](src/bionemo/evo2/data/README.md) and [utils.config.Evo2PreprocessingConfig](src/bionemo/evo2/utils/config.py).

```bash
preprocess_evo2 -c <CONFIG_PATH>
```

## Training

Given a preprocessed collection of preprocessed datasets, and optionally a pre-trained NeMo2 checkpoint for Evo2, training can be executed using the following command:

```bash
$ train_evo2 --help
usage: train_evo2 [-h] (-d DATASET_CONFIG | --mock-data) [--dataset-dir DATASET_DIR] [--num-nodes NUM_NODES] [--devices DEVICES] [--seq-length SEQ_LENGTH] [--tensor-parallel-size TENSOR_PARALLEL_SIZE]
                  [--pipeline-model-parallel-size PIPELINE_MODEL_PARALLEL_SIZE] [--context-parallel-size CONTEXT_PARALLEL_SIZE] [--create-tensorboard-logger]
                  [--wandb-entity WANDB_ENTITY] [--wandb-project WANDB_PROJECT] [--wandb-tags WANDB_TAGS [WANDB_TAGS ...]] [--wandb-group WANDB_GROUP] [--wandb-job-type WANDB_JOB_TYPE] [--wandb-id WANDB_ID]
                  [--wandb-anonymous] [--wandb-log-model] [--wandb-offline] [--sequence-parallel] [--fp8] [--micro-batch-size MICRO_BATCH_SIZE] [--global-batch-size GLOBAL_BATCH_SIZE] [--grad-acc-batches GRAD_ACC_BATCHES]
                  [--max-steps MAX_STEPS] [--early-stop-on-step EARLY_STOP_ON_STEP] [--val-check-interval VAL_CHECK_INTERVAL] [--grad-reduce-in-fp32] [--fp8-wgrad] [--use-megatron-comm-overlap-llama3-8k] [--tp-comm-overlap-backend {nccl,mpi,gloo}]
                  [--align-param-gather] [--model-size {1b,1b_nv,40b,40b_arc_longcontext,40b_nv,7b,7b_arc_longcontext,7b_nv,test,test_nv}] [--add-bias-output] [--result-dir RESULT_DIR] [--experiment-name EXPERIMENT_NAME]
                  [--limit-val-batches LIMIT_VAL_BATCHES] [--log-every-n-steps LOG_EVERY_N_STEPS] [--ckpt-dir CKPT_DIR] [--wd WD] [--restore-optimizer-from-ckpt] [--no-average-in-collective] [--seed SEED]
                  [--workers WORKERS] [--gc-interval GC_INTERVAL] [--enable-preemption] [--ckpt-async-save] [--ckpt-format {torch_dist,zarr}] [--eod-pad-in-loss-mask] [--cross-entropy-loss-fusion] [--no-fp32-residual-connection]
                  [--debug-ddp-parity-freq DEBUG_DDP_PARITY_FREQ] [--hybrid-override-pattern HYBRID_OVERRIDE_PATTERN] [--num-layers NUM_LAYERS] [--create-tflops-callback] [--log-parameters-and-shapes] [--lr LR] [--min-lr MIN_LR]
                  [--warmup-steps WARMUP_STEPS] [--nsys-profiling] [--nsys-start-step NSYS_START_STEP] [--nsys-end-step NSYS_END_STEP] [--no-renormalize-loss] [--nsys-ranks NSYS_RANKS [NSYS_RANKS ...]]
                  [--activation-checkpoint-recompute-num-layers ACTIVATION_CHECKPOINT_RECOMPUTE_NUM_LAYERS] [--disable-checkpointing] [--clip-grad CLIP_GRAD] [--seq-len-interpolation-factor SEQ_LEN_INTERPOLATION_FACTOR]
                  [--overlap-param-gather] [--overlap-grad-reduce] [--hidden-dropout HIDDEN_DROPOUT] [--attention-dropout ATTENTION_DROPOUT] [--save-top-k SAVE_TOP_K] [--metric-to-monitor-for-checkpoints METRIC_TO_MONITOR_FOR_CHECKPOINTS] [--save-last-checkpoint] [--no-save-last-checkpoint] [--no-activation-checkpointing | --selective-activation-checkpointing]

Train a Hyena model using NeMo 2.0.

options:
  -h, --help            show this help message and exit
  -d DATASET_CONFIG, --dataset-config DATASET_CONFIG
                        Path to the blended / weighted training dataset configuration YAML. (default: None)
  --mock-data           Train with Mock data (for testing/debugging), either set this or provide a dataset config. (default: False)
  --dataset-dir DATASET_DIR
                        Absolute path to the dataset directory. Defaults to using the absolute or relative paths (dataset_prefix) specified in the dataset config YAML. (default: None)
  --num-nodes NUM_NODES
                        Number of nodes to use for training, defaults to 1. (default: 1)
  --devices DEVICES     Number of devices to use for training, defaults to 1. (default: 1)
  --seq-length SEQ_LENGTH
                        Training sequence length (default: 8192)
  --tensor-parallel-size TENSOR_PARALLEL_SIZE
                        Order of tensor parallelism. Defaults to 1. (default: 1)
  --pipeline-model-parallel-size PIPELINE_MODEL_PARALLEL_SIZE
                        Order of pipeline parallelism. Defaults to 1. (default: 1)
  --context-parallel-size CONTEXT_PARALLEL_SIZE
                        Order of context parallelism. Defaults to 1. (default: 1)
  --create-tensorboard-logger
                        Create a tensorboard logger. (default: False)
  --wandb-entity WANDB_ENTITY
                        The team posting this run (default: None)
  --wandb-project WANDB_PROJECT
                        Wandb project name (default: None)
  --wandb-tags WANDB_TAGS [WANDB_TAGS ...]
                        Tags associated with this run (default: None)
  --wandb-group WANDB_GROUP
                        A unique string shared by all runs in a given group (default: None)
  --wandb-job-type WANDB_JOB_TYPE
                        A unique string representing a type of run, which is useful when you're grouping runs together into larger experiments using group. (default: None)
  --wandb-id WANDB_ID   Sets the version, mainly used to resume a previous run (default: None)
  --wandb-anonymous     Enable or explicitly disable anonymous logging (default: False)
  --wandb-log-model     Save checkpoints in wandb dir to upload on W&B servers (default: False)
  --wandb-offline       Use wandb in offline mode (default: False)
  --sequence-parallel   Set to enable sequence parallelism. (default: False)
  --fp8                 Set to enable FP8 (default: False)
  --micro-batch-size MICRO_BATCH_SIZE
                        Micro-batch size for data-parallel training. (default: 1)
  --global-batch-size GLOBAL_BATCH_SIZE
                        Global batch size for training. If set to None, infer it from the TP, CP, and PP parameters. (default: None)
  --grad-acc-batches GRAD_ACC_BATCHES
                        Number of batches to accumulate gradients over. (default: 1)
  --max-steps MAX_STEPS
                        Number of training optimizer update steps. This controls the total number of steps as well as the shape of the learning rate curve. (default: 500000)
  --early-stop-on-step EARLY_STOP_ON_STEP
                        Stop training on this step, if set. This may be useful for testing or debugging purposes. (default: None)
  --val-check-interval VAL_CHECK_INTERVAL
                        Number of steps between validation measurements and model checkpoints. (default: None)
  --grad-reduce-in-fp32
                        Gradient reduce in FP32. (default: False)
  --fp8-wgrad           Faster option that is maybe less accurate (TBD) when using fp8. (default: False)
  --use-megatron-comm-overlap-llama3-8k
  --tp-comm-overlap-backend {nccl,mpi,gloo}
                        TP communication backend to use. Defaults to 'nccl'. (default: nccl)
  --align-param-gather
  --model-size {1b,1b_nv,40b,40b_arc_longcontext,40b_nv,7b,7b_arc_longcontext,7b_nv,test,test_nv}
                        Model architecture to use, choose between 7b, 40b, or test (a sub-model of 4 layers, less than 1B parameters). '_arc_1m' models have GLU / FFN dimensions that support 1M context length when trained with TP<=8. (default: 7b)
  --add-bias-output     Add bias to the output layer to enable learning a simple prior. (default: False)
  --result-dir RESULT_DIR
                        Path to the result directory. (default: results)
  --experiment-name EXPERIMENT_NAME
                        Name of the experiment. (default: evo2)
  --limit-val-batches LIMIT_VAL_BATCHES
                        Number of validation steps (default: 20)
  --log-every-n-steps LOG_EVERY_N_STEPS
                        Number of steps between logging. (default: 1)
  --ckpt-dir CKPT_DIR   Directory to restore an initial checkpoint from. Use this for supervised fine-tuning. (default: None)
  --wd WD               Weight decay for optimizer. (default: 0.01)
  --restore-optimizer-from-ckpt
                        Restore optimizer state from initial checkpoint. Defaults to False. (default: False)
  --no-average-in-collective
                        Avaerage optimizer state in collective rather than dividing by dp size and summing. (default: False)
  --seed SEED           Set random seed for training. (default: 1234)
  --workers WORKERS     Number of workers to use for data loading. (default: 8)
  --gc-interval GC_INTERVAL
                        Set to a value > 0 if you want to synchronize garbage collection, will do gc every gc-interval steps. (default: 0)
  --enable-preemption   Enable preemption hooks. If enabled this will save a checkpoint whenever slurm exits. (default: False)
  --ckpt-async-save
  --ckpt-format {torch_dist,zarr}
                        Specify checkpoint format to use. Defaults to 'torch_dist', as 'zarr' is deprecated. Only use if resuming training from a zarr checkpoint. (default: torch_dist)
  --eod-pad-in-loss-mask
                        Do not predict EOD/Pad tokens (typical default, but not default in original evo2). (default: False)
  --cross-entropy-loss-fusion
                        Use the faster, but maybe less accurate fused form of cross entropy, which also has bf16 grads internally. (default: False)
  --no-fp32-residual-connection
                        If set, turn off fp32 residual connections which may be faster but may impact accuracy. (default: False)
  --debug-ddp-parity-freq DEBUG_DDP_PARITY_FREQ
                        Set to value > 0 to debug DDP weight parity between ranks. (default: 0)
  --hybrid-override-pattern HYBRID_OVERRIDE_PATTERN
                        Override the hybrid override pattern in the config (specifies hyena layer ordering and type). (default: None)
  --num-layers NUM_LAYERS
                        If set, override the number of layers specified in the requested config. (default: None)
  --create-tflops-callback
                        Enable tflops calculation callback for Hyena / Evo2. Defaults to False. (default: False)
  --log-parameters-and-shapes
                        Log training parameters shapes and dtypes for debugging. (default: False)
  --lr LR               Learning rate. (default: 0.0003)
  --min-lr MIN_LR       Min learning rate in cosine annealing. (default: 3e-05)
  --warmup-steps WARMUP_STEPS
                        Number of warmup steps in cosine annealing (default: 2500)
  --nsys-profiling      Enable targeted `nsys` profiling on the training loop for a defined step range. To actually get profiling output you must run the whole program with `nsys`. For example: `nsys profile -s none -o output_report_name -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop [regular python command
                        here]` (default: False)
  --nsys-start-step NSYS_START_STEP
                        Start nsys profiling after this step. (default: 0)
  --nsys-end-step NSYS_END_STEP
                        End nsys profiling after this step. (default: None)
  --no-renormalize-loss
                        Do not renormalize the loss weights. (default: False)
  --nsys-ranks NSYS_RANKS [NSYS_RANKS ...]
                        Enable nsys profiling for these ranks. (default: [0])
  --activation-checkpoint-recompute-num-layers ACTIVATION_CHECKPOINT_RECOMPUTE_NUM_LAYERS
                        If set, override the default value set in the config. (default: None)
  --disable-checkpointing
                        Disable creating a ModelCheckpoint callback. (default: True)
  --clip-grad CLIP_GRAD
                        Grad clip value. Note that when using DDP this may need to be inflated. (default: 1.0)
  --seq-len-interpolation-factor SEQ_LEN_INTERPOLATION_FACTOR
                        Adjusts the linear scaling of ROPE (Rotary Position Embedding) for context extension. Set this factor relative to your base context length e.g., for an original context length of 8192 and an extended context length of 524288, use 524288/8192 = 64. (default: None)
  --overlap-param-gather
                        Overlap the parameter gather with the optimizer step. This is currently disabled due to a NeMo bug when using DDP. Making this an option defaulting to False is a temporary solution until the bug is fixed. (default: False)
  --overlap-grad-reduce
                        Overlap the gradient reduce with the optimizer step. (default: False)
  --hidden-dropout HIDDEN_DROPOUT
                        Dropout probability for the hyena layers (default: 0.0)
  --attention-dropout ATTENTION_DROPOUT
                        Dropout probability for the attention layers. (default: 0.0)
  --save-top-k SAVE_TOP_K
                        Number of best checkpoints to keep. Set to -1 to save all checkpoints. (default: 5)
  --metric-to-monitor-for-checkpoints METRIC_TO_MONITOR_FOR_CHECKPOINTS
                        Metric to monitor for checkpoints. (default: val_loss)
  --save-last-checkpoint
                        Save the last checkpoint. (default: True)
  --no-save-last-checkpoint
                        Disable saving the last checkpoint. (default: True)
  --no-activation-checkpointing
  --selective-activation-checkpointing
```

To supply a pre-trained checkpoint, pass the NeMo2 checkpoint directory to `--ckpt-dir`, and the script will dump newly trained checkpoints and logs to `--experiment-dir`. However, if there are existing well-defined checkpoints in the directory specified by `--experiment-dir`, the script will automatically resume training from the most recent checkpoint in the experiment directory instead of starting from the checkpoint specified by `--ckpt-dir`, which streamlines long training sessions. (To disable this behavior, supply a new or clean `--experiment-dir` when restarting from `--ckpt-dir`.)

Training data and sampling weights can be specified using the `--dataset-config` argument as a YAML file adhering to the following schema: [utils.config.Evo2BlendedDatasetConfig](src/bionemo/evo2/utils/config.py). For more information about dataset sampling and blending during training with Megatron-LM, refer to [megatron/core/datasets/readme.md](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/readme.md). For example:

```yaml
- dataset_prefix: /workspace/bionemo2/data/metagenomics/pretraining_data_metagenomics/data_metagenomics_train_text_CharLevelTokenizer_document
  dataset_split: train
  dataset_weight: 0.18
- dataset_prefix: /workspace/bionemo2/data/gtdb_imgpr/pretraining_data_gtdb_imgpr/data_gtdb_imgpr_train_text_CharLevelTokenizer_document
  dataset_split: train
  dataset_weight: 0.24
- dataset_prefix: /workspace/bionemo2/data/imgvr_untagged/imgvr_untagged_data/data_imgvr_train_text_CharLevelTokenizer_document
  dataset_split: train
  dataset_weight: 0.03
- dataset_prefix: /workspace/bionemo2/data/promoters/pretraining_data_promoters/data_promoters_valid_text_CharLevelTokenizer_document
  dataset_split: validation
  dataset_weight: 0.0003
- dataset_prefix: /workspace/bionemo2/data/organelle/pretraining_data_organelle/data_organelle_valid_text_CharLevelTokenizer_document
  dataset_split: validation
  dataset_weight: 0.005
- dataset_prefix: /workspace/bionemo2/data/metagenomics/pretraining_data_metagenomics/data_metagenomics_test_text_CharLevelTokenizer_document
  dataset_split: test
  dataset_weight: 0.18
- dataset_prefix: /workspace/bionemo2/data/gtdb_v220/gtdb_v220_imgpr_merged_data/data_gtdb_imgpr_test_text_CharLevelTokenizer_document
  dataset_split: test
  dataset_weight: 0.24
```

## Inference

Once you have a pre-trained or fine-tuned Evo2 checkpoint, you can also prompt the model to generate DNA sequences using the following command:

```bash
$ infer_evo2 --help
usage: infer_evo2 [-h] [--prompt PROMPT] --ckpt-dir CKPT_DIR [--temperature TEMPERATURE] [--top-k TOP_K] [--top-p TOP_P] [--max-new-tokens MAX_NEW_TOKENS] [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--pipeline-model-parallel-size PIPELINE_MODEL_PARALLEL_SIZE] [--context-parallel-size CONTEXT_PARALLEL_SIZE] [--output-file OUTPUT_FILE]

options:
  -h, --help            show this help message and exit
  --prompt PROMPT       Prompt to generate text from Evo2. Defaults to a phylogenetic lineage tag for E coli.
  --ckpt-dir CKPT_DIR   Path to checkpoint directory containing pre-trained Evo2 model.
  --temperature TEMPERATURE
                        Temperature during sampling for generation.
  --top-k TOP_K         Top K during sampling for generation.
  --top-p TOP_P         Top P during sampling for generation.
  --max-new-tokens MAX_NEW_TOKENS
                        Maximum number of tokens to generate.
  --tensor-parallel-size TENSOR_PARALLEL_SIZE
                        Order of tensor parallelism. Defaults to 1.
  --pipeline-model-parallel-size PIPELINE_MODEL_PARALLEL_SIZE
                        Order of pipeline parallelism. Defaults to 1.
  --context-parallel-size CONTEXT_PARALLEL_SIZE
                        Order of context parallelism. Defaults to 1.
  --output-file OUTPUT_FILE
                        Output file containing the generated text produced by the Evo2 model. If not provided, the output will be logged.
```

As in `train_evo2`, `--ckpt-dir` points to the NeMo2 checkpoint directory for Evo2 that you want to load for inference. `--output-file` can be used to dump the output into a `.txt` file, and if not specified the output will be logged in the terminal.

```
[NeMo I 2025-01-06 17:22:22 infer:102] ['CTCTTCTGGTATTTGG']
```

## Prediction

To run a forward pass of the Evo2 model, you can call `predict_evo2`, which processes a batch of sequences and returns either raw token logits or, if `--output-log-prob-seqs` is set, log-probability scores.

For example, to predict the log-probability scores of a batch of sequences saved to `fasta_path`, you can run the following command:

```bash
predict_evo2 \
  --fasta <fasta_path> \
  --ckpt-dir <PATH_TO_CHECKPOINT> \
  --output-dir <PATH_TO_OUTPUT_FILE> \
  --model-size 1b \
  --tensor-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --output-log-prob-seqs
```

An example of using `predict_evo2` for variant effect prediction can be found in our [Evo 2 Zeroshot BRCA1 Notebook](https://docs.nvidia.com/bionemo-framework/latest/user-guide/examples/bionemo-evo2/evo2_zeroshot_brca). This notebook demonstrates how to use Evo2 to predict whether single nucleotide variants (SNVs) in the BRCA1 gene are likely to be harmful to protein function and potentially increase cancer risk, by comparing the model's log probability scores between the reference and variant sequences.

## Context Extension

Evo2 supports continuing training with longer context lengths beyond those used to train a prior checkpoint. For example, when training the original Evo2 model, the first phase of training was performed at 8192 context length while the next phase continued training at 1m context length, but starting from the prior 8192 context length checkpoint. We call this process context extension.

To change the sequence length used in training in this way, supply the prior checkpoint as the `--ckpt-dir` argument, and set your new desired sequence length with `--seq-length`. Only doing these two things will run, but one issue is that the model's ROPE embeddings may not be scaled properly out of the box for a new context length. The way that Arc institute handled this was by setting the `--seq-len-interpolation-factor` to linearly scale the ROPE embedding for context extension. For example, if the base context length is 8192 and the extended context length is 65536, the factor would be 65536/8192 = 8. There are other ways of accomplishing this as well that may require some minor code changes, such as the approach used in llama-3, which is also available in megatron and could be added into argparse as an alternative.

## Checkpoint conversion from hugging face to NeMo2

The following conversion script should work on any savanna formatted arc evo2 checkpoint. Make sure you match up the
model size with the checkpoint you are converting.
The pyproject.toml makes the conversion script available as a command line tool `evo2_convert_to_nemo2`, so you
can try replacing:

```bash
evo2_convert_to_nemo2 \
  ...
```

with the following if you want to run with `-m pdb` or something:

```bash
python \
  sub-packages/bionemo-evo2/src/bionemo/evo2/utils/checkpoint/convert_to_nemo.py \
  ...
```

### 1b-8k

```bash
evo2_convert_to_nemo2 \
  --model-path hf://arcinstitute/savanna_evo2_1b_base \
  --model-size 1b --output-dir nemo2_evo2_1b_8k
```

This new checkpoint `nemo2_evo2_1b_8k` is ready to go in nemo2 format in downstream pretraining or prediction workflows.

#### Optional steps if you want to register the model with NGC

If you want to register the checkpoint with NGC (typically only NVIDIA employees) then you can do the following.

To create the checkpoint for distribution in NGC, first cd into the checkpiont directory:

```bash
cd nemo2_evo2_1b_8k
```

Then run the following command to make a tar of the full directory that gets unpacked into the current directory which
our NGC loader expects:

```bash
tar -czvf ../nemo2_evo2_1b_8k.tar.gz .
```

Finally `sha256sum` the tar file to get the checksum:

```bash
sha256sum nemo2_evo2_1b_8k.tar.gz
```

Then register it into the loader for testing purposes by editing
`sub-packages/bionemo-core/src/bionemo/core/data/resources/evo2.yaml`.

### 7b-8k

```bash
evo2_convert_to_nemo2 \
  --model-path hf://arcinstitute/savanna_evo2_7b_base \
  --model-size 7b --output-dir nemo2_evo2_7b_8k
```

### 7b-1M

```bash
evo2_convert_to_nemo2 \
  --model-path hf://arcinstitute/savanna_evo2_7b \
  --model-size 7b_arc_longcontext --output-dir nemo2_evo2_7b_1m
```

### 40b-8k

```bash
evo2_convert_to_nemo2 \
  --model-path hf://arcinstitute/savanna_evo2_40b_base \
  --model-size 40b --output-dir nemo2_evo2_40b_8k
```

### 40b-1M

```bash
evo2_convert_to_nemo2 \
  --model-path hf://arcinstitute/savanna_evo2_40b \
  --model-size 40b_arc_longcontext --output-dir nemo2_evo2_40b_1m
```
