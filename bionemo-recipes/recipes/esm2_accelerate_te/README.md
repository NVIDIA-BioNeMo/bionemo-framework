# TransformerEngine-accelerated ESM-2 training with Hugging Face Trainer

This folder demonstrates how to train TE-accelerated ESM-2 using the [Hugging Face Transformers Trainer](https://huggingface.co/docs/transformers/en/trainer) class and [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch), including sequence packing and FP8 precision, using distributed training frameworks like FSDP and DeepSpeed.

## How to use this recipe

This folder contains an independent, minimal training example. It does not depend on any other code in the top-level
bionemo-framework repository. You can download a zipped directory of this folder alone by clicking
[here](https://download-directory.github.io?url=https://github.com/NVIDIA-BioNeMo/bionemo-framework/tree/main/bionemo-recipes/recipes/esm2_accelerate_te&filename=esm2-accelerate-te).

### How to deploy this recipe on cloud providers

🚧 Under development

## Supported Models and Training Features

| Model                                     | BF16 | FP8<sup>[1]</sup> | THD Input Format | FP8 with THD Input Format | MXFP8<sup>[2]</sup> | Context Parallelism |
| ----------------------------------------- | ---- | ----------------- | ---------------- | ------------------------- | ------------------- | ------------------- |
| [ESM-2](../../models/esm2/README.md)      | ✅   | ✅                | 🚧               | 🚧                        | 🚧                  | ❌                  |
| [AMPLIFY](../../models/amplify/README.md) | ✅   | ❌                | 🚧               | ❌                        | ❌                  | ❌                  |

✅: Supported <br/>
🚧: Under development <br/>
❌: Not supported <br/>

\[1\]: Requires compute capacity 9.0 and above (Hopper+) <br/>
\[2\]: Requires compute capacity 10.0 and 10.3 (Blackwell), 12.0 support pending <br/>

### Distributed Training

This recipe leverages [Hugging Face Accelerate](https://huggingface.co/docs/accelerate) for distributed training, which supports multiple distributed training frameworks through configuration files:

- [Distributed Data Parallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), shown in `accelerate_config/default.yaml`
- [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html), shown in
  `accelerate_config/fsdp1_te.yaml` and `accelerate_config/fsdp1_hf.yaml` (depending on whether the model is
  TransformerEngine-accelerated or not)
- [Fully Sharded Data Parallel 2 (FSDP2)](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html), shown in
  `accelerate_config/fsdp2_te.yaml` and `accelerate_config/fsdp2_hf.yaml`

The training strategy is configured through Accelerate's configuration system rather than separate training scripts.

## Commands to Launch Training

### Single Process Training

To run single-process training on one GPU:

```bash
python train.py --config-name=L0_sanity
```

To train the AMPLIFY model instead of ESM-2, use the `L0_sanity_amplify` config:

```bash
python train.py --config-name=L0_sanity_amplify
```

### Multi-Process Training

To run distributed training using Accelerate's launch command:

```bash
accelerate launch --config_file accelerate_config/fsdp2_te.yaml \
    --num_processes 2 train.py \
    --config-name=L0_sanity
```

### Multi-Node Training

For multi-node training, configure your accelerate setup with the appropriate machine rank, main process IP, and port:

```bash
accelerate launch --config_file accelerate_config/fsdp2_te.yaml \
    --main_process_ip 192.168.20.1 \
    --main_process_port 9898 \
    --num_machines 2 \
    --machine_rank 0 \
    train.py --config-name=L0_sanity
```

Refer to [`slurm.sh`](slurm.sh) for an example SLURM script.

### FP8 Training

FP8 precision is enabled with an accelerate config file, shown in `accelerate_config/fp8.yaml`.

```bash
accelerate launch --config_file accelerate_config/fp8.yaml \
    train.py --config-name L0_sanity
```

### Torch Dynamo (torch.compile) Support

An example accelerate config file, shown in `accelerate_config/dynamo.yaml`, is provided for torch.compile support.

```bash
accelerate launch --config_file accelerate_config/dynamo.yaml \
    train.py --config-name L0_sanity
```

### Known Limitations

- Combining FP8 and FSDP1 or FSDP2 does not seem to be supported currently.

## Saving and Loading Checkpoints

The Hugging Face Trainer automatically handles checkpointing based on the `TrainingArguments` configuration.
Checkpointing behavior is controlled through the trainer configuration in your hydra config.

### Enabling Checkpoints

To enable checkpoint saving, ensure that `trainer.output_dir` is set to a writable directory. Checkpointing frequency is
controlled by the `trainer.save_steps` configuration parameter.

```bash
accelerate launch train.py --config-name L0_sanity \
  trainer.output_dir=/path/to/ckpt_dir \
  trainer.save_steps=100
```

### Resuming from Checkpoints

The Trainer automatically detects and resumes from the latest checkpoint in the output directory. You can also specify a
specific checkpoint:

```bash
accelerate launch train.py --config-name L0_sanity \
  trainer.output_dir=/path/to/ckpt_dir \
  trainer.resume_from_checkpoint=/path/to/specific/checkpoint
```

### Evaluation

Configure evaluation strategy and frequency through TrainingArguments:

```bash
accelerate launch train.py --config-name L0_sanity \
  trainer.eval_strategy=steps \
  trainer.eval_steps=500 \
```

## Running Inference with the Trained Model

Models trained with this recipe can be loaded using standard Hugging Face methods. The final model is saved in a format compatible with the `AutoModel.from_pretrained` method:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("path/to/checkpoint")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

gfp_P42212 = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
    "VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLV"
    "NRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLAD"
    "HYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

inputs = tokenizer(gfp_P42212, return_tensors="pt")
model.eval()
output = model(**inputs)
```

## Performance

🚧 Under development

## References

- [ESM-2 Training with Native PyTorch](../esm2_native_te/README.md)
- [Hugging Face Trainer Documentation](https://huggingface.co/docs/transformers/en/trainer)
- [Hugging Face Accelerate Documentation](https://huggingface.co/docs/accelerate)

## Developer Guide

### Running Tests

To run tests locally, run `recipes_local_test.py` from the repository root with the recipe directory as an argument.

```bash
./ci/scripts/recipes_local_test.py bionemo-recipes/recipes/esm2_accelerate_te/
```

Tests should be kept relatively fast, using the smallest model and number of training steps required to validate the feature. Hardware requirements beyond those used in CI, like a single L4, should be annotated with pytest.mark.requires, such as `requires_fp8` and `requires_multi_gpu`.

### Development Container

To use the provided devcontainer, use "Dev Containers: Reopen in Container" from the VSCode menu, and choose the "BioNeMo Recipes Dev Container" option. To run the tests inside the container, run `pytest -v .` in the recipe directory.

### Hydra Tips

[Hydra](https://hydra.cc/) is a powerful configuration management library for Python. This recipe uses Hydra to manage training configurations, allowing for easy modification of training hyper-parameters and model settings.

Configuration parameters can be overridden from the command line, for example:

```bash
accelerate launch train.py --config-name L0_sanity fp8_config.enabled=true trainer.learning_rate=2e-5
```
