# Evo2 Checkpoint Conversion Library

This library contains helper scripts for converting checkpoint formats for Evo2.

## Converting ZeRO-1 / PyTorch Checkpoints to NeMo2 Checkpoints

To convert a single PyTorch or ZeRO-1 checkpoints (`.pt`) into NeMo2 format, run the following command:

```
python sub-packages/bionemo-evo2/src/bionemo/evo2/utils/checkpoint/convert_to_nemo.py --model-path <CKPT_FILE> --output-dir <OUTPUT_DIR> --model-size <MODEL_SIZE> --ckpt-format <CONVERTED_CKPT_FORMAT>
```

where `--model-size` can be set to `7b` or `40b` (or their `_arc_1m` variants with modified GLU dimensions) and `--ckpt-format` can be set to `torch_dist` or `zarr`.

The NeMo2 checkpoint should have the following structure for `torch_dist`:

```
default--val_loss=2.3738-epoch=0-consumed_samples=800.0-last
├── context
│   ├── io.json
│   └── model.yaml
└── weights
    ├── __*_*.distcp
    ├── common.pt
    └── metadata.json
```

and the following structure for `zarr`:

```
interleaved_hyena_7b_fix_shape
├── context
│   ├── io.json
│   └── model.yaml
└── weights
    ├── common.pt
    ├── metadata.json
    └── <MODEL_LAYER_NAME>  # Example: module.decoder.layers.0.mixer.dense
        └── shard_*_*.pt
```

## Converting ZeRO-1 MP{N} to ZeRO-1 MP1

To convert sharded (MP>1) ZeRO-1 checkpoints to un-sharded (MP1) checkpoints (or any order of model parallelism) compatible with the `convert_to_nemo.py` conversion script, you can run the following command:

```
python sub-packages/bionemo-evo2/src/bionemo/evo2/utils/checkpoint/convert_checkpoint_model_parallel_evo2.py --source_dir <CKPT_DIR> --output_dir <OUTPUT_DIR> --mp_size <TARGET_MODEL_PARALLEL_SIZE>
```

ZeRO-1 checkpoints should have the following structure:

```
arc_7b_tp8_pretrained_ckpt/global_step199400
└── mp_rank_*_model_states.pt
```

## Converting ZeRO-3 to ZeRO-1

To convert ZeRO-3 checkpoints into ZeRO-1 checkpoints, run the following command:

```
python sub-packages/bionemo-evo2/src/bionemo/evo2/utils/checkpoint/convert_zero3_to_zero1.py <INPUT_DIR> <OUTPUT_DIR> --overwrite --mp_size <MODEL_PARALLEL_SIZE>
```

ZeRO-3 checkpoints should have the following structure:

```
arc_40b_zero3_w32_mp8_test_notfinal_ckpt/global_step1
├── bf16_zero_pp_rank_*_mp_rank_*_optim_states.pt
├── configs
│   ├── 40b_test_chkpt.yml
│   └── opengenome.yml
└── zero_pp_rank_*_mp_rank_*_model_states.pt
```
