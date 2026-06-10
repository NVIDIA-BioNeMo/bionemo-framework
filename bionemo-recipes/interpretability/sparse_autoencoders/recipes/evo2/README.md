# Evo2 SAE recipe

Sparse Autoencoders for the [Evo2](../../../../recipes/evo2_megatron) DNA language model:
offline activation extraction + SAE training, a **live inference engine** (encode / steered
generation) with an HTTP server, and a feature-explorer **dashboard**.

```
recipes/evo2/
├── scripts/
│   ├── extract.py · train.py · chunk_fasta.py   # offline: activations -> SAE
│   ├── launch_inference.sh                       # live: serve / encode / batch
│   └── launch_dashboard.py                        # serve the dashboard on provided data
├── src/evo2_sae/        core.py (engine) · server.py · cli.py
├── feature_explorer/    React/Vite dashboard (4 panels)
└── tests/
```

The recipe reuses the evo2_megatron recipe for the model (`predict.load_model_to_layer`,
`infer.generate`) and the shared `sae` package for the autoencoder; `src/evo2_sae/` is only the
SAE layer (encode, the decode-only feature-clamp hook, feature labels, the serve object).

## 0. Environment

The recipe runs inside the **evo2_megatron venv** (provides `bionemo.evo2` + megatron + TE):

```bash
cd ../../../recipes/evo2_megatron && bash .ci_build.sh   # builds ./.venv (~15–30 min)
export VENV=$PWD/.venv
```

> The venv must include `predict.load_model_to_layer` and `infer.setup_inference_engine`
> (current `main`). `launch_inference.sh` reads `$VENV`; point it at the venv you built.

## 1. Config (defaults target 7B / layer 26)

```bash
export EVO2_CKPT_DIR=/data/interp/evo2/checkpoints/evo2_7b_mbridge
export SAE_CKPT_PATH=.../sae/v2_diverse/layer26_7B_ablate_normalize_input/checkpoints/checkpoint_final.pt
export EMBEDDING_LAYER=26
export FEATURE_ANNOTATIONS=.../sae_eval/dashboard_data/l26_7B_normalize/feature_metadata.parquet
export CUDA_VISIBLE_DEVICES=0
```

## 2. CLI inference

```bash
cd scripts
./launch_inference.sh encode --sequence ATGGCC...GTGCAT --organism "Human" --top-k 8   # one seq -> JSON
./launch_inference.sh batch  --fasta in.fa --out out.parquet                            # FASTA -> parquet
```

## 3. Inference server (dashboard backend)

```bash
./launch_inference.sh serve            # FastAPI on :8001 — /health /features /annotate /generate
curl localhost:8001/health
```

## 4. Dashboard

The dashboard reads atlas parquets **you provide** (it does not generate them):

```bash
cd ..   # recipes/evo2
# DIR must hold features_atlas.parquet, feature_metadata.parquet, feature_examples.parquet
"$VENV/bin/python" scripts/launch_dashboard.py --data-dir /path/to/dashboard_data
```

The **Feature atlas** tab is static (served from those parquets); **Sequence inspector** and
**Generative steering** call the server from step 3. See `feature_explorer/README.md`.

## 5. Tests

```bash
# CPU (no model):
PYTHONPATH=src "$VENV/bin/python" -m pytest tests/test_server.py tests/test_launch_dashboard.py \
    tests/test_steering.py::test_clamp_math -q
# GPU (slow, gated by the step-1 env vars): encode + steering on the real model
PYTHONPATH=src "$VENV/bin/python" -m pytest tests/test_steering.py -q
```

## Notes

- **Two-model design:** a truncated model (encode/inspect, loaded eagerly) + the full inference
  engine (generate, lazy). Keeps inspect cheap; the engine loads on first `serve`/`generate`.
- Generating the dashboard atlas parquets is a separate offline step (not yet in-recipe).
