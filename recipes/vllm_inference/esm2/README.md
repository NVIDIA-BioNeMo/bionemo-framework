# ESM-2 vLLM Inference

This recipe demonstrates running inference on
[ESM-2 TE checkpoints](../../../models/esm2/) using
[vLLM](https://github.com/vllm-project/vllm) (>= 0.14) as a pooling/embedding model.

The exported TE checkpoints on HuggingFace Hub are directly compatible with vLLM.
No conversion scripts or weight renaming are needed:

```python
from vllm import LLM

model = LLM(
    model="nvidia/esm2_t6_8M_UR50D",
    runner="pooling",
    trust_remote_code=True,
    enforce_eager=True,
    max_num_batched_tokens=1026,
)

prompts = ["MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK"]
outputs = model.embed(prompts)
print(outputs[0].outputs.embedding[:5])
```

See [tests/test_vllm.py](tests/test_vllm.py) for a full golden-value validation across
vLLM, native HuggingFace, and the nvidia Hub reference model.

## Installing vLLM in the container

There are two ways to get vLLM installed in the Docker image.

**Option 1: Build-time installation via Dockerfile build arg**

Pass `--build-arg INSTALL_VLLM=true` and `--build-arg TORCH_CUDA_ARCH_LIST=<arch>` when
building the image. `TORCH_CUDA_ARCH_LIST` is required when `INSTALL_VLLM=true` (the
Dockerfile will error if it is not set):

```bash
docker build -t esm2-vllm \
  --build-arg INSTALL_VLLM=true \
  --build-arg TORCH_CUDA_ARCH_LIST="9.0" .
```

**Option 2: Post-build installation via `install_vllm.sh`**

Build the base image normally, then run `install_vllm.sh` inside the container. The script
auto-detects the GPU architecture, or you can pass an explicit arch argument:

```bash
docker build -t esm2 .
docker run --rm -it --gpus all esm2 bash -c "./install_vllm.sh"
# or with an explicit architecture:
docker run --rm -it --gpus all esm2 bash -c "./install_vllm.sh 9.0"
```
