# Initialization Guide

In BioNeMo Recipes, each model or recipe owns its own environment through a local `Dockerfile`, and
the repository also includes VS Code devcontainer configuration for interactive development. Each
model and recipe folder is a self-contained example.

Before starting, confirm that your host satisfies the
[Hardware and Software Prerequisites](./pre-reqs.md), including a working NVIDIA container runtime.

## Choose an Environment

### VS Code Devcontainer

For interactive development, open the repository in VS Code and run **Dev Containers: Reopen in
Container**. The top-level `.devcontainer` builds a development image and mounts useful host
state such as caches, SSH configuration, and `~/.netrc` into the container.

Some recipes also provide recipe-specific devcontainers, for e.g. megatron support. Use those when
you are working only in that recipe and want its narrower dependency set.

### Recipe Dockerfile

For command-line workflows, build the Docker image from the recipe or model directory you plan to
use. These Dockerfiles generally start from the NGC PyTorch base image and install the local
package plus that recipe's dependencies.

```bash
cd recipes/evo2_megatron
docker build -t bionemo-evo2:dev .
```

Use a different tag for each recipe image, for example `bionemo-esm2-native:dev` or
`bionemo-codonfm:dev`.

## Credentials

There are a few options for passing environment into running containers:

- Mount credential files read-only, such as `~/.netrc`, `~/.ngc`, `~/.aws`, or `~/.ssh`.
- Pass individual environment variables with `-e`, such as `WANDB_API_KEY` or `NGC_CLI_API_KEY`.
- Let the VS Code devcontainer mount the same host credential files during development.

For example:

```bash
docker run --rm -it --gpus all \
  -v "$HOME/.netrc:/root/.netrc:ro" \
  -v "$HOME/.ngc:/root/.ngc:ro" \
  -e WANDB_API_KEY \
  bionemo-evo2:dev \
  bash
```

## Runtime Mounts

Mount only the directories needed by the workflow. Common mounts are:

- The recipe source directory, when iterating locally.
- A dataset or checkpoint cache.
- An output directory for logs, checkpoints, and predictions.
- Credential files, mounted read-only.

Example interactive shell:

```bash
cd recipes/evo2_megatron
docker run --rm -it --gpus all \
  --ipc=host \
  --shm-size=16g \
  -v "$PWD:/workspace/bionemo" \
  -v "$HOME/.cache:/root/.cache" \
  -v "$HOME/.netrc:/root/.netrc:ro" \
  -v "$PWD/results:/workspace/results" \
  -e WANDB_API_KEY \
  bionemo-evo2:dev \
  bash
```

Example one-shot training command:

```bash
cd recipes/evo2_megatron
docker run --rm -it --gpus all \
  --ipc=host \
  --shm-size=16g \
  -v "$PWD:/workspace/bionemo" \
  -v "$HOME/.cache:/root/.cache" \
  -v "$HOME/.netrc:/root/.netrc:ro" \
  -v "$PWD/results:/workspace/results" \
  -e WANDB_API_KEY \
  bionemo-evo2:dev \
  train_evo2 --help
```

Replace the image tag and command with the recipe you are using. Recipe READMEs document the
supported entrypoints and recommended commands.

## Jupyter

If the recipe image includes Jupyter, expose a local port and keep notebooks inside a mounted
workspace or output directory:

```bash
docker run --rm -it --gpus all \
  --ipc=host \
  --shm-size=16g \
  -p 8888:8888 \
  -v "$PWD:/workspace/bionemo" \
  -v "$HOME/.cache:/root/.cache" \
  -v "$HOME/.netrc:/root/.netrc:ro" \
  bionemo-evo2:dev \
  jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser
```

Then open `http://localhost:8888`.

## Useful Docker Options

- `--gpus all`: make host GPUs visible inside the container.
- `--ipc=host` or `--shm-size=<size>`: provide enough shared memory for PyTorch dataloaders and
  distributed workloads.
- `-v <host>:<container>[:ro]`: mount data, outputs, caches, or credentials. Add `:ro` for
  credentials.
- `-e NAME` or `-e NAME=value`: pass an environment variable into the container.
- `-u $(id -u):$(id -g)`: run as your host user when you need outputs owned by your local UID/GID.
