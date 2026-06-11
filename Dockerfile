# Build instructions:
#
# For x86_64/amd64 (default):
#   docker build -t bionemo .
#   # Or explicitly:
#   docker build --build-arg TARGETARCH=amd64 -t bionemo .
#
# For ARM64:
#   docker build --build-arg TARGETARCH=arm64 -t bionemo .
#
# For multi-platform build:
#   docker buildx create --use
#   docker buildx build --platform linux/amd64,linux/arm64 -t bionemo .
#
# Base image with apex and transformer engine, but without NeMo or Megatron-LM.
#  Note that the core NeMo docker container is defined here:
#   https://gitlab-master.nvidia.com/dl/JoC/nemo-ci/-/blob/main/llm_train/Dockerfile.train
#  with settings that get defined/injected from this config:
#   https://gitlab-master.nvidia.com/dl/JoC/nemo-ci/-/blob/main/.gitlab-ci.yml
#  We should keep versions in our container up to date to ensure that we get the latest tested perf improvements and
#   training loss curves from NeMo.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3

FROM rust:1.86.0 AS rust-env

RUN rustup set profile minimal && \
  rustup install 1.82.0 && \
  if [ "$TARGETARCH" = "arm64" ]; then \
    rustup target add aarch64-unknown-linux-gnu; \
  else \
    rustup target add x86_64-unknown-linux-gnu; \
  fi && \
  rustup default 1.82.0

FROM ${BASE_IMAGE} AS bionemo2-base
# Default to amd64 if no TARGETARCH is specified
ARG TARGETARCH=amd64

# Install core apt packages.
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
  <<EOF
set -eo pipefail
apt-get update -qy
apt-get install -qyy \
  libsndfile1 \
  ffmpeg \
  git \
  curl \
  pre-commit \
  sudo \
  emacs-nox \
  gnupg \
  unzip \
  libsqlite3-dev
apt-get upgrade -qyy \
  rsync
rm -rf /tmp/* /var/tmp/*
EOF


## BUMP and patch TE as a solution to the issues:
## 1. https://github.com/NVIDIA/bionemo-framework/issues/422
## 2. https://github.com/NVIDIA/bionemo-framework/issues/973
## Drop this when pytorch images ship the fixed commit.
ARG TE_TAG=9d4e11eaa508383e35b510dc338e58b09c30be73

COPY ./docker_build_patches/te.patch /tmp/te.patch
RUN git clone --recurse-submodules https://github.com/NVIDIA/TransformerEngine.git /tmp/TransformerEngine && \
    cd /tmp/TransformerEngine && \
    git checkout --recurse-submodules ${TE_TAG} && \
    patch -p1 < /tmp/te.patch && \
    PIP_CONSTRAINT= NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi \
    pip --disable-pip-version-check --no-cache-dir install .

# Install AWS CLI from source rather than prebuilt binary.
# This is good for two reasons:
#  1. It is the same on both ARM and x86
#  2. When installing this way, aws-cli doesn't bring its own pypi dependencies with it,
#     which it might do via a binary install. These extra pypi packages sometimes bring
#     CVEs with them.

RUN <<EOF
set -eo pipefail
cd /tmp
git clone --depth 1 --branch 2.27.59 https://github.com/aws/aws-cli.git
cd aws-cli
pip install .
cd /
rm -rf /tmp/aws-cli
EOF

# Use a branch of causal_conv1d while the repository works on Blackwell support.
ARG CAUSAL_CONV_TAG=52e06e3d5ca10af0c7eb94a520d768c48ef36f1f
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE pip --disable-pip-version-check --no-cache-dir install git+https://github.com/trvachov/causal-conv1d.git@${CAUSAL_CONV_TAG}

###############################################################################
# ARM
###############################################################################
# Certain dependencies do not have prebuild ARM wheels/binaries, so we build them
# from source here. Overall, ecosystem ARM support is much weaker than x86, so below
# you'll see some hardcoded patches/versions/experimental branches to get i
# everything to work.

# Decord installation
RUN --mount=type=bind,source=./docker_build_patches/decord_ffmpeg6_fix.patch,target=/decord_ffmpeg6_fix.patch \
    if [ "$TARGETARCH" = "arm64" ]; then \
    export BUILD_DIR=/build && mkdir ${BUILD_DIR} && cd ${BUILD_DIR} && \
    apt-get update && \
    apt-get install -y build-essential python3-dev python3-setuptools make cmake && \
    apt-get install -y ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev && \
    git clone --recursive https://github.com/dmlc/decord && \
    cd decord && \
    git apply /decord_ffmpeg6_fix.patch && \
    mkdir build && cd build && \
    cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release && \
    make && \
    cd ../python && \
    pip install . && \
    cd / && rm -rf ${BUILD_DIR}; \
fi

# TileDB installation
RUN if [ "$TARGETARCH" = "arm64" ]; then \
    mkdir -p /usr/lib/tiledb && \
    cd /usr/lib/tiledb && \
    wget https://github.com/TileDB-Inc/TileDB/releases/download/2.27.2/tiledb-linux-arm64-2.27.2-1757013.tar.gz -O tiledb.tar.gz && \
    tar -xvzf tiledb.tar.gz && export TILEDB_PATH=/usr/lib/tiledb && \
    cd / && \
    dpkg -l | awk '/libfmt/ {print $2}' | xargs apt-get remove -y && \
    dpkg -l | awk '/spdlog/ {print $2}' | xargs apt-get remove -y && \
    rm -f /usr/lib/*/cmake/spdlog/spdlogConfig.cmake && \
    rm -f /usr/lib/cmake/spdlog/spdlogConfig.cmake && \
    git clone --single-branch --branch 1.16.1 https://github.com/single-cell-data/TileDB-SOMA.git && \
    cd TileDB-SOMA/apis/python && \
    pip install .; \
fi

###############################################################################
# /end ARM
###############################################################################

# Fix the version of scikit-misc to 0.3.1 because newer versions of scikit-misc require numpy >= 2.0 to be built.
# Since there are not pre-built wheels for arm64, we need to install this specific version.
# Once bionemo is compatible with numpy >= 2.0, we can remove this.
# Technically, this is only needed for the ARM build, but we apply to all architectures to avoid library version
# divergence.
RUN apt-get update -qy && apt-get install -y libopenblas-dev && pip install scikit-misc==0.3.1

# Mamba dependancy installation
# See https://gitlab-master.nvidia.com/dl/JoC/nemo-ci/-/blob/d3c853c2d/docker/Dockerfile#L193-198
#  for the command we want to keep in sync. Note that the package install is modified slightly to first build a
#  wheel (no deps) and then install the wheel. This avoids issues with torch version conflicts.
RUN <<EOF
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout 2e16fc3062cdcd4ebef27a9aa4442676e1c7edf4
sed -i "/triton/d" setup.py
sed -i "/triton/d" pyproject.toml
pip3 wheel --disable-pip-version-check --no-build-isolation --no-deps .
pip3 --disable-pip-version-check --no-cache-dir install mamba_ssm-*.whl --no-deps
rm -f mamba_ssm-*.whl
cd ..
rm -rf mamba
EOF


# Nemo Run installation
# Some things are pip installed in advance to avoid dependency issues during nemo_run installation
RUN pip install hatchling urllib3  # needed to install nemo-run
ARG NEMU_RUN_TAG=v0.3.0
RUN pip install nemo_run@git+https://github.com/NVIDIA/NeMo-Run.git@${NEMU_RUN_TAG} --use-deprecated=legacy-resolver

# Rapids SingleCell Installation
RUN pip install 'rapids-singlecell' --extra-index-url=https://pypi.nvidia.com

RUN mkdir -p /workspace/bionemo2/

WORKDIR /workspace

# Addressing Security Scan Vulnerabilities
RUN rm -rf /opt/pytorch/pytorch/third_party/onnx


# Use UV to install python packages from the workspace. This just installs packages into the system's python
# environment, and does not use the current uv.lock file. Note that with python 3.12, we now need to set
# UV_BREAK_SYSTEM_PACKAGES, since the pytorch base image has made the decision not to use a virtual environment and UV
# does not respect the PIP_BREAK_SYSTEM_PACKAGES environment variable set in the base dockerfile.
COPY --from=ghcr.io/astral-sh/uv:0.6.13 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
  UV_COMPILE_BYTECODE=1 \
  UV_PYTHON_DOWNLOADS=never \
  UV_SYSTEM_PYTHON=true \
  UV_BREAK_SYSTEM_PACKAGES=1

COPY --from=rust-env /usr/local/cargo /usr/local/cargo
COPY --from=rust-env /usr/local/rustup /usr/local/rustup

ENV PATH="/usr/local/cargo/bin:/usr/local/rustup/bin:${PATH}"
ENV RUSTUP_HOME="/usr/local/rustup"

WORKDIR /workspace/bionemo2

# Install 3rd-party deps and bionemo submodules.
COPY ./LICENSE /workspace/bionemo2/LICENSE
COPY ./3rdparty /workspace/bionemo2/3rdparty
COPY ./sub-packages /workspace/bionemo2/sub-packages

RUN --mount=type=bind,source=./requirements-test.txt,target=/requirements-test.txt \
  --mount=type=bind,source=./requirements-cve.txt,target=/requirements-cve.txt \
  --mount=type=cache,target=/root/.cache <<EOF
set -eo pipefail
uv pip install maturin --no-build-isolation
# install nvidia-resiliency-ext separately because it doesn't yet have ARM wheels
git clone https://github.com/NVIDIA/nvidia-resiliency-ext
uv pip install nvidia-resiliency-ext/
rm -rf nvidia-resiliency-ext/
# ngcsdk causes strange dependency conflicts (ngcsdk requires protobuf<4, but nemo_toolkit requires protobuf==4.24.4, deleting it from the uv pip install prevents installation conflicts)
sed -i "/ngcsdk/d" ./sub-packages/bionemo-core/pyproject.toml
# Remove llama-index because bionemo doesn't use it and it adds CVEs to container
sed -i "/llama-index/d" ./3rdparty/NeMo/requirements/requirements_nlp.txt
# Pin 'nvidia-modelopt' to 0.27.1 due to an API incompatibility of version 0.25.0
sed -i -E "s|nvidia-modelopt\[torch\]>=[^,]+,<=([^ ;]+)|nvidia-modelopt[torch]==\1|" ./3rdparty/NeMo/requirements/requirements_nlp.txt
uv pip install --no-build-isolation \
./3rdparty/*  \
./sub-packages/bionemo-* \
-r /requirements-cve.txt \
-r /requirements-test.txt

# Install back ngcsdk, as a WAR for the protobuf version conflict with nemo_toolkit.
uv pip install ngcsdk==4.9.17  # latest version, 12/03/2025.
# Install >=0.46.1 bitsandbytes specifically because it has CUDA>12.9 support.
# TODO(trvachov) remove this once it stops conflicting with strange NeMo requirements.txt files
uv pip uninstall bitsandbytes && uv pip install bitsandbytes==0.46.1

# Addressing security scan issue - CVE vulnerability https://github.com/advisories/GHSA-g4r7-86gm-pgqc The package is a
# dependency of lm_eval from NeMo requirements_eval.txt.
uv pip uninstall sqlitedict

rm -rf ./3rdparty
rm -rf /tmp/*
rm -rf ./sub-packages/bionemo-noodles/target
EOF

# In the devcontainer image, we just copy over the finished `dist-packages` folder from the build image back into the
# base pytorch container. We can then set up a non-root user and uninstall the bionemo and 3rd-party packages, so that
# they can be installed in an editable fashion from the workspace directory. This lets us install all the package
# dependencies in a cached fashion, so they don't have to be built from scratch every time the devcontainer is rebuilt.
FROM ${BASE_IMAGE} AS dev

RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
  <<EOF
set -eo pipefail
apt-get update -qy
apt-get install -qyy \
  sudo
rm -rf /tmp/* /var/tmp/*
EOF

# Use a non-root user to use inside a devcontainer (with ubuntu 23 and later, we can use the default ubuntu user).
ARG USERNAME=ubuntu
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

# Here we delete the dist-packages directory from the pytorch base image, and copy over the dist-packages directory from
# the build image. This ensures we have all the necessary dependencies installed (megatron, nemo, etc.).
RUN <<EOF
  set -eo pipefail
  rm -rf /usr/local/lib/python3.12/dist-packages
  mkdir -p /usr/local/lib/python3.12/dist-packages
  chmod 777 /usr/local/lib/python3.12/dist-packages
  chmod 777 /usr/local/bin
EOF

USER $USERNAME

COPY --from=bionemo2-base --chown=$USERNAME:$USERNAME --chmod=777 \
  /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages

COPY --from=ghcr.io/astral-sh/uv:0.6.13 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
  UV_COMPILE_BYTECODE=0 \
  UV_PYTHON_DOWNLOADS=never \
  UV_SYSTEM_PYTHON=true \
  UV_BREAK_SYSTEM_PACKAGES=1

# Bring in the rust toolchain, as maturin is a dependency listed in requirements-dev
COPY --from=rust-env /usr/local/cargo /usr/local/cargo
COPY --from=rust-env /usr/local/rustup /usr/local/rustup

ENV PATH="/usr/local/cargo/bin:/usr/local/rustup/bin:${PATH}"
ENV RUSTUP_HOME="/usr/local/rustup"

RUN --mount=type=bind,source=./requirements-dev.txt,target=/workspace/bionemo2/requirements-dev.txt \
  --mount=type=cache,target=/root/.cache <<EOF
  set -eo pipefail
  uv pip install -r /workspace/bionemo2/requirements-dev.txt
  rm -rf /tmp/*
EOF

RUN <<EOF
  set -eo pipefail
  rm -rf /usr/local/lib/python3.12/dist-packages/bionemo*
  pip uninstall -y nemo_toolkit megatron_core
EOF


# Transformer engine attention defaults
# FIXME the following result in unstable training curves even if they are faster
#  see https://github.com/NVIDIA/bionemo-framework/pull/421
#ENV NVTE_FUSED_ATTN=1 NVTE_FLASH_ATTN=0
FROM dev AS development

WORKDIR /workspace/bionemo2
COPY --from=bionemo2-base /workspace/bionemo2/ .
# because of the `rm -rf ./3rdparty` in bionemo2-base
COPY ./3rdparty ./3rdparty

USER root
COPY --from=rust-env /usr/local/cargo /usr/local/cargo
COPY --from=rust-env /usr/local/rustup /usr/local/rustup

ENV PATH="/usr/local/cargo/bin:/usr/local/rustup/bin:${PATH}"
ENV RUSTUP_HOME="/usr/local/rustup"

RUN <<EOF
set -eo pipefail
find . -name __pycache__ -type d -print | xargs rm -rf
for sub in ./3rdparty/* ./sub-packages/bionemo-*; do
    uv pip install --no-deps --no-build-isolation --editable $sub
done
EOF

# Since the entire repo is owned by root, switching username for development breaks things.
ARG USERNAME=ubuntu
RUN chown $USERNAME:$USERNAME -R /workspace/bionemo2/
USER $USERNAME

# The 'release' target needs to be last so that it's the default build target. In the future, we could consider a setup
# similar to the devcontainer above, where we copy the dist-packages folder from the build image into the release image.
# This would reduce the overall image size by reducing the number of intermediate layers. In the meantime, we match the
# existing release image build by copying over remaining files from the repo into the container.
FROM bionemo2-base AS release

RUN mkdir -p /workspace/bionemo2/.cache/

COPY VERSION .
COPY ./README.md ./
# Copy over folders so that the image can run tests in a self-contained fashion.
COPY ./ci/scripts ./ci/scripts
COPY ./docs ./docs

COPY --from=rust-env /usr/local/cargo /usr/local/cargo
COPY --from=rust-env /usr/local/rustup /usr/local/rustup

# Fix a CRIT vuln: https://github.com/advisories/GHSA-vqfr-h8mv-ghfj
RUN uv pip install h11==0.16.0

# RUN rm -rf /usr/local/cargo /usr/local/rustup
RUN chmod 777 -R /workspace/bionemo2/

# Transformer engine attention defaults
# We have to declare this again because the devcontainer splits from the release image's base.
# FIXME the following results in unstable training curves even if faster.
#  See https://github.com/NVIDIA/bionemo-framework/pull/421
# ENV NVTE_FUSED_ATTN=1 NVTE_FLASH_ATTN=0
