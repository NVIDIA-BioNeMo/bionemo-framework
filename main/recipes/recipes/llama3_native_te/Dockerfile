# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:26.03-py3

# Rebuild TransformerEngine from main branch (includes PR #2753: MXFP8 FusedAdam support).
# To pin a specific commit, replace 'main' with a commit hash.
# Build: docker build -t llama3_native_te:te-main-26.03 .
RUN pip uninstall -y transformer_engine transformer_engine_torch transformer_engine_cu12 && \
    git clone --recursive https://github.com/NVIDIA/TransformerEngine.git /opt/te && \
    cd /opt/te && git checkout main && \
    NVTE_FRAMEWORK=pytorch MAX_JOBS=8 pip install --no-build-isolation . && \
    rm -rf /opt/te

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    PIP_CONSTRAINT= pip install -r /requirements.txt

WORKDIR /workspace/bionemo
COPY . .
