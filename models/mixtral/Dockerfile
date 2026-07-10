FROM nvcr.io/nvidia/pytorch:26.06-py3
WORKDIR /workspace/bionemo
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_CONSTRAINT= pip install -r requirements.txt
