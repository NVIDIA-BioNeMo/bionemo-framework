# To build a production image:
# docker build -t <image_name> --target production .
#
# To build a development image:
# docker build -t <image_name> --target development --build-arg USERNAME=$(whoami) --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) .

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:25.08-py3
FROM ${FROM_IMAGE_NAME} AS base

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

WORKDIR /workspace/codonfm

# Copy requirements first for better layer caching
COPY ./requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt \
    && apt-get update \
    && apt-get install -y libsndfile1 ffmpeg



# ----------------- Production Stage -----------------
FROM base AS production

WORKDIR /workspace/codonfm
COPY . .
# Add a CMD for production if you have a main script to run
# e.g., CMD ["python", "app.py"]


# ----------------- Development Stage -----------------
FROM base AS development

WORKDIR /workspace/codonfm

# Install development specific packages
RUN apt-get update && apt-get install -y htop rsync

# Create a non-root user for development (robust to empty args)
ARG USERNAME
ARG USER_UID
ARG USER_GID

RUN set -eux; \
    USERNAME_EFF="${USERNAME:-vscode}"; \
    USER_UID_EFF="${USER_UID:-1001}"; \
    USER_GID_EFF="${USER_GID:-1001}"; \
    if ! getent group "$USER_GID_EFF" >/dev/null 2>&1; then \
      groupadd --gid "$USER_GID_EFF" "$USERNAME_EFF"; \
    fi; \
    if ! id -u "$USERNAME_EFF" >/dev/null 2>&1; then \
      useradd -l --uid "$USER_UID_EFF" --gid "$USER_GID_EFF" -m "$USERNAME_EFF"; \
    fi  # noqa: E702

# Copy source code into the image for development
COPY . .

# Ensure developer owns the app directory
RUN chown -R ${USERNAME:-vscode}:${USERNAME:-vscode} /workspace/codonfm

# Switch to the non-root user
USER $USERNAME

# ----------------- For benchmarking only -----------------
# Warning: I was only able to build this image in an instance with 2TB of memory.
# Otherwise, a segmentation fault occurs during the build process:
# /bin/bash: line 1: 13517 Segmentation fault      (core dumped) ptxas -arch=sm_90 -m64 -v --generate-line-info "/tmp/tmpxft_00002f58_00000000-6_flash_fwd_hdim64_256_fp16_paged_split_sm90.ptx" -o "/tmp/tmpxft_00002f58_00000000-8_flash_fwd_hdim64_256_fp16_paged_split_sm90.cubin" > /tmp/tmpxft_00002f58_00000000-10_2eb7d280_stdout 2> /tmp/tmpxft_00002f58_00000000-10_2eb7d280_stderr
#
# Could have also been caused by CUDA-compatibility issues.

FROM base AS benchmarking

WORKDIR /workspace/codonfm

RUN pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@v0.0.32.post2#egg=xformers --no-deps

COPY . .
