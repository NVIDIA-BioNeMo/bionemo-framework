#!/bin/bash
# nvidia-resiliency-ext>=0.6.0 is required at runtime by megatron-core==0.17.1,
# but it depends on grpcio-tools>=1.76.0 which requires protobuf>=6.30.0,
# conflicting with nemo-toolkit==2.4.0 (protobuf~=5.29.5).
# Install it without deps to avoid the protobuf conflict, then install
# its safe transitive deps separately.
pip install --no-deps "nvidia-resiliency-ext>=0.6.0"
pip install "defusedxml" "httpx>=0.24.0" "nvidia-ml-py>=12.570.86"

# Install the package itself
PIP_CONSTRAINT= pip install -e .
