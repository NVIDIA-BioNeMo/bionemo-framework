#!/bin/bash -x
cd esm2
PIP_CONSTRAINT= pip install -r requirements.txt
./install_vllm.sh
