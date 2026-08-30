#!/bin/bash

NCCL_P2P_DISABLE=1 torchrun --nproc-per-node 1 --nnodes 1 train.py --cp 1
NCCL_P2P_DISABLE=1 torchrun --nproc-per-node 2 --nnodes 1 train.py --cp 2
