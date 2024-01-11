#!/bin/bash

rdzv_port=$((10000 + $RANDOM % 9000))

torchrun \
    --nproc_per_node=4 \
    --rdzv_endpoint "localhost:${rdzv_port}" \
    --rdzv_backend c10d \
    -m pytest \
    --cov-report=term \
    --cov-report=html \
    --cov=megatron \
    tests_aisg
