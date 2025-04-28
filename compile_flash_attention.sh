#!/bin/bash

# 创建输出目录
mkdir -p build

# 编译Flash Attention CUDA扩展
nvcc -o build/flash_attention.so --shared \
    src/flash_attention_kernel.cu \
    -I src/includes \
    -Xcompiler -fPIC \
    -O3 \
    --use_fast_math 