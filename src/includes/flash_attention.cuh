#pragma once
#include <cuda_runtime.h>
#include <cuda.h>

// Flash Attention Forward声明
void flash_attention_forward_cuda(
    const float* q,     // [B, H, N, D], batch size, head number, sequence length, head dimension
    const float* k,     // [B, H, N, D]
    const float* v,     // [B, H, N, D]
    float* out,         // [B, H, N, D]
    float* l,           // [B, H, N]
    float* m,           // [B, H, N]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float sm_scale,
    bool causal,
    cudaStream_t stream
);

// Flash Attention Backward声明
void flash_attention_backward_cuda(
    const float* q,     // [B, H, N, D]
    const float* k,     // [B, H, N, D]
    const float* v,     // [B, H, N, D]
    const float* out,   // [B, H, N, D]
    const float* dout,  // [B, H, N, D]
    const float* l,     // [B, H, N]
    const float* m,     // [B, H, N]
    float* dq,          // [B, H, N, D]
    float* dk,          // [B, H, N, D]
    float* dv,          // [B, H, N, D]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float sm_scale,
    bool causal,
    cudaStream_t stream
);