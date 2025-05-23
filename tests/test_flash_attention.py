import torch
from torch.nn import functional as F
import sys
import os
import minitorch
import numpy as np
from minitorch.tensor_functions import FlashAttention
import psutil
import gc
import torch
from torch.nn import functional as F
import time
import tracemalloc
from tqdm import tqdm
import argparse
from minitorch.cuda_kernel_ops import CudaKernelOps

cuda_backend = minitorch.TensorBackend(CudaKernelOps)
import ipdb



def test_flash_attention_correctness():
    """
    Test FlashAttention implementation against PyTorch's attention
    """

    batch_size = 2
    n_heads = 8
    seq_len = 40
    head_dim = 64
    
    np_q = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)
    np_k = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)
    np_v = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)
    

    mini_q = minitorch.tensor_from_numpy(np_q, backend=cuda_backend)
    mini_k = minitorch.tensor_from_numpy(np_k, backend=cuda_backend)
    mini_v = minitorch.tensor_from_numpy(np_v, backend=cuda_backend)
    
    torch_q = torch.tensor(np_q, requires_grad=True)
    torch_k = torch.tensor(np_k, requires_grad=True)
    torch_v = torch.tensor(np_v, requires_grad=True)
    
    scale = 1.0 / (head_dim ** 0.5)
    
    def torch_attention(q, k, v, causal=True):
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(mask, -float('inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)
    
    # Test forward pass
    torch_output = torch_attention(torch_q, torch_k, torch_v, causal=True)
    mini_output = FlashAttention.apply(mini_q, mini_k, mini_v, True, scale)
    
    np_mini_output = mini_output.to_numpy()
    np_torch_output = torch_output.detach().numpy()

    assert np.allclose(np_mini_output, np_torch_output, rtol=1e-4, atol=1e-4), \
        "FlashAttention forward pass doesn't match PyTorch attention"
    
    # Test backward pass
    torch_grad = torch.randn_like(torch_output)
    np_grad = torch_grad.numpy()
    mini_grad = minitorch.tensor_from_numpy(np_grad, backend=cuda_backend)
    
    # Backward pass with PyTorch
    torch_output.backward(torch_grad)
    
    # Backward pass with MiniTorch FlashAttention
    mini_output.backward(mini_grad)
    
    assert np.allclose(mini_grad.to_numpy(), torch_grad.numpy(), rtol=1e-4, atol=1e-5), \
        " Gradient doesn't match"
    
    print("FlashAttention correctness test passed!")

def test_flash_attention_memory():
    """
    Test peak memory usage (footprint) of FlashAttention compared to standard attention
    """
    batch_size = 32
    n_heads = 8
    seq_lengths = [512, 1024, 2048, 4096] 
    head_dim = 64
    
    print("\nMemory footprint comparison (in MB):")
    print("-" * 70)
    print(f"{'Seq Length':^15} | {'Standard Attention':^25} | {'Flash Attention':^25}")
    print("-" * 70)
    
    for seq_len in seq_lengths:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        np_q = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)
        np_k = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)
        np_v = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)
        
        mini_q = minitorch.tensor_from_numpy(np_q, backend=cuda_backend)
        mini_k = minitorch.tensor_from_numpy(np_k, backend=cuda_backend)
        mini_v = minitorch.tensor_from_numpy(np_v, backend=cuda_backend)

        torch_q = torch.tensor(np_q)
        torch_k = torch.tensor(np_k)
        torch_v = torch.tensor(np_v)
        
        scale = 1.0 / (head_dim ** 0.5)
        
        def standard_attention_torch():
            scores = torch.matmul(torch_q, torch_k.transpose(-2, -1)) * scale
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(mask, -float('inf'))
            attn_weights = F.softmax(scores, dim=-1)
            return torch.matmul(attn_weights, torch_v)
        
        def flash_attention():
            return FlashAttention.apply(mini_q, mini_k, mini_v, True, scale)

        try:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, 
                           torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else torch.profiler.ProfilerActivity.CPU],
                profile_memory=True,
                record_shapes=True
            ) as prof_standard:
                _ = standard_attention_torch()
            
            standard_memory = 0
            for event in prof_standard.key_averages():
                if event.self_cpu_memory_usage > 0:
                    standard_memory += event.self_cpu_memory_usage
                if hasattr(event, 'self_cuda_memory_usage') and event.self_cuda_memory_usage > 0:
                    standard_memory += event.self_cuda_memory_usage
            
            standard_memory = standard_memory / (1024 * 1024)
            
        except Exception as e:
            print(f"Standard attention failed for seq_len={seq_len}: {e}")
            standard_memory = float('inf')
        
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                           torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else torch.profiler.ProfilerActivity.CPU],
                profile_memory=True,
                record_shapes=True
            ) as prof_flash:
                _ = flash_attention()
            
            flash_memory = 0
            for event in prof_flash.key_averages():
                if event.self_cpu_memory_usage > 0:
                    flash_memory += event.self_cpu_memory_usage
                if hasattr(event, 'self_cuda_memory_usage') and event.self_cuda_memory_usage > 0:
                    flash_memory += event.self_cuda_memory_usage
            
            flash_memory = flash_memory / (1024 * 1024)
            
        except Exception as e:
            print(f"Flash attention failed for seq_len={seq_len}: {e}")
            flash_memory = float('inf')
        
        print(f"{seq_len:^15} | {standard_memory:^25.2f} | {flash_memory:^25.2f}")
        
def test_flash_attention_speed():
    """
    Test speed of FlashAttention compared to standard attention
    """
    
    batch_size = 32
    n_heads = 8
    seq_lengths = [512, 1024, 2048, 4096]  
    head_dim = 64
    iterations = 10  
    
    print("\nSpeed comparison (in milliseconds):")
    print("-" * 70)
    print(f"{'Seq Length':^15} | {'Standard Attention':^25} | {'Flash Attention':^25}")
    print("-" * 70)
    
    for seq_len in seq_lengths:
        np_q = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)
        np_k = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)
        np_v = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)
        
        scale = 1.0 / (head_dim ** 0.5)
        
        def run_standard_attention():
            q = minitorch.tensor_from_numpy(np_q, backend=cuda_backend)
            k = minitorch.tensor_from_numpy(np_k, backend=cuda_backend)
            v = minitorch.tensor_from_numpy(np_v, backend=cuda_backend)

            scores = q @ k.permute(0, 1, 3, 2) * scale
            
            mask = -np.finfo(np.float32).max * np.triu(np.ones((batch_size, n_heads, seq_len, seq_len), dtype=np.float32), 1)
            mask_tensor = minitorch.tensor_from_numpy(mask, backend=cuda_backend)
            
            scores = scores + mask_tensor
            
            attn_weights = minitorch.softmax(scores, dim=-1)
            return attn_weights @ v
        
        # Warmup
        _ = run_standard_attention()
        
        # Time standard attention
        standard_times = []
        for _ in tqdm(range(iterations), total=iterations, colour='green'):
            start_time = time.time()
            _ = run_standard_attention()
            standard_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        standard_avg_time = sum(standard_times) / len(standard_times)
        
        def run_flash_attention():
            q = minitorch.tensor_from_numpy(np_q, backend=cuda_backend)
            k = minitorch.tensor_from_numpy(np_k, backend=cuda_backend)
            v = minitorch.tensor_from_numpy(np_v, backend=cuda_backend)
            
            return FlashAttention.apply(q, k, v, True, scale)
        
        # Warmup
        _ = run_flash_attention()
        
        # Time FlashAttention
        flash_times = []
        for _ in range(iterations):
            start_time = time.time()
            _ = run_flash_attention()
            flash_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        flash_avg_time = sum(flash_times) / len(flash_times)
        
        # Print results
        print(f"{seq_len:^15} | {standard_avg_time:^25.2f} | {flash_avg_time:^25.2f}")
    
if __name__ == "__main__":

    print("=== Testing FlashAttention Correctness ===")
    test_flash_attention_correctness()

    print("\n=== Testing FlashAttention Memory Usage ===")
    test_flash_attention_memory()

    print("\n=== Testing FlashAttention Speed ===")
    test_flash_attention_speed()
    
    print("\nAll requested tests completed successfully!")
