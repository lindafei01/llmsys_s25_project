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
import argparse

def test_flash_attention_correctness():
    """
    Test FlashAttention implementation against PyTorch's attention
    """

    # Test parameters
    batch_size = 32
    n_heads = 8
    seq_len = 40
    head_dim = 64
    
    # Create inputs (use same values for both implementations)
    np_q = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)
    np_k = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)
    np_v = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)
    
    # MiniTorch inputs
    mini_q = minitorch.tensor_from_numpy(np_q)
    mini_k = minitorch.tensor_from_numpy(np_k)
    mini_v = minitorch.tensor_from_numpy(np_v)
    
    # PyTorch inputs
    torch_q = torch.tensor(np_q, requires_grad=True)
    torch_k = torch.tensor(np_k, requires_grad=True)
    torch_v = torch.tensor(np_v, requires_grad=True)
    
    # Scale factor
    scale = 1.0 / (head_dim ** 0.5)
    
    # Forward pass with PyTorch (standard attention)
    def torch_attention(q, k, v, causal=True):
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(mask, -float('inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)
    
    # Run both implementations
    torch_output = torch_attention(torch_q, torch_k, torch_v, causal=True)
    mini_output = FlashAttention.apply(mini_q, mini_k, mini_v, True, scale)
    
    # Compare forward pass results
    np_mini_output = mini_output.to_numpy()
    np_torch_output = torch_output.detach().numpy()
    assert np.allclose(np_mini_output, np_torch_output, rtol=1e-4, atol=1e-5), \
        "FlashAttention forward pass doesn't match PyTorch attention"
    
    # Test backward pass
    torch_grad = torch.randn_like(torch_output)
    np_grad = torch_grad.numpy()
    mini_grad = minitorch.tensor_from_numpy(np_grad)
    
    # Backward pass with PyTorch
    torch_output.backward(torch_grad)
    
    # Backward pass with MiniTorch FlashAttention
    mini_output.backward(mini_grad)
    
    # Compare gradients
    assert np.allclose(mini_q.grad.to_numpy(), torch_q.grad.numpy(), rtol=1e-4, atol=1e-5), \
        "Query gradient doesn't match"
    assert np.allclose(mini_k.grad.to_numpy(), torch_k.grad.numpy(), rtol=1e-4, atol=1e-5), \
        "Key gradient doesn't match"
    assert np.allclose(mini_v.grad.to_numpy(), torch_v.grad.numpy(), rtol=1e-4, atol=1e-5), \
        "Value gradient doesn't match"
    
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
        
        scale = 1.0 / (head_dim ** 0.5)
        
        def measure_standard_attention_memory():
            tracemalloc.start()
            
            q = minitorch.tensor_from_numpy(np_q)
            k = minitorch.tensor_from_numpy(np_k)
            v = minitorch.tensor_from_numpy(np_v)
            
            current, peak = tracemalloc.get_traced_memory()
            
            scores = q @ k.permute(0, 1, 3, 2) * scale
            
            # Apply causal mask
            batch_size, n_heads, seq_len, _ = scores.shape
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
            mask_tensor = minitorch.tensor_from_numpy(mask.numpy())
            scores = scores.masked_fill(mask_tensor, -float('inf'))
            
            current2, peak2 = tracemalloc.get_traced_memory()
            peak = max(peak, peak2)
            
            attn_weights = minitorch.softmax(scores, dim=-1)
            
            current3, peak3 = tracemalloc.get_traced_memory()
            peak = max(peak, peak3)
            
            result = attn_weights @ v
            
            current4, peak4 = tracemalloc.get_traced_memory()
            peak = max(peak, peak4)
            
            tracemalloc.stop()
            return result, peak / (1024 * 1024)  # Convert to MB
        
        def measure_flash_attention_memory():
            tracemalloc.start()
            
            q = minitorch.tensor_from_numpy(np_q)
            k = minitorch.tensor_from_numpy(np_k)
            v = minitorch.tensor_from_numpy(np_v)
            
            current, peak = tracemalloc.get_traced_memory()
            
            result = FlashAttention.apply(q, k, v, True, scale)
            
            current2, peak2 = tracemalloc.get_traced_memory()
            peak = max(peak, peak2)
            
            tracemalloc.stop()
            return result, peak / (1024 * 1024)
        
        # Measure memory footprint
        try:
            _, standard_memory = measure_standard_attention_memory()
        except Exception as e:
            print(f"Standard attention failed for seq_len={seq_len}: {e}")
            standard_memory = float('inf')
        
        gc.collect()  # Clear memory between tests
        
        try:
            _, flash_memory = measure_flash_attention_memory()
        except Exception as e:
            print(f"Flash attention failed for seq_len={seq_len}: {e}")
            flash_memory = float('inf')
        
        # Print results
        print(f"{seq_len:^15} | {standard_memory:^25.2f} | {flash_memory:^25.2f}")
    
    print("-" * 70)
    if standard_memory != float('inf') and flash_memory != float('inf'):
        print(f"Memory reduction: {(1 - flash_memory/standard_memory) * 100:.2f}% for seq_len={seq_lengths[-1]}")
    else:
        print("Could not calculate memory reduction due to OOM error")

def test_flash_attention_speed():
    """
    Test speed of FlashAttention compared to standard attention
    """
    
    # Test parameters
    batch_size = 32
    n_heads = 8
    seq_lengths = [512, 1024, 2048, 4096]  # Test with different sequence lengths
    head_dim = 64
    iterations = 10  # Number of iterations for averaging
    
    print("\nSpeed comparison (in milliseconds):")
    print("-" * 70)
    print(f"{'Seq Length':^15} | {'Standard Attention':^25} | {'Flash Attention':^25}")
    print("-" * 70)
    
    for seq_len in seq_lengths:
        # Create inputs
        np_q = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)
        np_k = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)
        np_v = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)
        
        # Scale factor
        scale = 1.0 / (head_dim ** 0.5)
        
        # Standard attention
        def run_standard_attention():
            q = minitorch.tensor_from_numpy(np_q)
            k = minitorch.tensor_from_numpy(np_k)
            v = minitorch.tensor_from_numpy(np_v)
            
            # Standard attention computation
            scores = q @ k.permute(0, 1, 3, 2) * scale
            
            # Apply causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
            mask_tensor = minitorch.tensor_from_numpy(mask.numpy())
            scores = scores.masked_fill(mask_tensor, -float('inf'))
            
            attn_weights = minitorch.softmax(scores, dim=-1)
            return attn_weights @ v
        
        # Warmup
        _ = run_standard_attention()
        
        # Time standard attention
        standard_times = []
        for _ in range(iterations):
            start_time = time.time()
            _ = run_standard_attention()
            standard_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        standard_avg_time = sum(standard_times) / len(standard_times)
        
        # FlashAttention
        def run_flash_attention():
            q = minitorch.tensor_from_numpy(np_q)
            k = minitorch.tensor_from_numpy(np_k)
            v = minitorch.tensor_from_numpy(np_v)
            
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
    
    print("-" * 70)
    print(f"Speedup: {standard_avg_time/flash_avg_time:.2f}x for seq_len={seq_lengths[-1]}")

if __name__ == "__main__":

    print("=== Testing FlashAttention Correctness ===")
    test_flash_attention_correctness()

    print("\n=== Testing FlashAttention Memory Usage ===")
    test_flash_attention_memory()

    print("\n=== Testing FlashAttention Speed ===")
    test_flash_attention_speed()
    
    print("\nAll requested tests completed successfully!")
