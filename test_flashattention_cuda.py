import torch
import time
import numpy as np
from torch.utils.cpp_extension import load

# 编译CUDA扩展
flash_attention_cuda = load(
    name="flash_attention_cuda",
    sources=["src/flash_attention_kernel.cu"],
    extra_include_paths=["src/includes"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

def test_correctness():
    # 准备输入
    batch_size = 32
    num_heads = 8
    seq_len = 1024
    head_dim = 64
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    
    # 运行Flash Attention
    out_flash, l, m = flash_attention_cuda.forward(q, k, v, True)
    
    # 运行标准Attention
    qk = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
    if True:  # causal
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        qk.masked_fill_(mask, float('-inf'))
    attn = torch.softmax(qk, dim=-1)
    out_standard = torch.matmul(attn, v)
    
    max_diff = (out_flash - out_standard).abs().max().item()
    print(f"Max difference: {max_diff}")
    assert max_diff < 1e-5, "Results don't match!"

def benchmark_speed():
    # 准备输入
    batch_size = 32
    num_heads = 8
    seq_len = 1024
    head_dim = 64
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    
    # 预热
    for _ in range(10):
        flash_attention_cuda.forward(q, k, v, True)
    
    # 测速
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        flash_attention_cuda.forward(q, k, v, True)
    torch.cuda.synchronize()
    flash_time = (time.time() - start) / 100
    
    # 标准Attention测速
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        qk = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
        if True:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            qk.masked_fill_(mask, float('-inf'))
        attn = torch.softmax(qk, dim=-1)
        out = torch.matmul(attn, v)
    torch.cuda.synchronize()
    standard_time = (time.time() - start) / 100
    
    print(f"Flash Attention time: {flash_time*1000:.2f}ms")
    print(f"Standard Attention time: {standard_time*1000:.2f}ms")
    print(f"Speedup: {standard_time/flash_time:.2f}x")

def measure_memory():
    # 使用nvprof或pytorch profiler测量内存访问
    from torch.profiler import profile, record_function, ProfilerActivity
    
    batch_size = 32
    num_heads = 8
    seq_len = 1024
    head_dim = 64
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    
    with profile(activities=[ProfilerActivity.CUDA],
                profile_memory=True, record_shapes=True) as prof:
        with record_function("flash_attention"):
            flash_attention_cuda.forward(q, k, v, True)
        with record_function("standard_attention"):
            qk = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            qk.masked_fill_(mask, float('-inf'))
            attn = torch.softmax(qk, dim=-1)
            out = torch.matmul(attn, v)
    
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

if __name__ == "__main__":
    test_correctness()
    benchmark_speed()
    measure_memory()