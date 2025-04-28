import torch
import time
import numpy as np
from torch.utils.cpp_extension import load

# 编译CUDA扩展
flash_attention_cuda = load(
    name="flash_attention_cuda",
    sources=["src/flash_attention_kernel.cu"],
    extra_include_paths=["src/includes"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    build_directory="build"
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
    out_flash, l, m = flash_attention_cuda.forward(
        q, k, v,
        sm_scale=1.0/np.sqrt(head_dim),
        dropout_prob=0.0,
        causal=True,
        stream=None
    )
    
    # 运行标准Attention
    qk = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    qk.masked_fill_(mask, float('-inf'))
    attn = torch.softmax(qk, dim=-1)
    out_standard = torch.matmul(attn, v)
    
    max_diff = (out_flash - out_standard).abs().max().item()
    print(f"Max difference: {max_diff}")
    assert max_diff < 1e-5, "Results don't match!"

def test_backward():
    # 准备输入
    batch_size = 32
    num_heads = 8
    seq_len = 1024
    head_dim = 64
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    
    # 运行Flash Attention forward
    out_flash, l, m = flash_attention_cuda.forward(
        q, k, v,
        sm_scale=1.0/np.sqrt(head_dim),
        dropout_prob=0.0,
        causal=True,
        stream=None
    )
    
    # 运行标准Attention forward
    qk = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    qk.masked_fill_(mask, float('-inf'))
    attn = torch.softmax(qk, dim=-1)
    out_standard = torch.matmul(attn, v)
    
    # 计算梯度
    grad = torch.randn_like(out_flash)
    out_flash.backward(grad)
    out_standard.backward(grad)
    
    # 比较梯度
    max_diff_q = (q.grad - q.grad).abs().max().item()
    max_diff_k = (k.grad - k.grad).abs().max().item()
    max_diff_v = (v.grad - v.grad).abs().max().item()
    
    print(f"Max gradient difference (q): {max_diff_q}")
    print(f"Max gradient difference (k): {max_diff_k}")
    print(f"Max gradient difference (v): {max_diff_v}")
    
    assert max_diff_q < 1e-5, "Q gradients don't match!"
    assert max_diff_k < 1e-5, "K gradients don't match!"
    assert max_diff_v < 1e-5, "V gradients don't match!"

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
        flash_attention_cuda.forward(
            q, k, v,
            sm_scale=1.0/np.sqrt(head_dim),
            dropout_prob=0.0,
            causal=True,
            stream=None
        )
    
    # 测速
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        flash_attention_cuda.forward(
            q, k, v,
            sm_scale=1.0/np.sqrt(head_dim),
            dropout_prob=0.0,
            causal=True,
            stream=None
        )
    torch.cuda.synchronize()
    flash_time = (time.time() - start) / 100
    
    # 标准Attention测速
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        qk = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
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
    # 使用pytorch profiler测量内存访问
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
            flash_attention_cuda.forward(
                q, k, v,
                sm_scale=1.0/np.sqrt(head_dim),
                dropout_prob=0.0,
                causal=True,
                stream=None
            )
        with record_function("standard_attention"):
            qk = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            qk.masked_fill_(mask, float('-inf'))
            attn = torch.softmax(qk, dim=-1)
            out = torch.matmul(attn, v)
    
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

def test_error_handling():
    # 测试无效的输入维度
    try:
        q = torch.randn(32, 8, 1024, 65, device='cuda')  # head_dim不是64
        k = torch.randn(32, 8, 1024, 64, device='cuda')
        v = torch.randn(32, 8, 1024, 64, device='cuda')
        flash_attention_cuda.forward(
            q, k, v,
            sm_scale=1.0/np.sqrt(64),
            dropout_prob=0.0,
            causal=True,
            stream=None
        )
        assert False, "Should have raised an error for invalid head_dim"
    except RuntimeError:
        pass
    
    # 测试共享内存不足的情况
    try:
        q = torch.randn(32, 8, 16384, 64, device='cuda')  # 过大的序列长度
        k = torch.randn(32, 8, 16384, 64, device='cuda')
        v = torch.randn(32, 8, 16384, 64, device='cuda')
        flash_attention_cuda.forward(
            q, k, v,
            sm_scale=1.0/np.sqrt(64),
            dropout_prob=0.0,
            causal=True,
            stream=None
        )
        assert False, "Should have raised an error for insufficient shared memory"
    except RuntimeError:
        pass

if __name__ == "__main__":
    test_correctness()
    test_backward()
    benchmark_speed()
    measure_memory()
    test_error_handling()