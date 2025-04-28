import numpy as np
import time
import torch  
import torch.nn.functional as F
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

# 添加 transpose 函数定义
def transpose(a: minitorch.Tensor) -> minitorch.Tensor:
    return a._new(a._tensor.permute(0, 1, 3, 2))

def run_flash_attention(q, k, v, grad, head_dim):
    """运行 Flash Attention 实现"""
    out = q.flash_attention(k, v, causal=True)
    out.backward(grad)
    return out

def run_minitorch_attention(q, k, v, grad, head_dim, seq_len, batch_size, num_heads):
    """运行 Minitorch 基础实现"""
    scores = (q @ transpose(k)) * (1.0 / np.sqrt(head_dim))
    
    # causal masking
    mask = -np.finfo(np.float32).max * np.triu(
        np.ones((batch_size, num_heads, seq_len, seq_len), dtype=np.float32), 
        k=1
    )
    mask_tensor = minitorch.tensor_from_numpy(mask, backend=q.backend)
    scores = scores + mask_tensor
    
    attn = minitorch.nn.softmax(scores, dim=-1)
    out = attn @ v
    out.backward(grad)
    return out

def run_pytorch_attention(q, k, v, grad, head_dim, seq_len):
    """运行 PyTorch 原生实现"""
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda'), diagonal=1).bool()
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=causal_mask,
        is_causal=True,
        scale=1.0 / np.sqrt(head_dim)
    )
    out.backward(grad)
    return out

def test_attention_implementations():
    """对比测试不同的 Attention 实现"""
    backend = minitorch.TensorBackend(CudaKernelOps)
    
    # 固定参数
    batch_size = 32
    num_heads = 8
    head_dim = 64
    
    # 测试不同的序列长度
    seq_lengths = [128, 256, 512, 1024, 2048]
    
    print("\nAttention Implementations Comparison:")
    print("=" * 70)
    print(f"Parameters: batch_size={batch_size}, num_heads={num_heads}, head_dim={head_dim}")
    
    results = []
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        shape = (batch_size, num_heads, seq_len, head_dim)
        
        # 创建 PyTorch 输入
        q_torch = torch.randn(*shape, device='cuda', requires_grad=True)
        k_torch = torch.randn(*shape, device='cuda', requires_grad=True)
        v_torch = torch.randn(*shape, device='cuda', requires_grad=True)
        grad_torch = torch.randn(*shape, device='cuda')
        
        # 创建 Minitorch 输入
        q = minitorch.tensor_from_numpy(q_torch.detach().cpu().numpy().astype(np.float32), 
                                      backend=backend, requires_grad=True)
        k = minitorch.tensor_from_numpy(k_torch.detach().cpu().numpy().astype(np.float32), 
                                      backend=backend, requires_grad=True)
        v = minitorch.tensor_from_numpy(v_torch.detach().cpu().numpy().astype(np.float32), 
                                      backend=backend, requires_grad=True)
        grad = minitorch.tensor_from_numpy(grad_torch.cpu().numpy().astype(np.float32), 
                                         backend=backend)
        
        implementations = {
            'Flash Attention': lambda: run_flash_attention(q, k, v, grad, head_dim),
            'Minitorch Base': lambda: run_minitorch_attention(q, k, v, grad, head_dim, seq_len, batch_size, num_heads),
            'PyTorch Native': lambda: run_pytorch_attention(q_torch, k_torch, v_torch, grad_torch, head_dim, seq_len)
        }
        
        # 存储每个实现的结果
        perf_results = {}
        
        for name, impl in implementations.items():
            # 重置 GPU 内存统计
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
            
            # 清除梯度
            if name == 'PyTorch Native':
                q_torch.grad = k_torch.grad = v_torch.grad = None
            else:
                q.grad = k.grad = v.grad = None
            
            # 运行实现并计时
            torch.cuda.synchronize()
            start_time = time.time()
            impl()
            torch.cuda.synchronize()
            exec_time = time.time() - start_time
            memory_used = torch.cuda.max_memory_allocated() - initial_memory
            
            perf_results[name] = {
                'time': exec_time * 1000,  # 转换为ms
                'memory': memory_used / (1024 * 1024)  # 转换为MB
            }
        
        # 计算相对性能
        flash_time = perf_results['Flash Attention']['time']
        flash_memory = perf_results['Flash Attention']['memory']
        
        results.append({
            'seq_len': seq_len,
            **{f"{k}_time": v['time'] for k, v in perf_results.items()},
            **{f"{k}_memory": v['memory'] for k, v in perf_results.items()},
            'speedup_vs_minitorch': perf_results['Minitorch Base']['time'] / flash_time,
            'speedup_vs_pytorch': perf_results['PyTorch Native']['time'] / flash_time,
            'memory_saved_vs_minitorch': perf_results['Minitorch Base']['memory'] - flash_memory,
            'memory_saved_vs_pytorch': perf_results['PyTorch Native']['memory'] - flash_memory
        })
        
        # 打印当前结果
        print("\nExecution Time:")
        for name, res in perf_results.items():
            print(f"{name:15}: {res['time']:.2f} ms")
        
        print("\nMemory Usage:")
        for name, res in perf_results.items():
            print(f"{name:15}: {res['memory']:.2f} MB")
        
        print("\nSpeedup Ratios:")
        print(f"vs Minitorch Base: {perf_results['Minitorch Base']['time'] / flash_time:.2f}x")
        print(f"vs PyTorch Native: {perf_results['PyTorch Native']['time'] / flash_time:.2f}x")
        
        print("\nMemory Savings:")
        print(f"vs Minitorch Base: {(perf_results['Minitorch Base']['memory'] - flash_memory):.2f} MB")
        print(f"vs PyTorch Native: {(perf_results['PyTorch Native']['memory'] - flash_memory):.2f} MB")
    
    # 打印汇总结果
    print("\n" + "=" * 120)
    print("Summary of Results:")
    print("=" * 120)
    headers = ["Seq Len", "Flash (ms)", "MT Base (ms)", "PT Native (ms)", 
              "vs MT", "vs PT", "Flash Mem", "MT Mem", "PT Mem"]
    print(" | ".join(f"{h:>12}" for h in headers))
    print("-" * 120)
    
    for r in results:
        print(f"{r['seq_len']:>12} | "
              f"{r['Flash Attention_time']:>12.2f} | "
              f"{r['Minitorch Base_time']:>12.2f} | "
              f"{r['PyTorch Native_time']:>12.2f} | "
              f"{r['speedup_vs_minitorch']:>12.2f}x | "
              f"{r['speedup_vs_pytorch']:>12.2f}x | "
              f"{r['Flash Attention_memory']:>12.2f} | "
              f"{r['Minitorch Base_memory']:>12.2f} | "
              f"{r['PyTorch Native_memory']:>12.2f}")

if __name__ == "__main__":
    test_attention_implementations()
