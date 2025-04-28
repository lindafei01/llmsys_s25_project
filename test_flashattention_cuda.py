import numpy as np
import time
import torch  
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

# 添加 transpose 函数定义
def transpose(a: minitorch.Tensor) -> minitorch.Tensor:
    return a._new(a._tensor.permute(0, 1, 3, 2))

def test_flash_attention():
    """测试不同序列长度下 Flash Attention 的性能和内存使用"""
    backend = minitorch.TensorBackend(CudaKernelOps)
    
    # 固定参数
    batch_size = 32
    num_heads = 8
    head_dim = 64
    
    # 测试不同的序列长度
    seq_lengths = [128, 256, 512, 1024, 2048]
    
    print("\nFlash Attention Performance Testing:")
    print("=" * 50)
    print(f"Fixed params: batch_size={batch_size}, num_heads={num_heads}, head_dim={head_dim}")
    
    # 存储结果用于最终汇总
    results = []
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        # 创建输入数据
        shape = (batch_size, num_heads, seq_len, head_dim)
        
        # 记录初始GPU内存
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        # 使用 tensor_from_numpy 替代 tensor
        q = minitorch.tensor_from_numpy(np.random.randn(*shape).astype(np.float32), 
                                      backend=backend, 
                                      requires_grad=True)
        k = minitorch.tensor_from_numpy(np.random.randn(*shape).astype(np.float32), 
                                      backend=backend, 
                                      requires_grad=True)
        v = minitorch.tensor_from_numpy(np.random.randn(*shape).astype(np.float32), 
                                      backend=backend, 
                                      requires_grad=True)
        grad = minitorch.tensor_from_numpy(
            np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32), 
            backend=backend
        )
        
        # 1. Flash Attention
        torch.cuda.synchronize()  # 确保GPU操作完成
        start_time = time.time()
        out_flash = q.flash_attention(k, v, causal=True)
        out_flash.backward(grad)
        torch.cuda.synchronize()  # 确保GPU操作完成
        flash_time = time.time() - start_time
        flash_memory = torch.cuda.max_memory_allocated() - initial_memory
        
        # 重置GPU内存统计
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        # 2. 基础 Attention 实现
        torch.cuda.synchronize()  # 确保GPU操作完成
        start_time = time.time()
        scores = (q @ transpose(k)) * (1.0 / np.sqrt(head_dim))
        
        if True:  # causal masking
            mask = -np.finfo(np.float32).max * np.triu(
                np.ones((batch_size, num_heads, seq_len, seq_len), dtype=np.float32), 
                k=1
            )
            mask_tensor = minitorch.tensor_from_numpy(mask, backend=backend)
            scores = scores + mask_tensor
        
        attn = minitorch.nn.softmax(scores, dim=-1)
        out_base = attn @ v
        out_base.backward(grad)
        torch.cuda.synchronize()  # 确保GPU操作完成
        base_time = time.time() - start_time
        base_memory = torch.cuda.max_memory_allocated() - initial_memory
        
        # 存储结果
        results.append({
            'seq_len': seq_len,
            'flash_time': flash_time * 1000,  # 转换为ms
            'base_time': base_time * 1000,
            'speedup': base_time / flash_time,
            'flash_memory': flash_memory / (1024 * 1024),  # 转换为MB
            'base_memory': base_memory / (1024 * 1024)
        })
        
        # 输出当前结果
        print(f"\nSequence Length: {seq_len}")
        print("\nPerformance Metrics:")
        print(f"Flash Attention time: {flash_time*1000:.2f} ms")
        print(f"Base Attention time: {base_time*1000:.2f} ms")
        print(f"Speedup: {base_time/flash_time:.2f}x")
        print(f"Flash Attention memory: {flash_memory/1024/1024:.2f} MB")
        print(f"Base Attention memory: {base_memory/1024/1024:.2f} MB")
        print(f"Memory saved: {(base_memory - flash_memory)/1024/1024:.2f} MB")
    
    # 打印汇总结果
    print("\n" + "=" * 70)
    print("Summary of Results:")
    print("=" * 70)
    print("Seq Length | Flash (ms) | Base (ms) | Speedup | Flash Mem (MB) | Base Mem (MB)")
    print("-" * 70)
    for result in results:
        print(f"{result['seq_len']:>10} | {result['flash_time']:>9.2f} | "
              f"{result['base_time']:>8.2f} | {result['speedup']:>7.2f}x | "
              f"{result['flash_memory']:>12.2f} | {result['base_memory']:>11.2f}")

if __name__ == "__main__":
    test_flash_attention()
