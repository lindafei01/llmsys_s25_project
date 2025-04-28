import numpy as np
import time
import torch  
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

# 添加 transpose 函数定义
def transpose(a: minitorch.Tensor) -> minitorch.Tensor:
    return a._new(a._tensor.permute(0, 1, 3, 2))

def test_flash_attention():
    """测试不同序列长度下 Flash Attention 的正确性和性能"""
    backend = minitorch.TensorBackend(CudaKernelOps)
    
    # 固定参数
    batch_size = 32
    num_heads = 8
    head_dim = 64
    
    # 测试不同的序列长度
    seq_lengths = [128, 256, 512, 1024, 2048]
    
    print("\nFlash Attention Testing:")
    print("=" * 50)
    print(f"Fixed params: batch_size={batch_size}, num_heads={num_heads}, head_dim={head_dim}")
    
    # 存储结果用于最终汇总
    results = []
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        # 创建输入数据
        shape = (batch_size, num_heads, seq_len, head_dim)
        
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
        
        # Forward
        start_time = time.time()
        out_flash = q.flash_attention(k, v, causal=True)
        grad = minitorch.tensor(np.random.randn(*shape), backend=backend)
        out_flash.backward(grad)
        flash_time = time.time() - start_time
        
        # 2. 基础 Attention 实现
        start_time = time.time()
        # 生成 causal mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e8
        # 扩展 mask 维度以匹配 scores: (1, 1, seq_len, seq_len)
        mask = mask.reshape(1, 1, seq_len, seq_len)
        mask_tensor = minitorch.tensor(mask, backend=backend)
        
        scores = (q @ transpose(k)) * (1.0 / np.sqrt(head_dim))
        scores = scores + mask_tensor
        attn = minitorch.nn.softmax(scores, dim=-1)
        out_base = attn @ v
        out_base.backward(grad)
        base_time = time.time() - start_time
        
        # 3. 结果比较
        # 转换为 torch tensor 便于比较
        out_flash = torch.tensor(out_flash._tensor._storage).float().cuda()
        dq_flash = torch.tensor(q.grad._tensor._storage).float().cuda()
        dk_flash = torch.tensor(k.grad._tensor._storage).float().cuda()
        dv_flash = torch.tensor(v.grad._tensor._storage).float().cuda()
        
        out_base = torch.tensor(out_base._tensor._storage).float().cuda()
        dq_base = torch.tensor(q.grad._tensor._storage).float().cuda()
        dk_base = torch.tensor(k.grad._tensor._storage).float().cuda()
        dv_base = torch.tensor(v.grad._tensor._storage).float().cuda()
        
        # 计算相对误差
        def rel_error(x, y):
            return torch.max(torch.abs(x - y) / (torch.max(torch.abs(y)) + 1e-8))
        
        # 计算误差
        output_error = rel_error(out_flash, out_base)
        dq_error = rel_error(dq_flash, dq_base)
        dk_error = rel_error(dk_flash, dk_base)
        dv_error = rel_error(dv_flash, dv_base)
        
        # 存储结果
        results.append({
            'seq_len': seq_len,
            'flash_time': flash_time * 1000,  # 转换为ms
            'base_time': base_time * 1000,
            'speedup': base_time / flash_time,
            'output_error': output_error.item(),
            'dq_error': dq_error.item(),
            'dk_error': dk_error.item(),
            'dv_error': dv_error.item()
        })
        
        # 输出当前结果
        print(f"\nSequence Length: {seq_len}")
        print("Correctness Check:")
        print(f"Output relative error: {output_error:.6f}")
        print(f"dQ relative error: {dq_error:.6f}")
        print(f"dK relative error: {dk_error:.6f}")
        print(f"dV relative error: {dv_error:.6f}")
        
        print("\nSpeed Comparison:")
        print(f"Flash Attention time: {flash_time*1000:.2f} ms")
        print(f"Base Attention time: {base_time*1000:.2f} ms")
        print(f"Speedup: {base_time/flash_time:.2f}x")
        
        # 检查数值是否在可接受范围内
        error_threshold = 1e-3
        assert output_error < error_threshold, f"Output error too large at seq_len={seq_len}"
        assert dq_error < error_threshold, f"dQ error too large at seq_len={seq_len}"
        assert dk_error < error_threshold, f"dK error too large at seq_len={seq_len}"
        assert dv_error < error_threshold, f"dV error too large at seq_len={seq_len}"
    
    # 打印汇总结果
    print("\n" + "=" * 50)
    print("Summary of Results:")
    print("=" * 50)
    print("Sequence Length | Flash (ms) | Base (ms) | Speedup | Max Error")
    print("-" * 70)
    for result in results:
        max_error = max(result['output_error'], result['dq_error'], 
                        result['dk_error'], result['dv_error'])
        print(f"{result['seq_len']:>14} | {result['flash_time']:>9.2f} | "
                f"{result['base_time']:>8.2f} | {result['speedup']:>7.2f}x | {max_error:.6f}")
        

if __name__ == "__main__":
    test_flash_attention()
