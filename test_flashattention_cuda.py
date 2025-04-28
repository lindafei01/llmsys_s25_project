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
    try:
        out = q.flash_attention(k, v, causal=True)
        out.backward(grad)
        return True
    except Exception as e:
        print(f"Flash Attention Error: {str(e)}")
        return False

def run_minitorch_attention(q, k, v, grad, head_dim, seq_len, batch_size, num_heads):
    """运行 Minitorch 基础实现"""
    try:
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
        return True
    except Exception as e:
        print(f"Minitorch Base Error: {str(e)}")
        return False

def run_pytorch_attention(q, k, v, grad, head_dim, seq_len):
    """运行 PyTorch 基础实现"""
    try:
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
        
        # 创建因果注意力掩码
        mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda'), diagonal=1)
        mask = mask.expand(q.size(0), q.size(1), seq_len, seq_len)
        scores = scores.masked_fill(mask.bool(), float('-inf'))
        
        # 应用 softmax
        attn = F.softmax(scores, dim=-1)
        
        # 计算输出
        out = torch.matmul(attn, v)
        out.backward(grad)
        return True
    except Exception as e:
        print(f"PyTorch Base Error: {str(e)}")
        return False

def test_attention_implementations():
    backend = minitorch.TensorBackend(CudaKernelOps)
    
    batch_size = 32
    num_heads = 8
    head_dim = 64
    
    # seq_lengths = [64, 128, 256, 512, 1024, 2048]
    seq_lengths = [256]
    
    print("\nAttention Implementations Comparison:")
    print("=" * 70)
    print(f"Parameters: batch_size={batch_size}, num_heads={num_heads}, head_dim={head_dim}")
    
    results = []
    
    try:
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
                'PyTorch Base': lambda: run_pytorch_attention(q_torch, k_torch, v_torch, grad_torch, head_dim, seq_len)
            }
            
            # 存储每个实现的结果
            perf_results = {}
            
            for name, impl in implementations.items():
                # 重置 GPU 内存统计
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()  # 清空 GPU 缓存
                initial_memory = torch.cuda.memory_allocated()
                
                # 清除梯度
                if name == 'PyTorch Base':
                    q_torch.grad = k_torch.grad = v_torch.grad = None
                else:
                    q.grad = k.grad = v.grad = None
                
                # 运行实现并计时
                torch.cuda.synchronize()
                start_time = time.time()
                success = impl()
                torch.cuda.synchronize()
                exec_time = time.time() - start_time
                memory_used = torch.cuda.max_memory_allocated() - initial_memory
                
                if success:
                    perf_results[name] = {
                        'time': exec_time * 1000,  # 转换为ms
                        'memory': memory_used / (1024 * 1024)  # 转换为MB
                    }
                else:
                    perf_results[name] = {
                        'time': float('inf'),
                        'memory': float('inf')
                    }
                    print(f"{name} failed for sequence length {seq_len}")
            
            # 如果所有实现都失败了，跳过这个序列长度
            if all(v['time'] == float('inf') for v in perf_results.values()):
                print(f"All implementations failed for sequence length {seq_len}, skipping...")
                continue
            
            # 计算相对性能
            flash_time = perf_results['Flash Attention']['time']
            flash_memory = perf_results['Flash Attention']['memory']
            
            results.append({
                'seq_len': seq_len,
                **{f"{k}_time": v['time'] for k, v in perf_results.items()},
                **{f"{k}_memory": v['memory'] for k, v in perf_results.items()},
                'speedup_vs_minitorch': perf_results['Minitorch Base']['time'] / flash_time if flash_time != float('inf') else 0,
                'speedup_vs_pytorch': perf_results['PyTorch Base']['time'] / flash_time if flash_time != float('inf') else 0,
                'memory_saved_vs_minitorch': perf_results['Minitorch Base']['memory'] - flash_memory,
                'memory_saved_vs_pytorch': perf_results['PyTorch Base']['memory'] - flash_memory
            })
            
            # 打印当前结果
            print("\nExecution Time:")
            for name, res in perf_results.items():
                if res['time'] == float('inf'):
                    print(f"{name:15}: Failed")
                else:
                    print(f"{name:15}: {res['time']:.2f} ms")
            
            print("\nMemory Usage:")
            for name, res in perf_results.items():
                if res['memory'] == float('inf'):
                    print(f"{name:15}: Failed")
                else:
                    print(f"{name:15}: {res['memory']:.2f} MB")
            
            if flash_time != float('inf'):
                print("\nSpeedup Ratios:")
                print(f"vs Minitorch Base: {perf_results['Minitorch Base']['time'] / flash_time:.2f}x")
                print(f"vs PyTorch Base: {perf_results['PyTorch Base']['time'] / flash_time:.2f}x")
                
                print("\nMemory Savings:")
                print(f"vs Minitorch Base: {(perf_results['Minitorch Base']['memory'] - flash_memory):.2f} MB")
                print(f"vs PyTorch Base: {(perf_results['PyTorch Base']['memory'] - flash_memory):.2f} MB")
            
            # 确保清理 GPU 内存
            torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Error during testing: {str(e)}")
    
    finally:
        # 清理 GPU 内存
        torch.cuda.empty_cache()
        
        if results:
            # 打印汇总结果
            print("\n" + "=" * 120)
            print("Summary of Results:")
            print("=" * 120)
            headers = ["Seq Len", "Flash (ms)", "MT Base (ms)", "PT Base (ms)", 
                      "vs MT", "vs PT", "Flash Mem", "MT Mem", "PT Mem"]
            print(" | ".join(f"{h:>12}" for h in headers))
            print("-" * 120)
            
            for r in results:
                print(f"{r['seq_len']:>12} | "
                      f"{r['Flash Attention_time']:>12.2f} | "
                      f"{r['Minitorch Base_time']:>12.2f} | "
                      f"{r['PyTorch Base_time']:>12.2f} | "
                      f"{r['speedup_vs_minitorch']:>12.2f}x | "
                      f"{r['speedup_vs_pytorch']:>12.2f}x | "
                      f"{r['Flash Attention_memory']:>12.2f} | "
                      f"{r['Minitorch Base_memory']:>12.2f} | "
                      f"{r['PyTorch Base_memory']:>12.2f}")

if __name__ == "__main__":
    test_attention_implementations()
