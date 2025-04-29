import numpy as np
import time
import torch  
import torch.nn.functional as F
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
import pycuda.driver as cuda
import atexit

# 初始化 CUDA
cuda.init()

# 获取 CUDA 设备和创建上下文
cuda_device = cuda.Device(0)
cuda_context = cuda_device.make_context()

# 确保程序退出时清理上下文
def cleanup_cuda():
    try:
        if cuda.Context.get_current() is not None:
            cuda.Context.pop()
    except Exception as e:
        print(f"Error during CUDA cleanup: {str(e)}")

atexit.register(cleanup_cuda)

def transpose(a: minitorch.Tensor) -> minitorch.Tensor:
    return a._new(a._tensor.permute(0, 1, 3, 2))

def get_gpu_memory_usage():
    """获取当前 GPU 内存使用情况（包括 PyTorch 和 PyCUDA）"""
    try:
        cuda.Context.push(cuda_context)  # 确保在正确的上下文中
        free_mem, total_mem = cuda.mem_get_info()
        cuda.Context.pop()  # 操作完成后弹出上下文
        return total_mem - free_mem
    except Exception as e:
        print(f"Error getting GPU memory: {str(e)}")
        return 0

def run_flash_attention(q, k, v, head_dim):
    """运行 Flash Attention 实现"""
    try:
        cuda.Context.push(cuda_context)  # 确保在正确的上下文中
        out = q.flash_attention(k, v, causal=True)
        cuda.Context.pop()  # 操作完成后弹出上下文
        return True, out
    except Exception as e:
        print(f"Flash Attention Error: {str(e)}")
        if cuda.Context.get_current() is not None:
            cuda.Context.pop()
        return False, None

def run_minitorch_attention(q, k, v, head_dim, seq_len, batch_size, num_heads):
    """运行 Minitorch 基础实现"""
    try:
        cuda.Context.push(cuda_context)  # 确保在正确的上下文中
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
        
        cuda.Context.pop()  # 操作完成后弹出上下文
        return True, out
    except Exception as e:
        print(f"Minitorch Base Error: {str(e)}")
        if cuda.Context.get_current() is not None:
            cuda.Context.pop()
        return False, None

def run_pytorch_attention(q, k, v, head_dim, seq_len):
    """运行 PyTorch 基础实现"""
    try:
        # PyTorch 使用自己的 CUDA 上下文管理，不需要显式管理
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda'), diagonal=1)
        mask = mask.expand(q.size(0), q.size(1), seq_len, seq_len)
        scores = scores.masked_fill(mask.bool(), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        return True, out
    except Exception as e:
        print(f"PyTorch Base Error: {str(e)}")
        return False, None

def check_attention_correctness(seq_len=128):
    """
    单独检查不同实现的正确性，比较输出结果
    
    Args:
        seq_len (int): 测试的序列长度
        
    Returns:
        bool: 是否通过正确性检查
    """
    print("\n" + "=" * 70)
    print(f"检查注意力实现的正确性 (seq_len={seq_len})")
    print("=" * 70)
    
    try:
        backend = minitorch.TensorBackend(CudaKernelOps)
        
        # 设置测试参数
        batch_size = 4  # 使用较小的batch size减轻内存压力
        num_heads = 2
        head_dim = 64
        shape = (batch_size, num_heads, seq_len, head_dim)
        
        print("创建输入数据...")
        # 创建输入数据 (使用相同的随机种子确保一致性)
        torch.manual_seed(42)
        q_torch = torch.randn(*shape, device='cuda')
        k_torch = torch.randn(*shape, device='cuda')
        v_torch = torch.randn(*shape, device='cuda')
        
        # 为minitorch创建相同的输入
        q = minitorch.tensor_from_numpy(q_torch.detach().cpu().numpy().astype(np.float32), backend=backend)
        k = minitorch.tensor_from_numpy(k_torch.detach().cpu().numpy().astype(np.float32), backend=backend)
        v = minitorch.tensor_from_numpy(v_torch.detach().cpu().numpy().astype(np.float32), backend=backend)
        
        # 运行不同实现
        print("运行 PyTorch Base 实现...")
        pt_success, pt_out = run_pytorch_attention(q_torch, k_torch, v_torch, head_dim, seq_len)
        if not pt_success:
            print("PyTorch Base 实现失败!")
            return False
            
        print("运行 Minitorch Base 实现...")
        mt_success, mt_out = run_minitorch_attention(q, k, v, head_dim, seq_len, batch_size, num_heads)
        if not mt_success:
            print("Minitorch Base 实现失败!")
            return False
            
        print("运行 Flash Attention 实现...")
        flash_success, flash_out = run_flash_attention(q, k, v, head_dim)
        if not flash_success:
            print("Flash Attention 实现失败!")
            return False
        
        # 将结果转换为NumPy进行比较
        print("比较输出结果...")
        pt_result = pt_out.detach().cpu().numpy()
        mt_result = mt_out._tensor.cpu().numpy()
        flash_result = flash_out._tensor.cpu().numpy()
        
        # 计算相对误差
        def relative_error(a, b):
            return np.max(np.abs(a - b) / (np.maximum(1e-8, np.abs(a) + np.abs(b)) / 2))
        
        flash_vs_pt = relative_error(flash_result, pt_result)
        flash_vs_mt = relative_error(flash_result, mt_result)
        mt_vs_pt = relative_error(mt_result, pt_result)
        
        print(f"Flash Attention vs PyTorch: 相对误差 = {flash_vs_pt:.6f}")
        print(f"Flash Attention vs Minitorch: 相对误差 = {flash_vs_mt:.6f}")
        print(f"Minitorch vs PyTorch: 相对误差 = {mt_vs_pt:.6f}")
        
        # 设置一个合理的容差阈值
        tolerance = 1e-4
        passed = flash_vs_pt < tolerance and flash_vs_mt < tolerance
        
        if passed:
            print("\n正确性检查通过!")
        else:
            print("\n正确性检查失败! 输出结果差异过大.")
        
        return passed
        
    except Exception as e:
        print(f"正确性检查过程中出错: {str(e)}")
        return False
    finally:
        # 清理内存
        torch.cuda.empty_cache()

def test_attention_implementations():
    try:
        backend = minitorch.TensorBackend(CudaKernelOps)
        
        batch_size = 32
        num_heads = 8
        head_dim = 64
        num_trials = 1  # 只测试一次，减少错误累积
        
        # 测试不同的序列长度
        seq_lengths = [256]  # 从小到大测试不同序列长度
        
        print("\nAttention Implementations Comparison:")
        print("=" * 70)
        print(f"Parameters: batch_size={batch_size}, num_heads={num_heads}, head_dim={head_dim}")
        print(f"Number of trials per sequence length: {num_trials}")
        
        results = []
        
        for seq_len in seq_lengths:
            print(f"\nTesting sequence length: {seq_len}")
            shape = (batch_size, num_heads, seq_len, head_dim)
            
            # 存储多次试验的结果
            trial_results = {
                'Flash Attention': {'times': [], 'memories': []},
                'Minitorch Base': {'times': [], 'memories': []},
                'PyTorch Base': {'times': [], 'memories': []}
            }
            
            for trial in range(num_trials):
                print(f"Running trial {trial + 1}/{num_trials}...")
                
                # 创建 PyTorch 输入
                q_torch = torch.randn(*shape, device='cuda')
                k_torch = torch.randn(*shape, device='cuda')
                v_torch = torch.randn(*shape, device='cuda')
                
                # 创建 Minitorch 输入
                q = minitorch.tensor_from_numpy(q_torch.detach().cpu().numpy().astype(np.float32), 
                                            backend=backend)
                k = minitorch.tensor_from_numpy(k_torch.detach().cpu().numpy().astype(np.float32), 
                                            backend=backend)
                v = minitorch.tensor_from_numpy(v_torch.detach().cpu().numpy().astype(np.float32), 
                                            backend=backend)
                
                implementations = {
                    'Flash Attention': lambda: run_flash_attention(q, k, v, head_dim),
                    'Minitorch Base': lambda: run_minitorch_attention(q, k, v, head_dim, seq_len, batch_size, num_heads),
                    'PyTorch Base': lambda: run_pytorch_attention(q_torch, k_torch, v_torch, head_dim, seq_len)
                }
                
                for name, impl in implementations.items():
                    # 获取初始内存使用
                    initial_memory = get_gpu_memory_usage()
                    
                    print(f"Running {name}...")
                    
                    # 运行实现并计时
                    start_time = time.time()
                    success, _ = impl()
                    exec_time = time.time() - start_time
                    peak_memory = get_gpu_memory_usage()
                    memory_used = peak_memory - initial_memory
                    
                    if success:
                        trial_results[name]['times'].append(exec_time * 1000)  # 转换为ms
                        trial_results[name]['memories'].append(memory_used / (1024 * 1024))  # 转换为MB
                        print(f"  Success: {exec_time * 1000:.2f} ms, {memory_used / (1024 * 1024):.2f} MB")
                    else:
                        print(f"  Failed!")
                
                # 清理内存
                torch.cuda.empty_cache()
            
            # 计算平均性能和标准差
            perf_results = {}
            for name, data in trial_results.items():
                if data['times']:  # 如果有成功的试验
                    perf_results[name] = {
                        'time': np.mean(data['times']),
                        'time_std': np.std(data['times']),
                        'memory': np.mean(data['memories']),
                        'memory_std': np.std(data['memories'])
                    }
                else:
                    perf_results[name] = {
                        'time': float('inf'),
                        'time_std': 0,
                        'memory': float('inf'),
                        'memory_std': 0
                    }
            
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
                **{f"{k}_time_std": v['time_std'] for k, v in perf_results.items()},
                **{f"{k}_memory": v['memory'] for k, v in perf_results.items()},
                **{f"{k}_memory_std": v['memory_std'] for k, v in perf_results.items()},
                'speedup_vs_minitorch': perf_results['Minitorch Base']['time'] / flash_time if flash_time != float('inf') else 0,
                'speedup_vs_pytorch': perf_results['PyTorch Base']['time'] / flash_time if flash_time != float('inf') else 0,
                'memory_saved_vs_minitorch': perf_results['Minitorch Base']['memory'] - flash_memory,
                'memory_saved_vs_pytorch': perf_results['PyTorch Base']['memory'] - flash_memory
            })
            
            # 打印当前结果
            print("\nAverage Execution Time (± std):")
            for name, res in perf_results.items():
                if res['time'] == float('inf'):
                    print(f"{name:15}: Failed")
                else:
                    print(f"{name:15}: {res['time']:.2f} ± {res['time_std']:.2f} ms")
            
            print("\nAverage Memory Usage (± std):")
            for name, res in perf_results.items():
                if res['memory'] == float('inf'):
                    print(f"{name:15}: Failed")
                else:
                    print(f"{name:15}: {res['memory']:.2f} ± {res['memory_std']:.2f} MB")
            
            if flash_time != float('inf'):
                print("\nSpeedup Ratios:")
                print(f"vs Minitorch Base: {perf_results['Minitorch Base']['time'] / flash_time:.2f}x")
                print(f"vs PyTorch Base: {perf_results['PyTorch Base']['time'] / flash_time:.2f}x")
                
                print("\nMemory Savings:")
                print(f"vs Minitorch Base: {(perf_results['Minitorch Base']['memory'] - flash_memory):.2f} MB")
                print(f"vs PyTorch Base: {(perf_results['PyTorch Base']['memory'] - flash_memory):.2f} MB")
    
    except Exception as e:
        print(f"Error during testing: {str(e)}")
    
    finally:
        # 清理 GPU 内存
        torch.cuda.empty_cache()
        
        if results:
            # 打印汇总结果
            print("\n" + "=" * 120)
            print("Summary of Results (averaged over trials):")
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
    # 首先检查正确性
    correct = check_attention_correctness(seq_len=64)  # 使用较小的序列长度检查正确性
    if correct:
        # 如果正确性检查通过，再进行性能测试
        test_attention_implementations()
    else:
        print("\n正确性检查未通过，跳过性能测试。")
