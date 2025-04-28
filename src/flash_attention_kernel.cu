#include "includes/flash_attention.cuh"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <curand_kernel.h>
#include <stdio.h>

// 块大小常量定义
#define BLOCK_M 64    // Query sequence block size
#define BLOCK_N 64    // Key/Value sequence block size
#define BLOCK_D 32    // Head dimension block size
#define WARP_SIZE 32

// 动态计算块大小的结构体
struct BlockConfig {
    int Bc;  // Key/Value sequence block size
    int Br;  // Query sequence block size
    int Tr;  // Number of query blocks
    int Tc;  // Number of key/value blocks
    size_t shared_mem_size;
};

__host__ __device__ BlockConfig calculate_block_config(int M, int d, int N) {
    BlockConfig config;
    config.Bc = M / (4 * d);
    config.Br = min(M / (4 * d), d);
    config.Tr = (N + config.Br - 1) / config.Br;
    config.Tc = (N + config.Bc - 1) / config.Bc;
    config.shared_mem_size = (config.Br * d + 2 * config.Bc * d) * sizeof(float);
    return config;
} // __device__表示这个函数在GPU设备上执行

// Forward kernel, __global__声明kernel函数，可以从CPU端调用但在GPU上执行，是CPU和GPU代码之间的桥梁
__global__ void flash_attention_forward_kernel(
    const float* __restrict__ q,        // [B, H, N, D]
    const float* __restrict__ k,        // [B, H, N, D]
    const float* __restrict__ v,        // [B, H, N, D]
    float* __restrict__ out,           // [B, H, N, D]
    float* __restrict__ l,             // [B, H, N]
    float* __restrict__ m,             // [B, H, N]
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float sm_scale,
    const float dropout_prob,
    const bool causal,
    const unsigned long long seed
) {
    // 计算块配置
    BlockConfig config = calculate_block_config(48 * 1024, head_dim, seq_len); // 假设48KB SRAM
    
    // 声明共享内存
    extern __shared__ float shared_mem[];
    float* q_shared = shared_mem;
    float* k_shared = q_shared + config.Br * head_dim;
    float* v_shared = k_shared + config.Bc * head_dim;
    float* mi = v_shared + config.Bc * head_dim;
    float* li = mi + config.Br;
    float* oi = li + config.Br;  // oi 长度为 config.Br * head_dim
    
    // 初始化随机数生成器（用于dropout）
    curandState rng_state;
    curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &rng_state);
    
    // 计算块索引
    const int batch_head_idx = blockIdx.x;  // 合并batch和head维度
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    const int block_idx = blockIdx.y;
    const int query_block = block_idx / config.Tc;
    const int key_block = block_idx % config.Tc;
    
    // 计算偏移量
    const int batch_stride = num_heads * seq_len * head_dim;
    const int head_stride = seq_len * head_dim;
    const int base_idx = batch_idx * batch_stride + head_idx * head_stride;
    
    // 初始化局部累加器
    #pragma unroll
    for (int i = 0; i < config.Br; i++) {
        mi[i] = -INFINITY;
        li[i] = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            oi[i * head_dim + d] = 0.0f;
        }
    }
    
    // 主循环：处理所有key/value块
    for (int kv_block = 0; kv_block < config.Tc; kv_block++) {
        // 加载K,V到共享内存
        const int kv_start = kv_block * config.Bc;
        for (int i = threadIdx.x; i < config.Bc && kv_start + i < seq_len; i += blockDim.x) {
            // threadIdx.x：线程在x维度的索引
            // blockDim.x: block在x维度的大小（线程数）
            // i += blockDim.x: 线程跨步，每个线程处理间隔为blockDim.x的数据
            for (int d = threadIdx.y; d < head_dim; d += blockDim.y) {
                const int kv_idx = base_idx + (kv_start + i) * head_dim + d;
                k_shared[i * head_dim + d] = k[kv_idx];
                v_shared[i * head_dim + d] = v[kv_idx];
            }
        }
        
        // 加载Q到共享内存
        const int q_start = query_block * config.Br;
        for (int i = threadIdx.x; i < config.Br && q_start + i < seq_len; i += blockDim.x) {
            for (int d = threadIdx.y; d < head_dim; d += blockDim.y) {
                const int q_idx = base_idx + (q_start + i) * head_dim + d;
                q_shared[i * head_dim + d] = q[q_idx];
            }
        }
        __syncthreads();//确保block内所有线程在继续执行之前，都完成了这个点之前的所有操作
        
        // 计算当前块的attention scores
        for (int i = threadIdx.x; i < config.Br && q_start + i < seq_len; i += blockDim.x) {
            for (int j = 0; j < config.Bc && kv_start + j < seq_len; j++) {
                // 跳过因果mask的部分
                if (causal && (q_start + i) > (kv_start + j)) {
                    continue;
                }
                
                // 计算Q·K^T
                float qk = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    qk += q_shared[i * head_dim + d] * k_shared[j * head_dim + d];
                }
                qk *= sm_scale;
                
                // 计算局部最大值
                float mij = qk;
                float mi_prev = mi[i];
                mi[i] = max(mi[i], mij);
                
                // 计算缩放因子
                float scale = __expf(mi_prev - mi[i]);
                float pij = __expf(qk - mi[i]);
                
                // 应用dropout
                if (dropout_prob > 0.0f) {
                    float rand = curand_uniform(&rng_state);
                    if (rand < dropout_prob) {
                        pij = 0.0f;
                    } else {
                        pij /= (1.0f - dropout_prob);
                    }
                }
                
                // 更新累加器
                li[i] = li[i] * scale + pij;
                for (int d = 0; d < head_dim; d++) {
                    oi[i * head_dim + d] = oi[i * head_dim + d] * scale + pij * v_shared[j * head_dim + d];
                }
            }
        }
        __syncthreads();
    }
    
    // 写回结果
    const int q_start = query_block * config.Br;
    for (int i = threadIdx.x; i < config.Br && q_start + i < seq_len; i += blockDim.x) {
        const int out_idx = base_idx + (q_start + i) * head_dim;
        
        // 保存中间结果
        l[batch_idx * num_heads * seq_len + head_idx * seq_len + q_start + i] = li[i];
        m[batch_idx * num_heads * seq_len + head_idx * seq_len + q_start + i] = mi[i];
        
        // 写出最终结果
        for (int d = 0; d < head_dim; d++) {
            out[out_idx + d] = oi[i * head_dim + d] / li[i];
        }
    }
}

__global__ void flash_attention_backward_kernel(
    const float* __restrict__ q,        // [B, H, N, D]
    const float* __restrict__ k,        // [B, H, N, D]
    const float* __restrict__ v,        // [B, H, N, D]
    const float* __restrict__ o,        // [B, H, N, D]
    const float* __restrict__ do_,      // [B, H, N, D]
    const float* __restrict__ l,        // [B, H, N]
    const float* __restrict__ m,        // [B, H, N]
    float* __restrict__ dq,            // [B, H, N, D]
    float* __restrict__ dk,            // [B, H, N, D]
    float* __restrict__ dv,            // [B, H, N, D]
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float sm_scale,              // τ
    const float dropout_prob,          // p_drop
    const bool causal,
    const unsigned long long seed      // R
) {
    // 步骤2：计算块大小
    BlockConfig config = calculate_block_config(48 * 1024, head_dim, seq_len);
    
    // 声明共享内存
    extern __shared__ float shared_mem[];
    // K_j, V_j 的共享内存
    float* k_shared = shared_mem;
    float* v_shared = k_shared + config.Bc * head_dim;
    // Q_i, O_i, dO_i, dQ_i 的共享内存
    float* q_shared = v_shared + config.Bc * head_dim;
    float* o_shared = q_shared + config.Br * head_dim;
    float* do_shared = o_shared + config.Br * head_dim;
    float* dq_shared = do_shared + config.Br * head_dim;
    // dK_j, dV_j 的共享内存（步骤8：Initialize）
    float* dk_shared = dq_shared + config.Br * head_dim;
    float* dv_shared = dk_shared + config.Bc * head_dim;
    
    // 计算块索引
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int kv_block_idx = blockIdx.z % ((seq_len + config.Bc - 1) / config.Bc);
    const int q_block_idx = blockIdx.z / ((seq_len + config.Bc - 1) / config.Bc);
    
    // 步骤8：初始化dK_j, dV_j
    for (int i = threadIdx.x; i < config.Bc * head_dim; i += blockDim.x) {
        dk_shared[i] = 0.0f;
        dv_shared[i] = 0.0f;
    }
    __syncthreads();
    
    // 步骤7：加载K_j, V_j到shared memory
    const int kv_start = kv_block_idx * config.Bc;
    for (int i = threadIdx.x; i < config.Bc && kv_start + i < seq_len; i += blockDim.x) {
        for (int d = threadIdx.y; d < head_dim; d += blockDim.y) {
            const int idx = batch_idx * (num_heads * seq_len * head_dim) +
                          head_idx * (seq_len * head_dim) +
                          (kv_start + i) * head_dim + d;
            k_shared[i * head_dim + d] = k[idx];
            v_shared[i * head_dim + d] = v[idx];
        }
    }
    
    // 步骤10：加载Q_i, O_i, dO_i, dQ_i, l_i, m_i
    const int q_start = q_block_idx * config.Br;
    float l_i = 0.0f, m_i = 0.0f;
    for (int i = threadIdx.x; i < config.Br && q_start + i < seq_len; i += blockDim.x) {
        const int base_idx = batch_idx * (num_heads * seq_len * head_dim) +
                           head_idx * (seq_len * head_dim) +
                           (q_start + i) * head_dim;
        const int stats_idx = batch_idx * (num_heads * seq_len) +
                            head_idx * seq_len +
                            (q_start + i);
        
        l_i = l[stats_idx];
        m_i = m[stats_idx];
        
        for (int d = threadIdx.y; d < head_dim; d += blockDim.y) {
            q_shared[i * head_dim + d] = q[base_idx + d];
            o_shared[i * head_dim + d] = o[base_idx + d];
            do_shared[i * head_dim + d] = do_[base_idx + d];
        }
    }
    __syncthreads();
    
    // 对每个query位置进行处理
    for (int i = threadIdx.x; i < config.Br && q_start + i < seq_len; i += blockDim.x) {
        // 步骤11：计算S_ij = τQ_iK_j^T
        for (int j = 0; j < config.Bc && kv_start + j < seq_len; j++) {
            if (causal && (q_start + i) < (kv_start + j)) continue;
            
            // 计算Q_iK_j^T
            float s_ij = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                s_ij += q_shared[i * head_dim + d] * k_shared[j * head_dim + d];
            }
            s_ij *= sm_scale; // τ
            
            // 步骤13：计算P_ij
            float p_ij = __expf(s_ij - m_i) / l_i;
            
            // 步骤14-15：计算dropout mask和P_ij^dropped
            float z_ij;
            if (dropout_prob > 0.0f) {
                // 使用相同的seed来重现forward pass的dropout pattern
                curandState state;
                curand_init(seed + (q_start + i) * seq_len + (kv_start + j),
                          0, 0, &state);
                z_ij = curand_uniform(&state) > dropout_prob ? 
                      1.0f / (1.0f - dropout_prob) : 0.0f;
            } else {
                z_ij = 1.0f;
            }
            float p_dropped = p_ij * z_ij;
            
            // 步骤16：计算dV_j
            for (int d = 0; d < head_dim; d++) {
                atomicAdd(&dv_shared[j * head_dim + d],
                         p_dropped * do_shared[i * head_dim + d]);
            }
            
            // 步骤17-18：计算dP_ij^dropped
            float dp_dropped = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dp_dropped += do_shared[i * head_dim + d] * 
                            v_shared[j * head_dim + d];
            }
            float dp = dp_dropped * z_ij;
            
            // 步骤19：计算D_i
            float d_i = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                d_i += do_shared[i * head_dim + d] * 
                       o_shared[i * head_dim + d];
            }
            
            // 步骤20：计算dS_ij
            float ds_ij = p_ij * (dp - d_i);
            
            // 步骤21-22：计算并累积dQ_i和dK_j
            for (int d = 0; d < head_dim; d++) {
                // dQ_i
                atomicAdd(&dq[batch_idx * (num_heads * seq_len * head_dim) +
                             head_idx * (seq_len * head_dim) +
                             (q_start + i) * head_dim + d],
                         sm_scale * ds_ij * k_shared[j * head_dim + d]);
                
                // dK_j
                atomicAdd(&dk_shared[j * head_dim + d],
                         sm_scale * ds_ij * q_shared[i * head_dim + d]);
            }
        }
    }
    __syncthreads();
    
    // 步骤24：写回dK_j, dV_j到HBM
    for (int i = threadIdx.x; i < config.Bc && kv_start + i < seq_len; i += blockDim.x) {
        for (int d = threadIdx.y; d < head_dim; d += blockDim.y) {
            const int idx = batch_idx * (num_heads * seq_len * head_dim) +
                          head_idx * (seq_len * head_dim) +
                          (kv_start + i) * head_dim + d;
            atomicAdd(&dk[idx], dk_shared[i * head_dim + d]);
            atomicAdd(&dv[idx], dv_shared[i * head_dim + d]);
        }
    }
}

// C++ 封装函数
void flash_attention_forward_cuda(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    float* l,
    float* m,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float sm_scale,
    float dropout_prob,
    bool causal,
    cudaStream_t stream
) {
    // 计算块配置
    BlockConfig config = calculate_block_config(48 * 1024, head_dim, seq_len);
    
    // 设置grid和block维度
    dim3 grid(batch_size * num_heads, config.Tr * config.Tc);
    dim3 block(32, 32);  // 可以根据需要调整
    
    // 生成随机种子
    unsigned long long seed = rand();
    
    // 计算shared memory大小
    size_t shared_mem_size = (
        config.Br * head_dim +     // Q块
        config.Bc * head_dim +     // K块
        config.Bc * head_dim +     // V块
        config.Br +                // mi
        config.Br +                // li
        config.Br * head_dim       // oi
    ) * sizeof(float);
    
    // 启动kernel
    flash_attention_forward_kernel<<<grid, block, shared_mem_size, stream>>>(
        q, k, v, out, l, m,
        batch_size, num_heads, seq_len, head_dim,
        sm_scale, dropout_prob, causal, seed
    );
}

void flash_attention_backward_cuda(
    const float* q,          // [B, H, N, D]
    const float* k,          // [B, H, N, D]
    const float* v,          // [B, H, N, D]
    const float* o,          // [B, H, N, D]
    const float* do_,        // [B, H, N, D]
    const float* l,          // [B, H, N]
    const float* m,          // [B, H, N]
    float* dq,              // [B, H, N, D]
    float* dk,              // [B, H, N, D]
    float* dv,              // [B, H, N, D]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float sm_scale,
    float dropout_prob,
    bool causal,
    unsigned long long seed, // 使用forward pass的seed来重现dropout pattern
    cudaStream_t stream
) {
    // 步骤2：计算块大小
    // 假设on-chip SRAM大小为48KB
    BlockConfig config = calculate_block_config(48 * 1024, head_dim, seq_len);
    
    // 计算总的block数
    // T_r = ⌈N/B_r⌉ blocks for queries
    // T_c = ⌈N/B_c⌉ blocks for keys/values
    const int num_q_blocks = (seq_len + config.Br - 1) / config.Br;
    const int num_kv_blocks = (seq_len + config.Bc - 1) / config.Bc;
    
    // 设置grid和block维度
    dim3 grid(
        batch_size,           // 每个batch单独处理
        num_heads,           // 每个head单独处理
        num_q_blocks * num_kv_blocks  // 所有Q和K/V块的组合
    );
    
    // 设置block维度
    // 使用2D block来并行化矩阵运算
    dim3 block(32, 32);  // 可以根据GPU架构调整
    
    // 计算shared memory大小
    // 需要存储：
    // 1. K_j, V_j 块
    // 2. Q_i, O_i, dO_i 块
    // 3. dK_j, dV_j 累积器
    size_t shared_mem_size = (
        // K_j, V_j
        2 * config.Bc * head_dim + 
        // Q_i, O_i, dO_i
        3 * config.Br * head_dim +
        // dK_j, dV_j
        2 * config.Bc * head_dim
    ) * sizeof(float);
    
    // 检查shared memory大小是否超过设备限制
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (shared_mem_size > prop.sharedMemPerBlock) {
        fprintf(stderr, "Required shared memory (%zu bytes) exceeds device limit (%zu bytes)\n",
                shared_mem_size, prop.sharedMemPerBlock);
        return;
    }
    
    // 设置shared memory大小
    cudaFuncSetAttribute(
        flash_attention_backward_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size
    );
    
    // 启动kernel
    flash_attention_backward_kernel<<<grid, block, shared_mem_size, stream>>>(
        q, k, v,          // 输入矩阵
        o,                // forward pass的输出
        do_,              // 输出的梯度
        l, m,             // softmax统计信息
        dq, dk, dv,       // 需要计算的梯度
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        sm_scale,         // softmax缩放因子 τ
        dropout_prob,     // dropout概率
        causal,          // 是否使用因果mask
        seed             // 用于重现dropout pattern的随机数种子
    );
    
    // 检查kernel启动是否成功
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(error));
        return;
    }
}

extern "C" {

void flashAttentionForward(
    float* q,
    float* k,
    float* v,
    float* out,
    float* l,
    float* m,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float sm_scale,
    float dropout_prob,
    bool causal,
    unsigned long long seed
) {
    // 设置grid和block维度
    dim3 grid(batch_size * num_heads, (seq_len + BLOCK_M - 1) / BLOCK_M);
    dim3 block(32, 32);
    
    // 计算shared memory大小
    size_t shared_mem_size = (
        BLOCK_M * head_dim +     // Q块
        BLOCK_N * head_dim +     // K块
        BLOCK_N * head_dim +     // V块
        BLOCK_M +                // mi
        BLOCK_M +                // li
        BLOCK_M * head_dim       // oi
    ) * sizeof(float);
    
    // 启动kernel
    flash_attention_forward_kernel<<<grid, block, shared_mem_size>>>(
        q, k, v, out, l, m,
        batch_size, num_heads, seq_len, head_dim,
        sm_scale, dropout_prob, causal, seed
    );
}

void flashAttentionBackward(
    float* q,
    float* k,
    float* v,
    float* out,
    float* dout,
    float* dq,
    float* dk,
    float* dv,
    float* l,
    float* m,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float sm_scale,
    float dropout_prob,
    bool causal,
    unsigned long long seed
) {
    // 设置grid和block维度
    dim3 grid(batch_size * num_heads, (seq_len + BLOCK_M - 1) / BLOCK_M);
    dim3 block(32, 32);
    
    // 计算shared memory大小
    size_t shared_mem_size = (
        BLOCK_M * head_dim +     // Q块
        BLOCK_N * head_dim +     // K块
        BLOCK_N * head_dim +     // V块
        BLOCK_M * head_dim +     // dO块
        BLOCK_M * head_dim +     // dQ块
        BLOCK_N * head_dim +     // dK块
        BLOCK_N * head_dim       // dV块
    ) * sizeof(float);
    
    // 启动kernel
    flash_attention_backward_kernel<<<grid, block, shared_mem_size>>>(
        q, k, v, out, dout,
        dq, dk, dv, l, m,
        batch_size, num_heads, seq_len, head_dim,
        sm_scale, dropout_prob, causal, seed
    );
}

} // extern "C"