#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  // 根据gridDim.x和blockDim.x，可知一个block负责一个seq,一个thread负责一个hidden_dimension
  
  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup
  
  // Step 1
  float l_sum = 0;
  float l_square_sum = 0;
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;  
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_sum += val.x + val.y + val.z + val.w;
    l_square_sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  } // 问题：当idx=hidden_size-3时，取出来的val时什么，val.x, val,y, val.z, val.w
  //对这个问题的理解会直接影响到对l_sum和l_square_sum求平均的时候需不需要hidden_size*4
  // problem solved, 我看了一下launch_layernorm里的hidden_dim>>=2, 明白是怎么回事了

  // Step 2
  blockReduce<ReduceType::kSum, 1>(&l_sum);
  blockReduce<ReduceType::kSum, 1>(&l_square_sum);

  // 计算均值和方差
  float mean = l_sum / (hidden_size * 4);
  float var = l_square_sum / (hidden_size * 4) - mean * mean;
  var = var + LN_EPSILON;  // 添加数值稳定性项

  // 将结果写入共享内存
  __shared__ float s_mean;
  __shared__ float s_var;
  if (threadIdx.x == 0) {
    s_mean = mean;
    s_var = var;
    if (means != nullptr) {
      means[blockIdx.x] = mean;
    }
    if (vars != nullptr) {
      vars[blockIdx.x] = var;
    }
  }
  __syncthreads();

  // Step 3
  float4 *res_f4 = reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale);
  const float4 *bias_f4 = reinterpret_cast<const float4 *>(bias);
  
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    float4 scale_val = scale_f4[idx];
    float4 bias_val = bias_f4[idx];
    
    // 计算标准化后的值
    float4 res;
    res.x = scale_val.x * (val.x - s_mean) / sqrtf(s_var) + bias_val.x;
    res.y = scale_val.y * (val.y - s_mean) / sqrtf(s_var) + bias_val.y;
    res.z = scale_val.z * (val.z - s_mean) / sqrtf(s_var) + bias_val.z;
    res.w = scale_val.w * (val.w - s_mean) / sqrtf(s_var) + bias_val.w;
    
    res_f4[idx] = res;
  }
  // assert(false && "Not Implemented");
  /// END ASSIGN3_2
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {
  // width代表隐藏层的维度，相当于是hidden_size

  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      -> https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what the g.shfl_down is doing. Usually, the threads inside a block need to load everything to shared memory and work together to reduce the result (like what you have implemented in the hw1 for reduce function). 
  //      -> Now g.shfl_down helps you do so without consuming any shared memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct position in the global output

  
  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);
  
  int col = blockIdx.x * blockDim.x + threadIdx.x; //处理第几个hidden size
  float sum_dgamma = 0.0f;
  float sum_dbetta = 0.0f;
  
  if (col < width) {
      for (int row = threadIdx.y; row < rows; row += blockDim.y) { //处理当前hidden size对应的第几行
          int idx = row * width + col;
          float dout = out_grad[idx];
          float x = inp[idx];
          float xhat;
          
          if (means != nullptr) {
              // 使用 means 和 vars 计算 xhat
              float mean = means[row];
              float var = max(vars[row], LN_EPSILON);
              float var_rsqrt = rsqrtf(var);
              xhat = (x - mean) * var_rsqrt;
          } else {
              // 使用 output 和 beta 计算 xhat
              float b = betta[col];
              float g = max(gamma[col], LN_EPSILON);
              xhat = (x - b) / g;
          }
          
          sum_dgamma += dout * xhat;
          sum_dbetta += dout;
      }
  }
  
  // 使用 shfl_down 进行规约，shfl_down是warp级别的操作，现在我们的block是32*32的，所以这里本质上是把一行的线程（一共32个，正好是一个warp）进行规约
  for (int i = g.size() / 2; i > 0; i /= 2) {
      sum_dgamma += g.shfl_down(sum_dgamma, i);
      sum_dbetta += g.shfl_down(sum_dbetta, i);
  }
  
  if (threadIdx.y == 0 && col < width) {
      gamma_grad[col] = sum_dgamma;
      betta_grad[col] = sum_dbetta;
  }

  // assert(false && "Not Implemented");
  /// END ASSIGN3_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  
  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient
  
  float dxhat_sum = 0.0f;
  float dxhat_xhat_sum = 0.0f;
  
  const float4 *out_grad_f4 = reinterpret_cast<const float4 *>(out_grad) + blockIdx.x * hidden_dim;
  const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_dim;
  const float4 *betta_f4 = reinterpret_cast<const float4 *>(betta);
  
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
      float4 dout = out_grad_f4[idx];
      float4 g = gamma_f4[idx];
      float4 x = inp_f4[idx];
      
      // 计算 dxhat = dout * gamma
      float4 dxhat;
      dxhat.x = dout.x * g.x;
      dxhat.y = dout.y * g.y;
      dxhat.z = dout.z * g.z;
      dxhat.w = dout.w * g.w;
      
      // 计算 xhat
      float4 xhat;
      if (means != nullptr) {
          // 使用 means 和 vars 计算 xhat
          float mean = means[blockIdx.x];
          float var = max(vars[blockIdx.x], LN_EPSILON);
          float var_rsqrt = rsqrtf(var);
          
          xhat.x = (x.x - mean) * var_rsqrt;
          xhat.y = (x.y - mean) * var_rsqrt;
          xhat.z = (x.z - mean) * var_rsqrt;
          xhat.w = (x.w - mean) * var_rsqrt;
      } else {
          // 使用 output 和 beta 计算 xhat
          float4 b = betta_f4[idx];
          
          xhat.x = (x.x - b.x) / max(g.x, LN_EPSILON);
          xhat.y = (x.y - b.y) / max(g.y, LN_EPSILON);
          xhat.z = (x.z - b.z) / max(g.z, LN_EPSILON);
          xhat.w = (x.w - b.w) / max(g.w, LN_EPSILON);
      }
      
      // 累加和
      dxhat_sum += dxhat.x + dxhat.y + dxhat.z + dxhat.w;
      dxhat_xhat_sum += dxhat.x * xhat.x + dxhat.y * xhat.y + 
                        dxhat.z * xhat.z + dxhat.w * xhat.w;
  }
  
  // 规约
  blockReduce<ReduceType::kSum, 1>(&dxhat_sum); // 将一个block内所有线程的dxhat_sum和dxhat_xhat_sum都加起来
  blockReduce<ReduceType::kSum, 1>(&dxhat_xhat_sum);
  
  // 计算最终梯度
  float4 *inp_grad_f4 = reinterpret_cast<float4 *>(inp_grad) + blockIdx.x * hidden_dim;
  float m = hidden_dim * 4.0f;
  float var = max(vars[blockIdx.x], LN_EPSILON);
  float var_rsqrt = rsqrtf(var);
  
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
      float4 dout = out_grad_f4[idx];
      float4 g = gamma_f4[idx];
      float4 x = inp_f4[idx];

      float4 xhat;
      if (means != nullptr) {
          float mean = means[blockIdx.x];
          xhat.x = (x.x - mean) * var_rsqrt;
          xhat.y = (x.y - mean) * var_rsqrt;
          xhat.z = (x.z - mean) * var_rsqrt;
          xhat.w = (x.w - mean) * var_rsqrt;
      } else {
          float4 b = betta_f4[idx];
          
          xhat.x = (x.x - b.x) / max(g.x, LN_EPSILON);
          xhat.y = (x.y - b.y) / max(g.y, LN_EPSILON);
          xhat.z = (x.z - b.z) / max(g.z, LN_EPSILON);
          xhat.w = (x.w - b.w) / max(g.w, LN_EPSILON);
      }
      
      float4 dx;
      dx.x = (dout.x * g.x - (dxhat_sum + xhat.x * dxhat_xhat_sum) / m) * var_rsqrt;
      dx.y = (dout.y * g.y - (dxhat_sum + xhat.y * dxhat_xhat_sum) / m) * var_rsqrt;
      dx.z = (dout.z * g.z - (dxhat_sum + xhat.z * dxhat_xhat_sum) / m) * var_rsqrt;
      dx.w = (dout.w * g.w - (dxhat_sum + xhat.w * dxhat_xhat_sum) / m) * var_rsqrt;
      
      inp_grad_f4[idx] = dx;
  }

  // assert(false && "Not Implemented");
  /// END ASSIGN3_2
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {
  
  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  // The result is then multiplied by TILE_DIM to ensure that the grid size is a multiple of TILE_DIM.
  dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}
