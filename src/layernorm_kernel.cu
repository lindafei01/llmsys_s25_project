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

template <typename T>
__forceinline__ __device__ T add_eps(T x) {
  return fabsf(x) > LN_EPSILON ? x : (x < 0 ? -LN_EPSILON : LN_EPSILON);
}

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
  }

  // Step 2
  float mean_dim = float(hidden_size) * 4.f;
  float reduce_val[2] = {l_sum, l_square_sum};
  blockReduce<ReduceType::kSum, 2>(reduce_val);

   __shared__ float s_mean, s_var;
  if (threadIdx.x == 0) {
    s_mean = reduce_val[0] / mean_dim;
    if (means != nullptr) {
      means[blockIdx.x] = s_mean;
    }
    s_var = reduce_val[1] / mean_dim - s_mean * s_mean + LN_EPSILON;
    vars[blockIdx.x] = s_var;
    s_var = rsqrtf(s_var);
  }
  __syncthreads();

  float4 *output_f4 =
      reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 vscale = __ldg(reinterpret_cast<const float4 *>(scale) + idx);
    float4 vbias = __ldg(reinterpret_cast<const float4 *>(bias) + idx);
    float4 val = inp_f4[idx];
    val.x = (val.x - s_mean) * s_var * vscale.x + vbias.x;
    val.y = (val.y - s_mean) * s_var * vscale.y + vbias.y;
    val.z = (val.z - s_mean) * s_var * vscale.z + vbias.z;
    val.w = (val.w - s_mean) * s_var * vscale.w + vbias.w;
    output_f4[idx] = val;
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
    fprintf(stderr, "Error: layernorm hidden_dim (%d) is not divisible by 4\n", hidden_dim);
    printf("Error: layernorm hidden_dim (%d) is not divisible by 4\n", hidden_dim);
    fflush(stderr);
    fflush(stdout);
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

  // __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  // __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];
  
  // cg::thread_block b = cg::this_thread_block();
  // cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);
  
  // int col = blockIdx.x * blockDim.x + threadIdx.x;
  // float sum_dgamma = 0.0f;
  // float sum_dbetta = 0.0f;
  
  // if (col < width) {
  //     for (int row = threadIdx.y; row < rows; row += blockDim.y) {
  //         int idx = row * width + col;
  //         float dout = out_grad[idx];
  //         float x = inp[idx];
  //         float xhat;
          
  //         if (means != nullptr) {
  //             float mean = means[row];
  //             float var = vars[row];
  //             float std = sqrtf(var);
  //             xhat = (x - mean) / (std + LN_EPSILON);
  //         } else {
  //             float be = betta[col];
  //             float ga = gamma[col];
  //             xhat = (x - be) / (ga + LN_EPSILON);
  //         }
          
  //         sum_dgamma += dout * xhat;
  //         sum_dbetta += dout;
  //     }
  // }

  // betta_buffer[threadIdx.y][threadIdx.x] = sum_dbetta;
  // gamma_buffer[threadIdx.y][threadIdx.x] = sum_dgamma;
  // __syncthreads();

  // if (threadIdx.y == 0) {
  //     sum_dbetta = 0.0f;
  //     sum_dgamma = 0.0f;
  //     for (int i = 0; i < TILE_DIM; i++) {
  //         sum_dbetta += betta_buffer[i][threadIdx.x];
  //         sum_dgamma += gamma_buffer[i][threadIdx.x];
  //     }
      
  //     for (int i = g.size() / 2; i > 0; i /= 2) {
  //         sum_dgamma += g.shfl_down(sum_dgamma, i);
  //         sum_dbetta += g.shfl_down(sum_dbetta, i);
  //     }
      
  //     if (threadIdx.x == 0 && col < width) {
  //         gamma_grad[blockIdx.x] = sum_dgamma;
  //         betta_grad[blockIdx.x] = sum_dbetta;
  //     }
  // }
  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int offset = threadIdx.y * width + idx;
  int y_stride = width * TILE_DIM;

  // Loop across inp height
  float dbetta = 0;
  float dgamma = 0;
  float dout, val;

  if (idx < width) {
    if (means == nullptr) {
      float vbetta = (float)betta[idx];
      float vgamma = (float)gamma[idx];
      for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        dout = (float)out_grad[offset];
        // inp is output when means is nullptr
        val = (float)inp[offset];
        dbetta += dout;
        dgamma += ((val - vbetta) / add_eps(vgamma) * dout);
        offset += y_stride;
      }
    } else {
      for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        dout = (float)out_grad[offset];
        // inp is input when means is not nullptr
        val = (float)inp[offset];
        dbetta += dout;
        dgamma += ((val - (float)means[r]) *
                   rsqrtf((float)vars[r] + LN_EPSILON) * dout);
        offset += y_stride;
      }
    }
  }

  // Save to shared memory buffer
  betta_buffer[threadIdx.x][threadIdx.y] = dbetta;
  gamma_buffer[threadIdx.x][threadIdx.y] = dgamma;
  __syncthreads();

  // Transpose for better memory access patterns in reduction
  float s1 = betta_buffer[threadIdx.y][threadIdx.x];
  float s2 = gamma_buffer[threadIdx.y][threadIdx.x];
  __syncthreads(); //为了让同一列的数据可以在同一个warp内规约
  // warp是按照threadIdx.x来划分的
  //第一个warp包含threadIdx.x=0 to 31且threadIdx.y = 0的所有线程
  //第二个warp包含threadIdx.x=0 to 31且threadIdx.y = 1的所有线程
  //我们实际想要做的是对于同一个threadIdx.x，不同threadIdx.y的值进行规约

  // Warp-level reduction using shfl_down
  for (int i = 1; i < TILE_DIM; i <<= 1) {
    s1 += g.shfl_down(s1, i);
    s2 += g.shfl_down(s2, i);
  }
  // 通过一个巧妙的读取方式模拟了转置的效果

  // Write results to global memory
  int pos = blockIdx.x * TILE_DIM + threadIdx.y;
  if (threadIdx.x == 0 && idx < width) {
    betta_grad[pos] = s1;
    gamma_grad[pos] = s2;
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
  
  // float dxhat_sum = 0.0f;
  // float dxhat_xhat_sum = 0.0f;
  
  // const float4 *out_grad_f4 = reinterpret_cast<const float4 *>(out_grad) + blockIdx.x * hidden_dim;
  // const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);
  // const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_dim;
  // const float4 *betta_f4 = reinterpret_cast<const float4 *>(betta);
  
  // for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
  //     float4 dout = out_grad_f4[idx];
  //     float4 g = gamma_f4[idx];
  //     float4 x = inp_f4[idx];

  //     float4 dxhat;
  //     dxhat.x = dout.x * g.x;
  //     dxhat.y = dout.y * g.y;
  //     dxhat.z = dout.z * g.z;
  //     dxhat.w = dout.w * g.w;

  //     float4 xhat;
  //     if (means != nullptr) {
  //         float mean = means[blockIdx.x];
  //         float var = max(vars[blockIdx.x], LN_EPSILON);
  //         float std = sqrtf(var);
          
  //         xhat.x = (x.x - mean) / (std + LN_EPSILON);
  //         xhat.y = (x.y - mean) / (std + LN_EPSILON);
  //         xhat.z = (x.z - mean) / (std + LN_EPSILON);
  //         xhat.w = (x.w - mean) / (std + LN_EPSILON);
  //     } else {
  //         float4 b = betta_f4[idx];
          
  //         xhat.x = (x.x - b.x) / (g.x + LN_EPSILON);
  //         xhat.y = (x.y - b.y) / (g.y + LN_EPSILON);
  //         xhat.z = (x.z - b.z) / (g.z + LN_EPSILON);
  //         xhat.w = (x.w - b.w) / (g.w + LN_EPSILON);
  //     }

  //     dxhat_sum += dxhat.x + dxhat.y + dxhat.z + dxhat.w;
  //     dxhat_xhat_sum += dxhat.x * xhat.x + dxhat.y * xhat.y + 
  //                       dxhat.z * xhat.z + dxhat.w * xhat.w;
  // }

  // blockReduce<ReduceType::kSum, 1>(&dxhat_sum);
  // blockReduce<ReduceType::kSum, 1>(&dxhat_xhat_sum);
  // __syncthreads();

  // float4 *inp_grad_f4 = reinterpret_cast<float4 *>(inp_grad) + blockIdx.x * hidden_dim;
  // float m = hidden_dim * 4.0f;
  // float var = max(vars[blockIdx.x], LN_EPSILON);
  // float std = sqrtf(var);
  
  // for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
  //     float4 dout = out_grad_f4[idx];
  //     float4 g = gamma_f4[idx];
  //     float4 x = inp_f4[idx];

  //     float4 xhat;
  //     if (means != nullptr) {
  //         float mean = means[blockIdx.x];
  //         xhat.x = (x.x - mean) / (std + LN_EPSILON);
  //         xhat.y = (x.y - mean) / (std + LN_EPSILON);
  //         xhat.z = (x.z - mean) / (std + LN_EPSILON);
  //         xhat.w = (x.w - mean) / (std + LN_EPSILON);
  //     } else {
  //         float4 b = betta_f4[idx];
          
  //         xhat.x = (x.x - b.x) / (g.x + LN_EPSILON);
  //         xhat.y = (x.y - b.y) / (g.y + LN_EPSILON);
  //         xhat.z = (x.z - b.z) / (g.z + LN_EPSILON);
  //         xhat.w = (x.w - b.w) / (g.w + LN_EPSILON);
  //     }
      
  //     float4 dx;
  //     dx.x = (dout.x * g.x - (dxhat_sum + xhat.x * dxhat_xhat_sum) / m) / std;
  //     dx.y = (dout.y * g.y - (dxhat_sum + xhat.y * dxhat_xhat_sum) / m) / std;
  //     dx.z = (dout.z * g.z - (dxhat_sum + xhat.z * dxhat_xhat_sum) / m) / std;
  //     dx.w = (dout.w * g.w - (dxhat_sum + xhat.w * dxhat_xhat_sum) / m) / std;
      
  //     inp_grad_f4[idx] = dx;
  // }

  int offset = blockIdx.x * hidden_dim + threadIdx.x;
  
  float4 dxhat, xhat;
  float var_rsqrt;
  
  if (threadIdx.x < hidden_dim) {

    dxhat = ((const float4 *)out_grad)[offset];
    float4 vgamma = ((const float4 *)gamma)[threadIdx.x];
    dxhat.x *= vgamma.x;
    dxhat.y *= vgamma.y;
    dxhat.z *= vgamma.z;
    dxhat.w *= vgamma.w;
    
    // xhat = (input - mean) * rsqrt(var) 或 (output - betta) / gamma
    xhat = ((const float4 *)inp)[offset];
    var_rsqrt = rsqrtf((float)vars[blockIdx.x] + LN_EPSILON);
    
    if (means == nullptr) {
      // inp是output，计算xhat = (output - betta) / gamma
      float4 vbetta = ((const float4 *)betta)[threadIdx.x];
      xhat.x = (xhat.x - vbetta.x) / add_eps(vgamma.x);
      xhat.y = (xhat.y - vbetta.y) / add_eps(vgamma.y);
      xhat.z = (xhat.z - vbetta.z) / add_eps(vgamma.z);
      xhat.w = (xhat.w - vbetta.w) / add_eps(vgamma.w);
    } else {
      // inp是input，计算xhat = (input - mean) * rsqrt(var)
      float fmean = (float)means[blockIdx.x];
      
      xhat.x = (xhat.x - fmean) * var_rsqrt;
      xhat.y = (xhat.y - fmean) * var_rsqrt;
      xhat.z = (xhat.z - fmean) * var_rsqrt;
      xhat.w = (xhat.w - fmean) * var_rsqrt;
    }
  }
  
  float reduce_val[2] = {0.f, 0.f};
  if (threadIdx.x < hidden_dim) {
    reduce_val[0] = dxhat.x + dxhat.y + dxhat.z + dxhat.w;
    reduce_val[1] = dxhat.x * xhat.x + dxhat.y * xhat.y + 
                    dxhat.z * xhat.z + dxhat.w * xhat.w;
  }
  
  blockReduce<ReduceType::kSum, 2>(reduce_val);
  
  __shared__ float s_sum_dxhat, s_sum_dxhat_xhat;
  if (threadIdx.x == 0) {
    float mean_dim = hidden_dim * 4;
    s_sum_dxhat = reduce_val[0] / mean_dim;
    s_sum_dxhat_xhat = reduce_val[1] / mean_dim;
  }
  __syncthreads();
  

  if (threadIdx.x >= hidden_dim) {
    return;
  }
  
  dxhat.x = (dxhat.x - s_sum_dxhat - xhat.x * s_sum_dxhat_xhat) * var_rsqrt;
  dxhat.y = (dxhat.y - s_sum_dxhat - xhat.y * s_sum_dxhat_xhat) * var_rsqrt;
  dxhat.z = (dxhat.z - s_sum_dxhat - xhat.z * s_sum_dxhat_xhat) * var_rsqrt;
  dxhat.w = (dxhat.w - s_sum_dxhat - xhat.w * s_sum_dxhat_xhat) * var_rsqrt;

  ((float4 *)inp_grad)[offset] = dxhat;

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
