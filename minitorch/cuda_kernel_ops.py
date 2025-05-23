from typing import Callable, Optional

from . import operators
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps
from .tensor_functions import tensor_from_numpy

import ctypes
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import torch

# Load the shared library
lib = ctypes.CDLL("minitorch/cuda_kernels/combine.so")
lib_softmax = ctypes.CDLL("minitorch/cuda_kernels/softmax_kernel.so")
lib_layernorm = ctypes.CDLL("minitorch/cuda_kernels/layernorm_kernel.so")
datatype = np.float32

# function map
fn_map = {
  operators.add: 1,
  operators.mul: 2,
  operators.id: 3,
  operators.neg: 4,
  operators.lt: 5,
  operators.eq: 6,
  operators.sigmoid: 7,
  operators.relu: 8,
  operators.relu_back: 9,
  operators.log: 10,
  operators.log_back: 11,
  operators.exp: 12,
  operators.inv: 13,
  operators.inv_back: 14,
  operators.is_close: 15,
  operators.max: 16,
  operators.pow: 17, 
  operators.tanh: 18
}

THREADS_PER_BLOCK = 32

class CudaKernelOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        fn_id = fn_map[fn]

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            lib.tensorMap.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # in_size
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorMap.restype = None
            
            # assert out.size == a.size, f"zip {out.size}, {a.size}"

            lib.tensorMap(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)

            lib.tensorZip.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                ctypes.c_int,                                                            # out_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_strides
                ctypes.c_int,                                                            # a_size
                ctypes.c_int,                                                            # a_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # b_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_strides
                ctypes.c_int,                                                            # b_size
                ctypes.c_int,                                                            # b_shape_size
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorZip.restype = None

            # assert out.size == a.size, f"zip {out.size}, {a.size}"
            # assert out.size == b.size, f"zip {out.size}, {b.size}"

            lib.tensorZip(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                len(out.shape),
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                b._tensor._storage,
                b._tensor._shape.astype(np.int32),
                b._tensor._strides.astype(np.int32),
                b.size,
                len(b.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0) -> Callable[[Tensor, int], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))

            lib.tensorReduce.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # reduce_dim
                ctypes.c_double,                                                         # reduce_value
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorReduce.restype = None

            lib.tensorReduce(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                dim,
                start,
                len(a.shape),
                fn_id
            )

            return out

        return ret

    @staticmethod
    def matrix_multiply_cublas(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]

        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2],
                                    a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2],
                                    b.shape[-1])
        assert a.shape[0] == b.shape[0]

        bs, m, n, k = a.shape[0], a.shape[1], b.shape[2], a.shape[2]
        A, B = a.to_numpy(), b.to_numpy()

        # Convert A and B to column-major order
        A_fortran = np.transpose(A, (0, 2, 1))
        B_fortran = np.transpose(B, (0, 2, 1))

        # Flatten A and B for sending to GPU
        A_flat = A_fortran.reshape(bs, -1)
        B_flat = B_fortran.reshape(bs, -1)

        # Allocate memory on GPU
        A_gpu = cuda.mem_alloc(A_flat.nbytes)
        B_gpu = cuda.mem_alloc(B_flat.nbytes)
        C_gpu = cuda.mem_alloc(bs * m * n * A.itemsize)

        # Copy data to GPU
        cuda.memcpy_htod(A_gpu, A_flat)
        cuda.memcpy_htod(B_gpu, B_flat)

        # Prepare arrays of pointers
        A_gpu_ptrs = np.array(
            [int(A_gpu) + i * m * k * A.itemsize for i in range(bs)],
            dtype=np.uint64)
        B_gpu_ptrs = np.array(
            [int(B_gpu) + i * k * n * B.itemsize for i in range(bs)],
            dtype=np.uint64)
        C_gpu_ptrs = np.array(
            [int(C_gpu) + i * m * n * A.itemsize for i in range(bs)],
            dtype=np.uint64)

        # Allocate device memory for arrays of pointers
        A_array_gpu = cuda.mem_alloc(A_gpu_ptrs.nbytes)
        B_array_gpu = cuda.mem_alloc(B_gpu_ptrs.nbytes)
        C_array_gpu = cuda.mem_alloc(C_gpu_ptrs.nbytes)

        # Copy arrays of pointers to device memory
        cuda.memcpy_htod(A_array_gpu, A_gpu_ptrs)
        cuda.memcpy_htod(B_array_gpu, B_gpu_ptrs)
        cuda.memcpy_htod(C_array_gpu, C_gpu_ptrs)

        # Set argument types for the kernel function
        lib_mm.batchedMatMulKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int]

        # Launch kernel
        lib_mm.batchedMatMulKernel(
            int(A_array_gpu), int(B_array_gpu), int(C_array_gpu), m, k, n, bs)

        # Synchronize device to ensure computation is complete
        cuda.Context.synchronize()

        # Copy back the result
        C = np.empty((bs, n, m), dtype=A.dtype)
        cuda.memcpy_dtoh(C, C_gpu)
        C = np.transpose(C, (0, 2, 1))

        c = tensor_from_numpy(
            np.ascontiguousarray(C),
            backend=a.backend, requires_grad=a.requires_grad()).contiguous()

        # Undo 3d if we added it.
        if both_2d:
            c = c.view(c.shape[1], c.shape[2])
        if len(ls) > 3:
            c = c.view(*ls)
        return c

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # handle cases with more dimensions [64, 4, 32, 128] x [64, 4, 128, 32]
        more_3d = False
        if len(out.shape) > 3:
            # print(f"Debug in matmul: output shape {ls}")
            more_3d = True
            out = out.view(np.prod(out.shape[:-2]), out.shape[-2], out.shape[-1])
            nshape = out._tensor._shape
            nstrides = out._tensor._strides
            # print(f"Debug in matmul: batched dim [:-2] and get the strides {nshape, nstrides}")
        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2], a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2], b.shape[-1])
        
        assert a.shape[0] == b.shape[0]
        assert a.shape[0] == out.shape[0]

        lib.MatrixMultiply.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # b_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_strides
            ctypes.c_int,                                                             # batch_size
            ctypes.c_int,                                                             # out_shape[1], m
            ctypes.c_int                                                              # out_shape[2], p
        ]

        lib.MatrixMultiply.restype = None

        assert len(out._tensor._shape) == 3, f"{len(out._tensor._shape)}"
        assert len(out._tensor._strides) == 3, f"{len(out._tensor._strides)}"
        assert len(a._tensor._shape) == 3
        assert len(a._tensor._strides) == 3
        assert len(b._tensor._shape) == 3
        assert len(b._tensor._strides) == 3

        lib.MatrixMultiply(
            out._tensor._storage,
            out._tensor._shape.astype(np.int32),
            out._tensor._strides.astype(np.int32),
            a._tensor._storage,
            a._tensor._shape.astype(np.int32),
            a._tensor._strides.astype(np.int32),
            b._tensor._storage,
            b._tensor._shape.astype(np.int32),
            b._tensor._strides.astype(np.int32),
            a.shape[0],
            a.shape[1],
            b.shape[2]
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        if more_3d:
            out = out.view(*ls)
            # print(f"Debug in matmul: output shape {out.shape}")
        return out

    @staticmethod
    def attn_softmax_fw(inp: Tensor, mask: Tensor):
      batch_size, nhead, from_len, to_len = inp.shape
      is_dec_self_attn = False
      stream = torch.cuda.current_stream().cuda_stream

      lib_softmax.launch_attn_softmax.argtypes = [
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_void_p
      ]
      lib_softmax.launch_attn_softmax.restype = None

      lib_softmax.launch_attn_softmax(
        inp._tensor._storage,
        mask._tensor._storage,
        batch_size,
        nhead,
        from_len,
        to_len,
        is_dec_self_attn,
        stream
      ) 

      return inp

    @staticmethod
    def attn_softmax_bw(out_grad: Tensor, soft_inp: Tensor):
      #   BEGIN ASSIGN3_1
        #   raise("Not implemented")
        # Get dimensions
        batch_size, nhead, from_len, to_len = out_grad.shape
        rows = batch_size * nhead * from_len  # total number of rows
        softmax_len = to_len  # sequence length
        
        # Get CUDA stream
        stream = torch.cuda.current_stream().cuda_stream
        
        # Define argument types for the CUDA kernel
        lib_softmax.launch_attn_softmax_bw.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out_grad
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # soft_inp
            ctypes.c_int,    # rows
            ctypes.c_int,    # softmax_len
            ctypes.c_void_p  # stream
        ]
        lib_softmax.launch_attn_softmax_bw.restype = None
        
        # Call the CUDA kernel
        lib_softmax.launch_attn_softmax_bw(
            out_grad._tensor._storage,
            soft_inp._tensor._storage,
            rows,
            softmax_len,
            stream
        )
        
        return out_grad
      #   END ASSIGN3_1

    @staticmethod
    def layernorm_fw(inp: Tensor, gamma: Tensor, beta: Tensor):
      #   BEGIN ASSIGN3_2
    #   raise("Not implemented")
        # 获取输入维度
        batch_size = inp.shape[0]
        hidden_dim = inp.shape[1]
        
        # 获取CUDA流
        stream = torch.cuda.current_stream().cuda_stream
        
        # 定义参数类型
        lib_layernorm.launch_layernorm.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # ln_res
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # vars
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # means
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # inp
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # scale
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # bias
            ctypes.c_int,    # batch_size
            ctypes.c_int,    # hidden_dim
            ctypes.c_void_p  # stream
        ]
        lib_layernorm.launch_layernorm.restype = None
        
        # 创建输出张量
        out = inp.zeros(inp.shape)
        
        # 创建中间结果张量
        vars = inp.zeros((batch_size,))
        means = inp.zeros((batch_size,))
        
        # 调用CUDA核函数
        lib_layernorm.launch_layernorm(
            out._tensor._storage,
            vars._tensor._storage,
            means._tensor._storage,
            inp._tensor._storage,
            gamma._tensor._storage,
            beta._tensor._storage,
            batch_size,
            hidden_dim,
            stream
        )
        
        return out
      #   END ASSIGN3_2
      
    @staticmethod
    def layernorm_bw(out_grad: Tensor, inp: Tensor, gamma: Tensor, beta: Tensor, var: Tensor, mean: Tensor):
      #   BEGIN ASSIGN3_2
    #   raise("Not implemented")
        # 获取维度
        batch_size = out_grad.shape[0]
        hidden_dim = out_grad.shape[1]
        
        # 获取CUDA流
        stream_1 = torch.cuda.current_stream().cuda_stream
        stream_2 = torch.cuda.current_stream().cuda_stream
        
        # 定义参数类型
        lib_layernorm.launch_layernorm_bw.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # gamma_grad
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # beta_grad
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # inp_grad
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out_grad
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # inp
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # gamma
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # beta
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # vars
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # means
            ctypes.c_int,    # batch_size
            ctypes.c_int,    # hidden_dim
            ctypes.c_void_p, # stream_1
            ctypes.c_void_p  # stream_2
        ]
        lib_layernorm.launch_layernorm_bw.restype = None
        
        # 创建输出梯度张量
        gamma_grad = gamma.zeros(gamma.shape)
        beta_grad = beta.zeros(beta.shape)
        inp_grad = inp.zeros(inp.shape)
        
        # 调用CUDA核函数
        lib_layernorm.launch_layernorm_bw(
            gamma_grad._tensor._storage,
            beta_grad._tensor._storage,
            inp_grad._tensor._storage,
            out_grad._tensor._storage,
            inp._tensor._storage,
            gamma._tensor._storage,
            beta._tensor._storage,
            var._tensor._storage,
            mean._tensor._storage,
            batch_size,
            hidden_dim,
            stream_1,
            stream_2
        )
        
        return inp_grad, gamma_grad, beta_grad   
      #   END ASSIGN3_2
      
    @staticmethod
    def flash_attention_fw(q: Tensor, k: Tensor, v: Tensor, causal: bool = False) -> Tensor:
        """Flash Attention forward implementation"""
        # 1. 获取输入维度
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # 2. 加载库并定义C函数的参数类型
        lib_flash = ctypes.CDLL("minitorch/cuda_kernels/flash_attention.so")
        lib_flash.flashAttentionForward.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # q
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # k
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # v
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # l
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # m
            ctypes.c_int,    # batch_size
            ctypes.c_int,    # num_heads
            ctypes.c_int,    # seq_len
            ctypes.c_int,    # head_dim
            ctypes.c_float,  # sm_scale
            ctypes.c_bool    # causal
        ]
        lib_flash.flashAttentionForward.restype = None
        
        # 3. 创建输出和中间结果tensor
        out = q.zeros(q.shape)
        l = q.zeros((batch_size, num_heads, seq_len))
        m = q.zeros((batch_size, num_heads, seq_len))
        
        # 4. 调用CUDA kernel
        lib_flash.flashAttentionForward(
            q._tensor._storage,
            k._tensor._storage,
            v._tensor._storage,
            out._tensor._storage,
            l._tensor._storage,
            m._tensor._storage,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            1.0 / np.sqrt(head_dim),
            causal
        )
        
        # 5. 保存反向传播需要的context
        out.ctx = {
            'q': q,
            'k': k,
            'v': v,
            'l': l,
            'm': m,
            'causal': causal,
            'head_dim': head_dim,
            'out': out  # 保存输出用于反向传播
        }
        
        return out

    @staticmethod
    def flash_attention_bw(ctx, grad_output):
        """Flash Attention backward implementation"""
        # 1. 获取保存的context
        q = ctx['q']
        k = ctx['k']
        v = ctx['v']
        l = ctx['l']
        m = ctx['m']
        causal = ctx['causal']
        head_dim = ctx['head_dim']
        out = ctx['out']
        batch_size, num_heads, seq_len, _ = q.shape
        
        # 2. 加载库并定义C函数的参数类型
        lib_flash = ctypes.CDLL("minitorch/cuda_kernels/flash_attention.so")
        lib_flash.flashAttentionBackward.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # q
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # k
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # v
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # dout
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # dq
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # dk
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # dv
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # l
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # m
            ctypes.c_int,    # batch_size
            ctypes.c_int,    # num_heads
            ctypes.c_int,    # seq_len
            ctypes.c_int,    # head_dim
            ctypes.c_float,  # sm_scale
            ctypes.c_bool    # causal
        ]
        lib_flash.flashAttentionBackward.restype = None
        
        # 3. 创建梯度tensor
        dq = q.zeros_like()
        dk = k.zeros_like()
        dv = v.zeros_like()
        
        # 4. 调用CUDA kernel
        lib_flash.flashAttentionBackward(
            q._tensor._storage,
            k._tensor._storage,
            v._tensor._storage,
            out._tensor._storage,     # 前向传播的输出
            grad_output._tensor._storage,  # 反向传播的梯度
            dq._tensor._storage,
            dk._tensor._storage,
            dv._tensor._storage,
            l._tensor._storage,
            m._tensor._storage,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            1.0 / np.sqrt(head_dim),
            causal
        )
        
        return dq, dk, dv
      
