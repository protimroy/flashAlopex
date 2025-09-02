#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <stdint.h>

template <typename scalar_t>
__device__ inline float to_float_cast(scalar_t v) { return static_cast<float>(v); }
template <> __device__ inline float to_float_cast<half>(half v) { return __half2float(v); }
template <> __device__ inline float to_float_cast<nv_bfloat16>(nv_bfloat16 v) { return __bfloat162float(v); }

template <typename scalar_t>
__device__ inline scalar_t from_float_cast(float v) { return static_cast<scalar_t>(v); }
template <> __device__ inline half from_float_cast<half>(float v) { return __float2half(v); }
template <> __device__ inline nv_bfloat16 from_float_cast<nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename scalar_t>
__global__ void alopex_step_kernel(
    scalar_t* __restrict__ param,
    signed char* __restrict__ xstate,  // int8: -1 or +1
    const float delta,
    const float p_flip,
    const int64_t N,
    const uint64_t seed,
    const uint64_t offset
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // cuRAND Philox: unique RNG per element via offset+idx
    curandStatePhilox4_32_10_t st;
    curand_init(seed, /*sequence=*/0ULL, /*offset=*/offset + static_cast<uint64_t>(idx), &st);
    float r = curand_uniform(&st);  // (0,1]

    int s = static_cast<int>(xstate[idx]);   // -1 or +1
    int sgn = (r < p_flip) ? 1 : -1;
    s = s * sgn;
    xstate[idx] = static_cast<signed char>(s);

    float pval = to_float_cast<scalar_t>(param[idx]);
    pval += delta * static_cast<float>(s);
    param[idx] = from_float_cast<scalar_t>(pval);
}

// C++ dispatcher (no PYBIND11_MODULE here)
void alopex_step_cuda(at::Tensor param, at::Tensor xstate, double delta, double p_flip, uint64_t seed, uint64_t offset) {
    TORCH_CHECK(param.is_cuda(), "param must be CUDA");
    TORCH_CHECK(xstate.is_cuda(), "xstate must be CUDA");
    TORCH_CHECK(param.is_contiguous(), "param must be contiguous");
    TORCH_CHECK(xstate.is_contiguous(), "xstate must be contiguous");
    TORCH_CHECK(param.numel() == xstate.numel(), "param and xstate size mismatch");
    TORCH_CHECK(xstate.dtype() == at::kChar, "xstate must be int8");
    TORCH_CHECK(p_flip >= 0.0 && p_flip <= 1.0, "p_flip must be in [0,1]");

    int64_t N = param.numel();
    if (N == 0) return;

    const int threads = 256;
    const int blocks = static_cast<int>((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, param.scalar_type(), "alopex_step_kernel", [&] {
        alopex_step_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            param.data_ptr<scalar_t>(),
            xstate.data_ptr<signed char>(),
            static_cast<float>(delta),
            static_cast<float>(p_flip),
            N,
            seed,
            offset
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}