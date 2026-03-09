#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace relm::mp {

using at::Tensor;

namespace {

constexpr int64_t kThreads = 256;

at::ScalarType dtype_of(const Tensor& t)
{
   return t.scalar_type();
}

bool is_fastpath_dtype(at::ScalarType dtype)
{
   return dtype == at::kFloat || dtype == at::kDouble;
}

Tensor ensure_contiguous(const Tensor& t)
{
   return t.is_contiguous() ? t : t.contiguous();
}

void check_same_cuda_device(
   const Tensor& lhs,
   const Tensor& rhs,
   const char* lhs_name,
   const char* rhs_name
)
{
   TORCH_CHECK(
      lhs.get_device() == rhs.get_device(),
      lhs_name,
      " and ",
      rhs_name,
      " must be on the same CUDA device."
   );
}

cudaStream_t current_cuda_stream(const Tensor& t)
{
   return at::cuda::getCurrentCUDAStream(t.get_device()).stream();
}

void check_kernel_launch(const char* kernel_name)
{
   const cudaError_t err = cudaGetLastError();
   TORCH_CHECK(err == cudaSuccess, kernel_name, " launch failed: ", cudaGetErrorString(err));
}

int grid_for(int64_t total)
{
   const int64_t blocks = (total + kThreads - 1) / kThreads;
   return static_cast< int >(
      std::min< int64_t >(blocks, static_cast< int64_t >(std::numeric_limits< int >::max()))
   );
}

enum class FanoutBackwardKernelMode
{
   kAuto,
   k1D,
   k2D,
};

FanoutBackwardKernelMode fanout_backward_kernel_mode()
{
   static const FanoutBackwardKernelMode mode = []() {
      const char* raw = std::getenv("RELM_MP_FANOUT_BWD_KERNEL");
      if(raw == nullptr) {
         return FanoutBackwardKernelMode::kAuto;
      }
      const std::string value(raw);
      if(value == "1d" || value == "legacy") {
         return FanoutBackwardKernelMode::k1D;
      }
      if(value == "2d") {
         return FanoutBackwardKernelMode::k2D;
      }
      return FanoutBackwardKernelMode::kAuto;
   }();
   return mode;
}

bool choose_fanout_backward_2d_kernel(int64_t num_edges, int64_t emb)
{
   const FanoutBackwardKernelMode mode = fanout_backward_kernel_mode();
   if(mode == FanoutBackwardKernelMode::k1D) {
      return false;
   }
   if(mode == FanoutBackwardKernelMode::k2D) {
      return true;
   }

   constexpr int64_t kAuto2DMinWork = 1 << 20;  // One million edge-dimension updates.
   if(emb <= 0) {
      return false;
   }
   return num_edges >= (kAuto2DMinWork + emb - 1) / emb;
}

enum class FaninLogSumExpForwardMode
{
   kAuto,
   kAtomic,
   kSegmented,
};

FaninLogSumExpForwardMode fanin_logsumexp_forward_mode()
{
   static const FaninLogSumExpForwardMode mode = []() {
      const char* raw = std::getenv("RELM_MP_FANIN_LSE_FWD");
      if(raw == nullptr) {
         return FaninLogSumExpForwardMode::kAuto;
      }
      const std::string value(raw);
      if(value == "atomic" || value == "legacy" || value == "1d") {
         return FaninLogSumExpForwardMode::kAtomic;
      }
      if(value == "segmented" || value == "sorted") {
         return FaninLogSumExpForwardMode::kSegmented;
      }
      return FaninLogSumExpForwardMode::kAuto;
   }();
   return mode;
}

bool choose_fanin_logsumexp_segmented(int64_t num_edges, int64_t dim_size, int64_t emb)
{
   const FaninLogSumExpForwardMode mode = fanin_logsumexp_forward_mode();
   if(mode == FaninLogSumExpForwardMode::kAtomic) {
      return false;
   }
   if(mode == FaninLogSumExpForwardMode::kSegmented) {
      return true;
   }

   if(num_edges <= 0 || dim_size <= 0 || emb <= 0) {
      return false;
   }
   const int64_t avg_degree = (num_edges + dim_size - 1) / dim_size;
   if(avg_degree > 16) {
      return false;
   }
   return num_edges >= (1 << 16) && emb >= 128;
}

template < typename scalar_t >
__device__ inline void atomic_add_compat(scalar_t* dst, scalar_t value)
{
   atomicAdd(dst, value);
}

template <>
__device__ inline void atomic_add_compat< double >(double* dst, double value)
{
#if __CUDA_ARCH__ >= 600
   atomicAdd(dst, value);
#else
   auto* dst_bits = reinterpret_cast< unsigned long long int* >(dst);
   unsigned long long int old_bits = *dst_bits;
   unsigned long long int assumed_bits = 0;
   do {
      assumed_bits = old_bits;
      old_bits = atomicCAS(
         dst_bits, assumed_bits, __double_as_longlong(value + __longlong_as_double(assumed_bits))
      );
   } while(assumed_bits != old_bits);
#endif
}

template < typename scalar_t >
__device__ inline void atomic_max_compat(scalar_t* dst, scalar_t value);

template <>
__device__ inline void atomic_max_compat< float >(float* dst, float value)
{
   auto* dst_bits = reinterpret_cast< int* >(dst);
   int old_bits = *dst_bits;
   while(true) {
      const float old_value = __int_as_float(old_bits);
      if(!(old_value < value)) {
         return;
      }
      const int new_bits = __float_as_int(value);
      const int prev_bits = atomicCAS(dst_bits, old_bits, new_bits);
      if(prev_bits == old_bits) {
         return;
      }
      old_bits = prev_bits;
   }
}

template <>
__device__ inline void atomic_max_compat< double >(double* dst, double value)
{
   auto* dst_bits = reinterpret_cast< unsigned long long int* >(dst);
   unsigned long long int old_bits = *dst_bits;
   while(true) {
      const double old_value = __longlong_as_double(static_cast< long long int >(old_bits));
      if(!(old_value < value)) {
         return;
      }
      const unsigned long long int new_bits =
         static_cast< unsigned long long int >(__double_as_longlong(value));
      const unsigned long long int prev_bits = atomicCAS(dst_bits, old_bits, new_bits);
      if(prev_bits == old_bits) {
         return;
      }
      old_bits = prev_bits;
   }
}

template < typename scalar_t >
__device__ inline scalar_t exp_compat(scalar_t value)
{
   return exp(value);
}

template <>
__device__ inline float exp_compat< float >(float value)
{
   return expf(value);
}

template < typename scalar_t >
__device__ inline scalar_t log_compat(scalar_t value)
{
   return log(value);
}

template <>
__device__ inline float log_compat< float >(float value)
{
   return logf(value);
}

template < typename scalar_t >
__device__ inline scalar_t erf_compat(scalar_t value)
{
   return erf(value);
}

template <>
__device__ inline float erf_compat< float >(float value)
{
   return erff(value);
}

template < typename scalar_t >
__device__ inline scalar_t softplus_compat(scalar_t value)
{
   return log_compat(static_cast< scalar_t >(1) + exp_compat(value));
}

template < typename scalar_t >
__device__ inline scalar_t mish_compat(scalar_t value)
{
   return value * tanh(softplus_compat(value));
}

template < typename scalar_t >
__device__ inline scalar_t sigmoid_compat(scalar_t value)
{
   return static_cast< scalar_t >(1)
          / (static_cast< scalar_t >(1) + exp_compat(-value));
}

template < typename scalar_t >
__device__ inline scalar_t mish_grad_from_pre_activation_compat(scalar_t value)
{
   const scalar_t sp = softplus_compat(value);
   const scalar_t tsp = tanh(sp);
   const scalar_t sig = sigmoid_compat(value);
   return tsp + value * sig * (static_cast< scalar_t >(1) - tsp * tsp);
}

template < typename scalar_t >
__device__ inline scalar_t silu_compat(scalar_t value)
{
   return value * sigmoid_compat(value);
}

template < typename scalar_t >
__device__ inline scalar_t silu_grad_from_pre_activation_compat(scalar_t value)
{
   const scalar_t sig = sigmoid_compat(value);
   return sig * (static_cast< scalar_t >(1) + value * (static_cast< scalar_t >(1) - sig));
}

template < typename scalar_t >
__device__ inline scalar_t gelu_none_compat(scalar_t value)
{
   constexpr double kInvSqrt2 = 0.70710678118654752440;
   const scalar_t scaled = value * static_cast< scalar_t >(kInvSqrt2);
   return static_cast< scalar_t >(0.5) * value * (static_cast< scalar_t >(1) + erf_compat(scaled));
}

template < typename scalar_t >
__device__ inline scalar_t gelu_none_grad_from_pre_activation_compat(scalar_t value)
{
   constexpr double kInvSqrt2 = 0.70710678118654752440;
   constexpr double kInvSqrt2Pi = 0.39894228040143267794;
   const scalar_t scaled = value * static_cast< scalar_t >(kInvSqrt2);
   const scalar_t erf_term = erf_compat(scaled);
   const scalar_t exp_term =
      exp_compat(static_cast< scalar_t >(-0.5) * value * value)
      * static_cast< scalar_t >(kInvSqrt2Pi);
   return static_cast< scalar_t >(0.5) * (static_cast< scalar_t >(1) + erf_term)
          + value * exp_term;
}

template < typename scalar_t >
__device__ inline scalar_t gelu_tanh_compat(scalar_t value)
{
   constexpr double kSqrt2OverPi = 0.79788456080286535588;
   constexpr double kCubic = 0.044715;
   const scalar_t inner = static_cast< scalar_t >(kSqrt2OverPi)
                          * (value + static_cast< scalar_t >(kCubic) * value * value * value);
   return static_cast< scalar_t >(0.5) * value * (static_cast< scalar_t >(1) + tanh(inner));
}

template < typename scalar_t >
__device__ inline scalar_t gelu_tanh_grad_from_pre_activation_compat(scalar_t value)
{
   constexpr double kSqrt2OverPi = 0.79788456080286535588;
   constexpr double kCubic = 0.044715;
   const scalar_t x2 = value * value;
   const scalar_t inner = static_cast< scalar_t >(kSqrt2OverPi)
                          * (value + static_cast< scalar_t >(kCubic) * value * x2);
   const scalar_t tanh_inner = tanh(inner);
   const scalar_t sech2 = static_cast< scalar_t >(1) - tanh_inner * tanh_inner;
   const scalar_t inner_grad = static_cast< scalar_t >(kSqrt2OverPi)
                               * (static_cast< scalar_t >(1)
                                  + static_cast< scalar_t >(3.0 * kCubic) * x2);
   return static_cast< scalar_t >(0.5) * (static_cast< scalar_t >(1) + tanh_inner)
          + static_cast< scalar_t >(0.5) * value * sech2 * inner_grad;
}

template < typename scalar_t >
__device__ inline scalar_t pointwise_apply_compat(int64_t code, scalar_t value)
{
   if(code == 0) {
      return value;
   }
   if(code == 1) {
      return value > static_cast< scalar_t >(0) ? value : static_cast< scalar_t >(0);
   }
   if(code == 2) {
      return mish_compat(value);
   }
   if(code == 3) {
      return gelu_none_compat(value);
   }
   if(code == 4) {
      return gelu_tanh_compat(value);
   }
   if(code == 5) {
      return silu_compat(value);
   }
   if(code == 6) {
      return tanh(value);
   }
   return value;
}

template < typename scalar_t >
__device__ inline scalar_t pointwise_grad_from_pre_activation_compat(int64_t code, scalar_t value)
{
   if(code == 0) {
      return static_cast< scalar_t >(1);
   }
   if(code == 1) {
      return value > static_cast< scalar_t >(0) ? static_cast< scalar_t >(1) : static_cast< scalar_t >(0);
   }
   if(code == 2) {
      return mish_grad_from_pre_activation_compat(value);
   }
   if(code == 3) {
      return gelu_none_grad_from_pre_activation_compat(value);
   }
   if(code == 4) {
      return gelu_tanh_grad_from_pre_activation_compat(value);
   }
   if(code == 5) {
      return silu_grad_from_pre_activation_compat(value);
   }
   if(code == 6) {
      const scalar_t t = tanh(value);
      return static_cast< scalar_t >(1) - t * t;
   }
   return static_cast< scalar_t >(1);
}

template < typename scalar_t >
__global__ void fanout_scatter_cuda_kernel(
   const scalar_t* x_ptr,
   const int64_t* src_ptr,
   const int64_t* dst_ptr,
   int64_t num_edges,
   int64_t emb,
   scalar_t* out_ptr
)
{
   const int64_t total = num_edges * emb;
   for(int64_t idx = static_cast< int64_t >(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast< int64_t >(blockDim.x) * gridDim.x) {
      const int64_t edge = idx / emb;
      const int64_t dim = idx - edge * emb;
      const int64_t src = src_ptr[edge];
      const int64_t dst = dst_ptr[edge];
      out_ptr[dst * emb + dim] = x_ptr[src * emb + dim];
   }
}

template < typename scalar_t >
__global__ void fanout_scatter_backward_cuda_kernel_1d(
   const scalar_t* grad_out_ptr,
   const int64_t* src_ptr,
   const int64_t* dst_ptr,
   int64_t num_edges,
   int64_t emb,
   scalar_t* grad_x_ptr
)
{
   const int64_t total = num_edges * emb;
   for(int64_t idx = static_cast< int64_t >(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast< int64_t >(blockDim.x) * gridDim.x) {
      const int64_t edge = idx / emb;
      const int64_t dim = idx - edge * emb;
      const int64_t src = src_ptr[edge];
      const int64_t dst = dst_ptr[edge];
      atomic_add_compat(grad_x_ptr + src * emb + dim, grad_out_ptr[dst * emb + dim]);
   }
}

template < typename scalar_t >
__global__ void fanout_scatter_backward_cuda_kernel_2d(
   const scalar_t* grad_out_ptr,
   const int64_t* src_ptr,
   const int64_t* dst_ptr,
   int64_t num_edges,
   int64_t emb,
   scalar_t* grad_x_ptr
)
{
   const int64_t dim = static_cast< int64_t >(blockIdx.x) * blockDim.x + threadIdx.x;
   if(dim >= emb) {
      return;
   }

   const int64_t edge0 = static_cast< int64_t >(blockIdx.y) * blockDim.y + threadIdx.y;
   const int64_t edge_stride = static_cast< int64_t >(gridDim.y) * blockDim.y;

   for(int64_t edge = edge0; edge < num_edges; edge += edge_stride) {
      const int64_t src = src_ptr[edge];
      const int64_t dst = dst_ptr[edge];
      atomic_add_compat(grad_x_ptr + src * emb + dim, grad_out_ptr[dst * emb + dim]);
   }
}

template < typename scalar_t >
__global__ void fanin_reduce_sum_cuda_kernel(
   const scalar_t* rel_ptr,
   const int64_t* src_ptr,
   const int64_t* dst_ptr,
   int64_t num_edges,
   int64_t emb,
   scalar_t* out_ptr
)
{
   const int64_t total = num_edges * emb;
   for(int64_t idx = static_cast< int64_t >(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast< int64_t >(blockDim.x) * gridDim.x) {
      const int64_t edge = idx / emb;
      const int64_t dim = idx - edge * emb;
      const int64_t src = src_ptr[edge];
      const int64_t dst = dst_ptr[edge];
      atomic_add_compat(out_ptr + dst * emb + dim, rel_ptr[src * emb + dim]);
   }
}

template < typename scalar_t >
__global__ void fanin_reduce_sum_backward_cuda_kernel(
   const scalar_t* grad_out_ptr,
   const int64_t* src_ptr,
   const int64_t* dst_ptr,
   int64_t num_edges,
   int64_t emb,
   scalar_t* grad_rel_ptr
)
{
   const int64_t total = num_edges * emb;
   for(int64_t idx = static_cast< int64_t >(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast< int64_t >(blockDim.x) * gridDim.x) {
      const int64_t edge = idx / emb;
      const int64_t dim = idx - edge * emb;
      const int64_t src = src_ptr[edge];
      const int64_t dst = dst_ptr[edge];
      atomic_add_compat(grad_rel_ptr + src * emb + dim, grad_out_ptr[dst * emb + dim]);
   }
}

template < typename scalar_t >
__global__ void fanin_reduce_logsumexp_max_cuda_kernel(
   const scalar_t* rel_ptr,
   const int64_t* src_ptr,
   const int64_t* dst_ptr,
   int64_t num_edges,
   int64_t emb,
   scalar_t* max_ptr
)
{
   const int64_t total = num_edges * emb;
   for(int64_t idx = static_cast< int64_t >(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast< int64_t >(blockDim.x) * gridDim.x) {
      const int64_t edge = idx / emb;
      const int64_t dim = idx - edge * emb;
      const int64_t src = src_ptr[edge];
      const int64_t dst = dst_ptr[edge];
      const scalar_t rel_val = rel_ptr[src * emb + dim];
      atomic_max_compat(max_ptr + dst * emb + dim, rel_val);
   }
}

template < typename scalar_t >
__global__ void fanin_reduce_logsumexp_sumexp_cuda_kernel(
   const scalar_t* rel_ptr,
   const int64_t* src_ptr,
   const int64_t* dst_ptr,
   const scalar_t* max_ptr,
   int64_t num_edges,
   int64_t emb,
   scalar_t* sum_ptr
)
{
   const int64_t total = num_edges * emb;
   for(int64_t idx = static_cast< int64_t >(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast< int64_t >(blockDim.x) * gridDim.x) {
      const int64_t edge = idx / emb;
      const int64_t dim = idx - edge * emb;
      const int64_t src = src_ptr[edge];
      const int64_t dst = dst_ptr[edge];
      const scalar_t rel_val = rel_ptr[src * emb + dim];
      const scalar_t max_val = max_ptr[dst * emb + dim];
      const scalar_t contrib = exp_compat< scalar_t >(rel_val - max_val);
      atomic_add_compat(sum_ptr + dst * emb + dim, contrib);
   }
}

template < typename scalar_t >
__global__ void fanin_reduce_logsumexp_finalize_cuda_kernel(
   const scalar_t* sum_ptr,
   int64_t total_out,
   scalar_t* out_ptr
)
{
   for(int64_t idx = static_cast< int64_t >(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total_out;
       idx += static_cast< int64_t >(blockDim.x) * gridDim.x) {
      out_ptr[idx] = log_compat< scalar_t >(sum_ptr[idx]) + out_ptr[idx];
   }
}

template < typename scalar_t >
__global__ void fanin_reduce_logsumexp_segmented_cuda_kernel(
   const scalar_t* rel_ptr,
   const int64_t* sorted_src_ptr,
   const int64_t* offsets_ptr,
   int64_t dim_size,
   int64_t emb,
   scalar_t* out_ptr
)
{
   const scalar_t neg_inf = -std::numeric_limits< scalar_t >::infinity();
   const int64_t total = dim_size * emb;
   for(int64_t idx = static_cast< int64_t >(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast< int64_t >(blockDim.x) * gridDim.x) {
      const int64_t dst = idx / emb;
      const int64_t dim = idx - dst * emb;
      const int64_t begin = offsets_ptr[dst];
      const int64_t end = offsets_ptr[dst + 1];
      const int64_t out_offset = dst * emb + dim;
      if(begin >= end) {
         out_ptr[out_offset] = neg_inf;
         continue;
      }

      scalar_t max_val = neg_inf;
      for(int64_t i = begin; i < end; ++i) {
         const int64_t src = sorted_src_ptr[i];
         const scalar_t v = rel_ptr[src * emb + dim];
         if(max_val < v) {
            max_val = v;
         }
      }

      scalar_t sum_exp = static_cast< scalar_t >(0);
      for(int64_t i = begin; i < end; ++i) {
         const int64_t src = sorted_src_ptr[i];
         const scalar_t v = rel_ptr[src * emb + dim];
         sum_exp += exp_compat< scalar_t >(v - max_val);
      }
      out_ptr[out_offset] = log_compat< scalar_t >(sum_exp) + max_val;
   }
}

template < typename scalar_t >
__global__ void fanin_reduce_logsumexp_backward_cuda_kernel(
   const scalar_t* grad_out_ptr,
   const scalar_t* rel_ptr,
   const int64_t* src_ptr,
   const int64_t* dst_ptr,
   const scalar_t* out_ptr,
   int64_t num_edges,
   int64_t emb,
   scalar_t* grad_rel_ptr
)
{
   const int64_t total = num_edges * emb;
   for(int64_t idx = static_cast< int64_t >(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast< int64_t >(blockDim.x) * gridDim.x) {
      const int64_t edge = idx / emb;
      const int64_t dim = idx - edge * emb;
      const int64_t src = src_ptr[edge];
      const int64_t dst = dst_ptr[edge];
      const scalar_t rel_val = rel_ptr[src * emb + dim];
      const scalar_t out_val = out_ptr[dst * emb + dim];
      const scalar_t grad_val = grad_out_ptr[dst * emb + dim];
      const scalar_t contrib = grad_val * exp_compat< scalar_t >(rel_val - out_val);
      atomic_add_compat(grad_rel_ptr + src * emb + dim, contrib);
   }
}

template < typename scalar_t >
void launch_fanout_scatter(
   const Tensor& x,
   const Tensor& src,
   const Tensor& dst,
   int64_t emb,
   Tensor& out,
   cudaStream_t stream
)
{
   const int64_t num_edges = src.size(0);
   const int64_t total = num_edges * emb;
   const int blocks = grid_for(total);
   fanout_scatter_cuda_kernel< scalar_t >
      <<<blocks, static_cast< int >(kThreads), 0, stream>>>(
         x.data_ptr< scalar_t >(),
         src.data_ptr< int64_t >(),
         dst.data_ptr< int64_t >(),
         num_edges,
         emb,
         out.data_ptr< scalar_t >()
      );
   check_kernel_launch("fanout_scatter_cuda_kernel");
}

template < typename scalar_t >
void launch_fanout_scatter_backward(
   const Tensor& grad_out,
   const Tensor& src,
   const Tensor& dst,
   int64_t emb,
   Tensor& grad_x,
   cudaStream_t stream
)
{
   const int64_t num_edges = src.size(0);
   if(choose_fanout_backward_2d_kernel(num_edges, emb)) {
      constexpr int threads_x = 32;
      constexpr int threads_y = 8;
      const dim3 threads(threads_x, threads_y, 1);

      const int64_t blocks_x64 = (emb + threads_x - 1) / threads_x;
      const int64_t blocks_y64 = (num_edges + threads_y - 1) / threads_y;
      const unsigned int blocks_x = static_cast< unsigned int >(
         std::max< int64_t >(1, std::min< int64_t >(blocks_x64, 65535))
      );
      const unsigned int blocks_y = static_cast< unsigned int >(
         std::max< int64_t >(1, std::min< int64_t >(blocks_y64, 65535))
      );
      const dim3 blocks(blocks_x, blocks_y, 1);

      fanout_scatter_backward_cuda_kernel_2d< scalar_t >
         <<<blocks, threads, 0, stream>>>(
            grad_out.data_ptr< scalar_t >(),
            src.data_ptr< int64_t >(),
            dst.data_ptr< int64_t >(),
            num_edges,
            emb,
            grad_x.data_ptr< scalar_t >()
         );
   } else {
      const int64_t total = num_edges * emb;
      const int blocks = grid_for(total);
      fanout_scatter_backward_cuda_kernel_1d< scalar_t >
         <<<blocks, static_cast< int >(kThreads), 0, stream>>>(
            grad_out.data_ptr< scalar_t >(),
            src.data_ptr< int64_t >(),
            dst.data_ptr< int64_t >(),
            num_edges,
            emb,
            grad_x.data_ptr< scalar_t >()
         );
   }
   check_kernel_launch("fanout_scatter_backward_cuda_kernel");
}

template < typename scalar_t >
void launch_fanin_reduce_sum(
   const Tensor& rel,
   const Tensor& src,
   const Tensor& dst,
   int64_t emb,
   Tensor& out,
   cudaStream_t stream
)
{
   const int64_t num_edges = src.size(0);
   const int64_t total = num_edges * emb;
   const int blocks = grid_for(total);
   fanin_reduce_sum_cuda_kernel< scalar_t >
      <<<blocks, static_cast< int >(kThreads), 0, stream>>>(
         rel.data_ptr< scalar_t >(),
         src.data_ptr< int64_t >(),
         dst.data_ptr< int64_t >(),
         num_edges,
         emb,
         out.data_ptr< scalar_t >()
      );
   check_kernel_launch("fanin_reduce_sum_cuda_kernel");
}

template < typename scalar_t >
void launch_fanin_reduce_logsumexp_atomic(
   const Tensor& rel,
   const Tensor& src,
   const Tensor& dst,
   int64_t dim_size,
   int64_t emb,
   Tensor& out,
   Tensor& exp_sums,
   cudaStream_t stream
)
{
   const int64_t num_edges = src.size(0);
   const int64_t edge_total = num_edges * emb;
   const int edge_blocks = grid_for(edge_total);

   fanin_reduce_logsumexp_max_cuda_kernel< scalar_t >
      <<<edge_blocks, static_cast< int >(kThreads), 0, stream>>>(
         rel.data_ptr< scalar_t >(),
         src.data_ptr< int64_t >(),
         dst.data_ptr< int64_t >(),
         num_edges,
         emb,
         out.data_ptr< scalar_t >()
      );
   check_kernel_launch("fanin_reduce_logsumexp_max_cuda_kernel");

   fanin_reduce_logsumexp_sumexp_cuda_kernel< scalar_t >
      <<<edge_blocks, static_cast< int >(kThreads), 0, stream>>>(
         rel.data_ptr< scalar_t >(),
         src.data_ptr< int64_t >(),
         dst.data_ptr< int64_t >(),
         out.data_ptr< scalar_t >(),
         num_edges,
         emb,
         exp_sums.data_ptr< scalar_t >()
      );
   check_kernel_launch("fanin_reduce_logsumexp_sumexp_cuda_kernel");

   const int64_t out_total = dim_size * emb;
   const int out_blocks = grid_for(out_total);
   fanin_reduce_logsumexp_finalize_cuda_kernel< scalar_t >
      <<<out_blocks, static_cast< int >(kThreads), 0, stream>>>(
         exp_sums.data_ptr< scalar_t >(), out_total, out.data_ptr< scalar_t >()
      );
   check_kernel_launch("fanin_reduce_logsumexp_finalize_cuda_kernel");
}

template < typename scalar_t >
void launch_fanin_reduce_logsumexp_segmented(
   const Tensor& rel,
   const Tensor& sorted_src,
   const Tensor& offsets,
   int64_t dim_size,
   int64_t emb,
   Tensor& out,
   cudaStream_t stream
)
{
   const int64_t total = dim_size * emb;
   const int blocks = grid_for(total);
   fanin_reduce_logsumexp_segmented_cuda_kernel< scalar_t >
      <<<blocks, static_cast< int >(kThreads), 0, stream>>>(
         rel.data_ptr< scalar_t >(),
         sorted_src.data_ptr< int64_t >(),
         offsets.data_ptr< int64_t >(),
         dim_size,
         emb,
         out.data_ptr< scalar_t >()
      );
   check_kernel_launch("fanin_reduce_logsumexp_segmented_cuda_kernel");
}

template < typename scalar_t >
void launch_fanin_reduce_sum_backward(
   const Tensor& grad_out,
   const Tensor& src,
   const Tensor& dst,
   int64_t emb,
   Tensor& grad_rel,
   cudaStream_t stream
)
{
   const int64_t num_edges = src.size(0);
   const int64_t total = num_edges * emb;
   const int blocks = grid_for(total);
   fanin_reduce_sum_backward_cuda_kernel< scalar_t >
      <<<blocks, static_cast< int >(kThreads), 0, stream>>>(
         grad_out.data_ptr< scalar_t >(),
         src.data_ptr< int64_t >(),
         dst.data_ptr< int64_t >(),
         num_edges,
         emb,
         grad_rel.data_ptr< scalar_t >()
      );
   check_kernel_launch("fanin_reduce_sum_backward_cuda_kernel");
}

template < typename scalar_t >
void launch_fanin_reduce_logsumexp_backward(
   const Tensor& grad_out,
   const Tensor& rel_flat,
   const Tensor& src,
   const Tensor& dst,
   const Tensor& out,
   int64_t emb,
   Tensor& grad_rel,
   cudaStream_t stream
)
{
   const int64_t num_edges = src.size(0);
   const int64_t total = num_edges * emb;
   const int blocks = grid_for(total);
   fanin_reduce_logsumexp_backward_cuda_kernel< scalar_t >
      <<<blocks, static_cast< int >(kThreads), 0, stream>>>(
         grad_out.data_ptr< scalar_t >(),
         rel_flat.data_ptr< scalar_t >(),
         src.data_ptr< int64_t >(),
         dst.data_ptr< int64_t >(),
         out.data_ptr< scalar_t >(),
         num_edges,
         emb,
         grad_rel.data_ptr< scalar_t >()
      );
   check_kernel_launch("fanin_reduce_logsumexp_backward_cuda_kernel");
}

template < typename FnFloat, typename FnDouble >
void dispatch_float_or_double(
   const Tensor& t,
   FnFloat&& fn_float,
   FnDouble&& fn_double,
   const char* opname
)
{
   const at::ScalarType dtype = dtype_of(t);
   TORCH_CHECK(is_fastpath_dtype(dtype), opname, " supports only float32/float64.");
   if(dtype == at::kFloat) {
      fn_float();
      return;
   }
   fn_double();
}

template < typename scalar_t, typename index_t >
__global__ void fused_two_layer_pointwise_from_indices_cuda_kernel(
   const scalar_t* x_ptr,
   const index_t* relation_args_ptr,
   const int64_t* slot_offsets_ptr,
   const int64_t* row_offsets_ptr,
   const int64_t* out_offsets_ptr,
   int64_t groups,
   int64_t arity,
   int64_t emb,
   const scalar_t* w1_ptr,
   const scalar_t* b1_ptr,
   bool has_b1,
   int64_t hidden,
   const scalar_t* w2_ptr,
   const scalar_t* b2_ptr,
   bool has_b2,
   int64_t pointwise_code,
   scalar_t* out_ptr,
   int64_t* node_idx_ptr
)
{
   const int64_t row = static_cast< int64_t >(blockIdx.x);
   if(row >= row_offsets_ptr[groups]) {
      return;
   }

   int64_t group = 0;
   while(group + 1 < groups && row >= row_offsets_ptr[group + 1]) {
      ++group;
   }
   const int64_t row_in_group = row - row_offsets_ptr[group];
   const int64_t slot_base = slot_offsets_ptr[group] + row_in_group * arity;
   const int64_t out_base = out_offsets_ptr[group] + row_in_group * arity;
   const int64_t in_dim = arity * emb;

   extern __shared__ unsigned char smem_raw[];
   scalar_t* input = reinterpret_cast< scalar_t* >(smem_raw);
   scalar_t* hidden_buf = input + in_dim;

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      const int64_t slot = i / emb;
      const int64_t dim = i - slot * emb;
      const int64_t node = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
      input[i] = x_ptr[node * emb + dim];
   }
   for(int64_t slot = threadIdx.x; slot < arity; slot += blockDim.x) {
      node_idx_ptr[out_base + slot] = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
   }
   __syncthreads();

   const scalar_t* w1_group = w1_ptr + group * hidden * in_dim;
   const scalar_t* b1_group = has_b1 ? (b1_ptr + group * hidden) : nullptr;
   for(int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
      scalar_t acc = has_b1 ? b1_group[h] : static_cast< scalar_t >(0);
      const scalar_t* w_row = w1_group + h * in_dim;
      for(int64_t i = 0; i < in_dim; ++i) {
         acc += w_row[i] * input[i];
      }
      hidden_buf[h] = pointwise_apply_compat(pointwise_code, acc);
   }
   __syncthreads();

   const scalar_t* w2_group = w2_ptr + group * in_dim * hidden;
   const scalar_t* b2_group = has_b2 ? (b2_ptr + group * in_dim) : nullptr;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      scalar_t acc = has_b2 ? b2_group[o] : static_cast< scalar_t >(0);
      const scalar_t* w_row = w2_group + o * hidden;
      for(int64_t h = 0; h < hidden; ++h) {
         acc += w_row[h] * hidden_buf[h];
      }
      const int64_t slot = o / emb;
      const int64_t dim = o - slot * emb;
      out_ptr[(out_base + slot) * emb + dim] = input[o] + acc;
   }
}

template < typename scalar_t, typename index_t >
__global__ void fused_program_two_layer_silu_then_two_layer_silu_from_indices_cuda_kernel(
   const scalar_t* x_ptr,
   const index_t* relation_args_ptr,
   const int64_t* slot_offsets_ptr,
   const int64_t* row_offsets_ptr,
   const int64_t* out_offsets_ptr,
   int64_t groups,
   int64_t arity,
   int64_t emb,
   const scalar_t* w10_ptr,
   const scalar_t* b10_ptr,
   int64_t hidden1,
   const scalar_t* w20_ptr,
   const scalar_t* b20_ptr,
   const scalar_t* w11_ptr,
   const scalar_t* b11_ptr,
   int64_t hidden2,
   const scalar_t* w21_ptr,
   const scalar_t* b21_ptr,
   scalar_t* out_ptr,
   int64_t* node_idx_ptr
)
{
   const int64_t row = static_cast< int64_t >(blockIdx.x);
   if(row >= row_offsets_ptr[groups]) {
      return;
   }

   int64_t group = 0;
   while(group + 1 < groups && row >= row_offsets_ptr[group + 1]) {
      ++group;
   }
   const int64_t row_in_group = row - row_offsets_ptr[group];
   const int64_t slot_base = slot_offsets_ptr[group] + row_in_group * arity;
   const int64_t out_base = out_offsets_ptr[group] + row_in_group * arity;
   const int64_t in_dim = arity * emb;

   extern __shared__ unsigned char smem_raw[];
   scalar_t* input = reinterpret_cast< scalar_t* >(smem_raw);
   scalar_t* hidden1_buf = input + in_dim;
   scalar_t* stage1_buf = hidden1_buf + hidden1;
   scalar_t* hidden2_buf = stage1_buf + in_dim;

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      const int64_t slot = i / emb;
      const int64_t dim = i - slot * emb;
      const int64_t node = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
      input[i] = x_ptr[node * emb + dim];
   }
   for(int64_t slot = threadIdx.x; slot < arity; slot += blockDim.x) {
      node_idx_ptr[out_base + slot] = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
   }
   __syncthreads();

   const scalar_t* w10_group = w10_ptr + group * hidden1 * in_dim;
   const scalar_t* b10_group = b10_ptr + group * hidden1;
   for(int64_t h = threadIdx.x; h < hidden1; h += blockDim.x) {
      scalar_t acc = b10_group[h];
      const scalar_t* w_row = w10_group + h * in_dim;
      for(int64_t i = 0; i < in_dim; ++i) {
         acc += w_row[i] * input[i];
      }
      hidden1_buf[h] = silu_compat(acc);
   }
   __syncthreads();

   const scalar_t* w20_group = w20_ptr + group * in_dim * hidden1;
   const scalar_t* b20_group = b20_ptr + group * in_dim;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      scalar_t acc = b20_group[o];
      const scalar_t* w_row = w20_group + o * hidden1;
      for(int64_t h = 0; h < hidden1; ++h) {
         acc += w_row[h] * hidden1_buf[h];
      }
      stage1_buf[o] = acc;
   }
   __syncthreads();

   const scalar_t* w11_group = w11_ptr + group * hidden2 * in_dim;
   const scalar_t* b11_group = b11_ptr + group * hidden2;
   for(int64_t h = threadIdx.x; h < hidden2; h += blockDim.x) {
      scalar_t acc = b11_group[h];
      const scalar_t* w_row = w11_group + h * in_dim;
      for(int64_t i = 0; i < in_dim; ++i) {
         acc += w_row[i] * stage1_buf[i];
      }
      hidden2_buf[h] = silu_compat(acc);
   }
   __syncthreads();

   const scalar_t* w21_group = w21_ptr + group * in_dim * hidden2;
   const scalar_t* b21_group = b21_ptr + group * in_dim;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      scalar_t acc = b21_group[o];
      const scalar_t* w_row = w21_group + o * hidden2;
      for(int64_t h = 0; h < hidden2; ++h) {
         acc += w_row[h] * hidden2_buf[h];
      }
      const int64_t slot = o / emb;
      const int64_t dim = o - slot * emb;
      out_ptr[(out_base + slot) * emb + dim] = input[o] + acc;
   }
}

template < typename scalar_t, typename index_t >
void launch_fused_two_layer_pointwise_from_indices(
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t arity,
   const Tensor& w1_stack,
   const Tensor& b1_stack,
   const Tensor& w2_stack,
   const Tensor& b2_stack,
   int64_t pointwise_code,
   Tensor& rel_cat,
   Tensor& node_idx,
   cudaStream_t stream
)
{
   const int64_t groups = slot_offsets.size(0);
   if(total_rows <= 0) {
      return;
   }
   const int64_t emb = x.size(1);
   const int64_t hidden = w1_stack.size(1);
   const int64_t in_dim = arity * emb;
   const size_t shared_bytes =
      static_cast< size_t >(in_dim + hidden) * sizeof(scalar_t);
   const dim3 grid(static_cast< unsigned int >(total_rows));
   fused_two_layer_pointwise_from_indices_cuda_kernel< scalar_t, index_t >
      <<<grid, static_cast< int >(kThreads), shared_bytes, stream>>>(
         x.data_ptr< scalar_t >(),
         relation_args.data_ptr< index_t >(),
         slot_offsets.data_ptr< int64_t >(),
         row_offsets.data_ptr< int64_t >(),
         out_offsets.data_ptr< int64_t >(),
         groups,
         arity,
         emb,
         w1_stack.data_ptr< scalar_t >(),
         b1_stack.numel() > 0 ? b1_stack.data_ptr< scalar_t >() : nullptr,
         b1_stack.numel() > 0,
         hidden,
         w2_stack.data_ptr< scalar_t >(),
         b2_stack.numel() > 0 ? b2_stack.data_ptr< scalar_t >() : nullptr,
         b2_stack.numel() > 0,
         pointwise_code,
         rel_cat.data_ptr< scalar_t >(),
         node_idx.data_ptr< int64_t >()
      );
   check_kernel_launch("fused_two_layer_pointwise_from_indices_cuda_kernel");
}

template < typename scalar_t, typename index_t >
void launch_fused_program_two_layer_silu_then_two_layer_silu_from_indices(
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t arity,
   const Tensor& w10_stack,
   const Tensor& b10_stack,
   const Tensor& w20_stack,
   const Tensor& b20_stack,
   const Tensor& w11_stack,
   const Tensor& b11_stack,
   const Tensor& w21_stack,
   const Tensor& b21_stack,
   Tensor& rel_cat,
   Tensor& node_idx,
   cudaStream_t stream
)
{
   const int64_t groups = slot_offsets.size(0);
   if(total_rows <= 0) {
      return;
   }
   const int64_t emb = x.size(1);
   const int64_t hidden1 = w10_stack.size(1);
   const int64_t hidden2 = w11_stack.size(1);
   const int64_t in_dim = arity * emb;
   const size_t shared_bytes =
      static_cast< size_t >(2 * in_dim + hidden1 + hidden2) * sizeof(scalar_t);
   const dim3 grid(static_cast< unsigned int >(total_rows));
   fused_program_two_layer_silu_then_two_layer_silu_from_indices_cuda_kernel< scalar_t, index_t >
      <<<grid, static_cast< int >(kThreads), shared_bytes, stream>>>(
         x.data_ptr< scalar_t >(),
         relation_args.data_ptr< index_t >(),
         slot_offsets.data_ptr< int64_t >(),
         row_offsets.data_ptr< int64_t >(),
         out_offsets.data_ptr< int64_t >(),
         groups,
         arity,
         emb,
         w10_stack.data_ptr< scalar_t >(),
         b10_stack.data_ptr< scalar_t >(),
         hidden1,
         w20_stack.data_ptr< scalar_t >(),
         b20_stack.data_ptr< scalar_t >(),
         w11_stack.data_ptr< scalar_t >(),
         b11_stack.data_ptr< scalar_t >(),
         hidden2,
         w21_stack.data_ptr< scalar_t >(),
         b21_stack.data_ptr< scalar_t >(),
         rel_cat.data_ptr< scalar_t >(),
         node_idx.data_ptr< int64_t >()
      );
   check_kernel_launch("fused_program_two_layer_silu_then_two_layer_silu_from_indices_cuda_kernel");
}

template < typename scalar_t, typename index_t >
__global__ void fused_program_two_layer_silu_then_two_layer_silu_from_indices_backward_cuda_kernel(
   const scalar_t* grad_rel_ptr,
   const scalar_t* x_ptr,
   const index_t* relation_args_ptr,
   const int64_t* slot_offsets_ptr,
   const int64_t* row_offsets_ptr,
   const int64_t* out_offsets_ptr,
   int64_t groups,
   int64_t arity,
   int64_t emb,
   const scalar_t* w10_ptr,
   const scalar_t* b10_ptr,
   int64_t hidden1,
   const scalar_t* w20_ptr,
   const scalar_t* b20_ptr,
   const scalar_t* w11_ptr,
   const scalar_t* b11_ptr,
   int64_t hidden2,
   const scalar_t* w21_ptr,
   const scalar_t* b21_ptr,
   scalar_t* grad_x_ptr,
   scalar_t* grad_w10_ptr,
   scalar_t* grad_b10_ptr,
   scalar_t* grad_w20_ptr,
   scalar_t* grad_b20_ptr,
   scalar_t* grad_w11_ptr,
   scalar_t* grad_b11_ptr,
   scalar_t* grad_w21_ptr,
   scalar_t* grad_b21_ptr
)
{
   const int64_t row = static_cast< int64_t >(blockIdx.x);
   if(row >= row_offsets_ptr[groups]) {
      return;
   }

   int64_t group = 0;
   while(group + 1 < groups && row >= row_offsets_ptr[group + 1]) {
      ++group;
   }
   const int64_t row_in_group = row - row_offsets_ptr[group];
   const int64_t slot_base = slot_offsets_ptr[group] + row_in_group * arity;
   const int64_t out_base = out_offsets_ptr[group] + row_in_group * arity;
   const int64_t in_dim = arity * emb;

   extern __shared__ unsigned char smem_raw[];
   scalar_t* input = reinterpret_cast< scalar_t* >(smem_raw);
   scalar_t* z1 = input + in_dim;
   scalar_t* hidden1_buf = z1 + hidden1;
   scalar_t* stage1_buf = hidden1_buf + hidden1;
   scalar_t* z2 = stage1_buf + in_dim;
   scalar_t* hidden2_buf = z2 + hidden2;
   scalar_t* grad_out = hidden2_buf + hidden2;
   scalar_t* grad_stage1 = grad_out + in_dim;
   scalar_t* grad_pre1 = grad_stage1 + in_dim;

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      const int64_t slot = i / emb;
      const int64_t dim = i - slot * emb;
      const int64_t node = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
      input[i] = x_ptr[node * emb + dim];
      grad_out[i] = grad_rel_ptr[(out_base + slot) * emb + dim];
      grad_stage1[i] = static_cast< scalar_t >(0);
   }
   __syncthreads();

   const scalar_t* w10_group = w10_ptr + group * hidden1 * in_dim;
   const scalar_t* b10_group = b10_ptr + group * hidden1;
   for(int64_t h = threadIdx.x; h < hidden1; h += blockDim.x) {
      scalar_t acc = b10_group[h];
      const scalar_t* w_row = w10_group + h * in_dim;
      for(int64_t i = 0; i < in_dim; ++i) {
         acc += w_row[i] * input[i];
      }
      z1[h] = acc;
      hidden1_buf[h] = silu_compat(acc);
   }
   __syncthreads();

   const scalar_t* w20_group = w20_ptr + group * in_dim * hidden1;
   const scalar_t* b20_group = b20_ptr + group * in_dim;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      scalar_t acc = b20_group[o];
      const scalar_t* w_row = w20_group + o * hidden1;
      for(int64_t h = 0; h < hidden1; ++h) {
         acc += w_row[h] * hidden1_buf[h];
      }
      stage1_buf[o] = acc;
   }
   __syncthreads();

   const scalar_t* w11_group = w11_ptr + group * hidden2 * in_dim;
   const scalar_t* b11_group = b11_ptr + group * hidden2;
   for(int64_t h = threadIdx.x; h < hidden2; h += blockDim.x) {
      scalar_t acc = b11_group[h];
      const scalar_t* w_row = w11_group + h * in_dim;
      for(int64_t i = 0; i < in_dim; ++i) {
         acc += w_row[i] * stage1_buf[i];
      }
      z2[h] = acc;
      hidden2_buf[h] = silu_compat(acc);
   }
   __syncthreads();

   const scalar_t* w21_group = w21_ptr + group * in_dim * hidden2;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      const scalar_t go = grad_out[o];
      atomic_add_compat(grad_b21_ptr + group * in_dim + o, go);
      const scalar_t* w_row = w21_group + o * hidden2;
      for(int64_t h = 0; h < hidden2; ++h) {
         atomic_add_compat(grad_w21_ptr + (group * in_dim + o) * hidden2 + h, go * hidden2_buf[h]);
      }
   }
   __syncthreads();

   for(int64_t h = threadIdx.x; h < hidden2; h += blockDim.x) {
      scalar_t grad_h2 = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         grad_h2 += w21_group[o * hidden2 + h] * grad_out[o];
      }
      const scalar_t grad_pre2 = grad_h2 * silu_grad_from_pre_activation_compat(z2[h]);
      atomic_add_compat(grad_b11_ptr + group * hidden2 + h, grad_pre2);
      const scalar_t* stage1_ptr = stage1_buf;
      for(int64_t i = 0; i < in_dim; ++i) {
         atomic_add_compat(grad_w11_ptr + (group * hidden2 + h) * in_dim + i, grad_pre2 * stage1_ptr[i]);
         atomic_add_compat(grad_stage1 + i, w11_group[h * in_dim + i] * grad_pre2);
      }
   }
   __syncthreads();

   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      const scalar_t gs1 = grad_stage1[o];
      atomic_add_compat(grad_b20_ptr + group * in_dim + o, gs1);
      const scalar_t* w_row = w20_group + o * hidden1;
      for(int64_t h = 0; h < hidden1; ++h) {
         atomic_add_compat(grad_w20_ptr + (group * in_dim + o) * hidden1 + h, gs1 * hidden1_buf[h]);
      }
   }
   __syncthreads();

   for(int64_t h = threadIdx.x; h < hidden1; h += blockDim.x) {
      scalar_t grad_h1 = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         grad_h1 += w20_group[o * hidden1 + h] * grad_stage1[o];
      }
      const scalar_t gp1 = grad_h1 * silu_grad_from_pre_activation_compat(z1[h]);
      grad_pre1[h] = gp1;
      atomic_add_compat(grad_b10_ptr + group * hidden1 + h, gp1);
      const scalar_t* input_ptr = input;
      for(int64_t i = 0; i < in_dim; ++i) {
         atomic_add_compat(grad_w10_ptr + (group * hidden1 + h) * in_dim + i, gp1 * input_ptr[i]);
      }
   }
   __syncthreads();

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      scalar_t grad_input = grad_out[i];
      for(int64_t h = 0; h < hidden1; ++h) {
         grad_input += w10_group[h * in_dim + i] * grad_pre1[h];
      }
      const int64_t slot = i / emb;
      const int64_t dim = i - slot * emb;
      const int64_t node = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
      atomic_add_compat(grad_x_ptr + node * emb + dim, grad_input);
   }
}

template < typename scalar_t, typename index_t >
void launch_fused_program_two_layer_silu_then_two_layer_silu_from_indices_backward(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t arity,
   const Tensor& w10_stack,
   const Tensor& b10_stack,
   const Tensor& w20_stack,
   const Tensor& b20_stack,
   const Tensor& w11_stack,
   const Tensor& b11_stack,
   const Tensor& w21_stack,
   const Tensor& b21_stack,
   Tensor& grad_x,
   Tensor& grad_w10,
   Tensor& grad_b10,
   Tensor& grad_w20,
   Tensor& grad_b20,
   Tensor& grad_w11,
   Tensor& grad_b11,
   Tensor& grad_w21,
   Tensor& grad_b21,
   cudaStream_t stream
)
{
   const int64_t groups = slot_offsets.size(0);
   if(total_rows <= 0) {
      return;
   }
   const int64_t emb = x.size(1);
   const int64_t hidden1 = w10_stack.size(1);
   const int64_t hidden2 = w11_stack.size(1);
   const int64_t in_dim = arity * emb;
   const size_t shared_bytes =
      static_cast< size_t >(4 * in_dim + 3 * hidden1 + 2 * hidden2) * sizeof(scalar_t);
   const dim3 grid(static_cast< unsigned int >(total_rows));
   fused_program_two_layer_silu_then_two_layer_silu_from_indices_backward_cuda_kernel< scalar_t, index_t >
      <<<grid, static_cast< int >(kThreads), shared_bytes, stream>>>(
         grad_rel.data_ptr< scalar_t >(),
         x.data_ptr< scalar_t >(),
         relation_args.data_ptr< index_t >(),
         slot_offsets.data_ptr< int64_t >(),
         row_offsets.data_ptr< int64_t >(),
         out_offsets.data_ptr< int64_t >(),
         groups,
         arity,
         emb,
         w10_stack.data_ptr< scalar_t >(),
         b10_stack.data_ptr< scalar_t >(),
         hidden1,
         w20_stack.data_ptr< scalar_t >(),
         b20_stack.data_ptr< scalar_t >(),
         w11_stack.data_ptr< scalar_t >(),
         b11_stack.data_ptr< scalar_t >(),
         hidden2,
         w21_stack.data_ptr< scalar_t >(),
         b21_stack.data_ptr< scalar_t >(),
         grad_x.data_ptr< scalar_t >(),
         grad_w10.data_ptr< scalar_t >(),
         grad_b10.data_ptr< scalar_t >(),
         grad_w20.data_ptr< scalar_t >(),
         grad_b20.data_ptr< scalar_t >(),
         grad_w11.data_ptr< scalar_t >(),
         grad_b11.data_ptr< scalar_t >(),
         grad_w21.data_ptr< scalar_t >(),
         grad_b21.data_ptr< scalar_t >()
      );
   check_kernel_launch(
      "fused_program_two_layer_silu_then_two_layer_silu_from_indices_backward_cuda_kernel"
   );
}

template < typename scalar_t, typename index_t >
__global__ void fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_cuda_kernel(
   const scalar_t* x_ptr,
   const index_t* relation_args_ptr,
   const int64_t* slot_offsets_ptr,
   const int64_t* row_offsets_ptr,
   const int64_t* out_offsets_ptr,
   int64_t groups,
   int64_t arity,
   int64_t emb,
   const scalar_t* w10_ptr,
   const scalar_t* b10_ptr,
   int64_t hidden1,
   const scalar_t* w20_ptr,
   const scalar_t* b20_ptr,
   const scalar_t* w11_ptr,
   const scalar_t* b11_ptr,
   int64_t hidden2,
   const scalar_t* w21_ptr,
   const scalar_t* b21_ptr,
   const scalar_t* ln_weight_ptr,
   const scalar_t* ln_bias_ptr,
   bool has_ln_weight,
   bool has_ln_bias,
   scalar_t ln_eps,
   scalar_t* out_ptr,
   int64_t* node_idx_ptr
)
{
   const int64_t row = static_cast< int64_t >(blockIdx.x);
   if(row >= row_offsets_ptr[groups]) {
      return;
   }

   int64_t group = 0;
   while(group + 1 < groups && row >= row_offsets_ptr[group + 1]) {
      ++group;
   }
   const int64_t row_in_group = row - row_offsets_ptr[group];
   const int64_t slot_base = slot_offsets_ptr[group] + row_in_group * arity;
   const int64_t out_base = out_offsets_ptr[group] + row_in_group * arity;
   const int64_t in_dim = arity * emb;

   extern __shared__ unsigned char smem_raw[];
   scalar_t* input = reinterpret_cast< scalar_t* >(smem_raw);
   scalar_t* hidden1_buf = input + in_dim;
   scalar_t* stage1_buf = hidden1_buf + hidden1;
   scalar_t* hidden2_buf = stage1_buf + in_dim;
   scalar_t* norm_buf = hidden2_buf + hidden2;
   __shared__ scalar_t mean_val;
   __shared__ scalar_t invstd_val;

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      const int64_t slot = i / emb;
      const int64_t dim = i - slot * emb;
      const int64_t node = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
      input[i] = x_ptr[node * emb + dim];
   }
   for(int64_t slot = threadIdx.x; slot < arity; slot += blockDim.x) {
      node_idx_ptr[out_base + slot] = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
   }
   __syncthreads();

   const scalar_t* w10_group = w10_ptr + group * hidden1 * in_dim;
   const scalar_t* b10_group = b10_ptr + group * hidden1;
   for(int64_t h = threadIdx.x; h < hidden1; h += blockDim.x) {
      scalar_t acc = b10_group[h];
      const scalar_t* w_row = w10_group + h * in_dim;
      for(int64_t i = 0; i < in_dim; ++i) {
         acc += w_row[i] * input[i];
      }
      hidden1_buf[h] = silu_compat(acc);
   }
   __syncthreads();

   const scalar_t* w20_group = w20_ptr + group * in_dim * hidden1;
   const scalar_t* b20_group = b20_ptr + group * in_dim;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      scalar_t acc = b20_group[o];
      const scalar_t* w_row = w20_group + o * hidden1;
      for(int64_t h = 0; h < hidden1; ++h) {
         acc += w_row[h] * hidden1_buf[h];
      }
      stage1_buf[o] = acc;
   }
   __syncthreads();

   const scalar_t* w11_group = w11_ptr + group * hidden2 * in_dim;
   const scalar_t* b11_group = b11_ptr + group * hidden2;
   for(int64_t h = threadIdx.x; h < hidden2; h += blockDim.x) {
      scalar_t acc = b11_group[h];
      const scalar_t* w_row = w11_group + h * in_dim;
      for(int64_t i = 0; i < in_dim; ++i) {
         acc += w_row[i] * stage1_buf[i];
      }
      hidden2_buf[h] = silu_compat(acc);
   }
   __syncthreads();

   const scalar_t* w21_group = w21_ptr + group * in_dim * hidden2;
   const scalar_t* b21_group = b21_ptr + group * in_dim;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      scalar_t acc = b21_group[o];
      const scalar_t* w_row = w21_group + o * hidden2;
      for(int64_t h = 0; h < hidden2; ++h) {
         acc += w_row[h] * hidden2_buf[h];
      }
      norm_buf[o] = acc;
   }
   __syncthreads();

   if(threadIdx.x == 0) {
      scalar_t mean = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         mean += norm_buf[o];
      }
      mean /= static_cast< scalar_t >(in_dim);
      scalar_t var = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         const scalar_t centered = norm_buf[o] - mean;
         var += centered * centered;
      }
      var /= static_cast< scalar_t >(in_dim);
      mean_val = mean;
      invstd_val = static_cast< scalar_t >(1) / sqrt(var + ln_eps);
   }
   __syncthreads();

   const scalar_t* ln_weight_group = has_ln_weight ? (ln_weight_ptr + group * in_dim) : nullptr;
   const scalar_t* ln_bias_group = has_ln_bias ? (ln_bias_ptr + group * in_dim) : nullptr;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      scalar_t xhat = (norm_buf[o] - mean_val) * invstd_val;
      if(has_ln_weight) {
         xhat *= ln_weight_group[o];
      }
      if(has_ln_bias) {
         xhat += ln_bias_group[o];
      }
      const int64_t slot = o / emb;
      const int64_t dim = o - slot * emb;
      out_ptr[(out_base + slot) * emb + dim] = input[o] + xhat;
   }
}

template < typename scalar_t, typename index_t >
void launch_fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices(
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t arity,
   const Tensor& w10_stack,
   const Tensor& b10_stack,
   const Tensor& w20_stack,
   const Tensor& b20_stack,
   const Tensor& w11_stack,
   const Tensor& b11_stack,
   const Tensor& w21_stack,
   const Tensor& b21_stack,
   const Tensor& ln_weight_stack,
   const Tensor& ln_bias_stack,
   scalar_t ln_eps,
   Tensor& rel_cat,
   Tensor& node_idx,
   cudaStream_t stream
)
{
   const int64_t groups = slot_offsets.size(0);
   if(total_rows <= 0) {
      return;
   }
   const int64_t emb = x.size(1);
   const int64_t hidden1 = w10_stack.size(1);
   const int64_t hidden2 = w11_stack.size(1);
   const int64_t in_dim = arity * emb;
   const size_t shared_bytes =
      static_cast< size_t >(3 * in_dim + hidden1 + hidden2) * sizeof(scalar_t);
   const dim3 grid(static_cast< unsigned int >(total_rows));
   fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_cuda_kernel< scalar_t, index_t >
      <<<grid, static_cast< int >(kThreads), shared_bytes, stream>>>(
         x.data_ptr< scalar_t >(),
         relation_args.data_ptr< index_t >(),
         slot_offsets.data_ptr< int64_t >(),
         row_offsets.data_ptr< int64_t >(),
         out_offsets.data_ptr< int64_t >(),
         groups,
         arity,
         emb,
         w10_stack.data_ptr< scalar_t >(),
         b10_stack.data_ptr< scalar_t >(),
         hidden1,
         w20_stack.data_ptr< scalar_t >(),
         b20_stack.data_ptr< scalar_t >(),
         w11_stack.data_ptr< scalar_t >(),
         b11_stack.data_ptr< scalar_t >(),
         hidden2,
         w21_stack.data_ptr< scalar_t >(),
         b21_stack.data_ptr< scalar_t >(),
         ln_weight_stack.numel() > 0 ? ln_weight_stack.data_ptr< scalar_t >() : nullptr,
         ln_bias_stack.numel() > 0 ? ln_bias_stack.data_ptr< scalar_t >() : nullptr,
         ln_weight_stack.numel() > 0,
         ln_bias_stack.numel() > 0,
         ln_eps,
         rel_cat.data_ptr< scalar_t >(),
         node_idx.data_ptr< int64_t >()
      );
   check_kernel_launch(
      "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_cuda_kernel"
   );
}

template < typename scalar_t, typename index_t >
__global__ void fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_backward_cuda_kernel(
   const scalar_t* grad_rel_ptr,
   const scalar_t* x_ptr,
   const index_t* relation_args_ptr,
   const int64_t* slot_offsets_ptr,
   const int64_t* row_offsets_ptr,
   const int64_t* out_offsets_ptr,
   int64_t groups,
   int64_t arity,
   int64_t emb,
   const scalar_t* w10_ptr,
   const scalar_t* b10_ptr,
   int64_t hidden1,
   const scalar_t* w20_ptr,
   const scalar_t* b20_ptr,
   const scalar_t* w11_ptr,
   const scalar_t* b11_ptr,
   int64_t hidden2,
   const scalar_t* w21_ptr,
   const scalar_t* b21_ptr,
   const scalar_t* ln_weight_ptr,
   const scalar_t* ln_bias_ptr,
   bool has_ln_weight,
   bool has_ln_bias,
   scalar_t ln_eps,
   scalar_t* grad_x_ptr,
   scalar_t* grad_w10_ptr,
   scalar_t* grad_b10_ptr,
   scalar_t* grad_w20_ptr,
   scalar_t* grad_b20_ptr,
   scalar_t* grad_w11_ptr,
   scalar_t* grad_b11_ptr,
   scalar_t* grad_w21_ptr,
   scalar_t* grad_b21_ptr,
   scalar_t* grad_ln_weight_ptr,
   scalar_t* grad_ln_bias_ptr
)
{
   const int64_t row = static_cast< int64_t >(blockIdx.x);
   if(row >= row_offsets_ptr[groups]) {
      return;
   }

   int64_t group = 0;
   while(group + 1 < groups && row >= row_offsets_ptr[group + 1]) {
      ++group;
   }
   const int64_t row_in_group = row - row_offsets_ptr[group];
   const int64_t slot_base = slot_offsets_ptr[group] + row_in_group * arity;
   const int64_t out_base = out_offsets_ptr[group] + row_in_group * arity;
   const int64_t in_dim = arity * emb;

   extern __shared__ unsigned char smem_raw[];
   scalar_t* input = reinterpret_cast< scalar_t* >(smem_raw);
   scalar_t* z1 = input + in_dim;
   scalar_t* hidden1_buf = z1 + hidden1;
   scalar_t* stage1_buf = hidden1_buf + hidden1;
   scalar_t* z2 = stage1_buf + in_dim;
   scalar_t* hidden2_buf = z2 + hidden2;
   scalar_t* xhat_buf = hidden2_buf + hidden2;
   scalar_t* grad_z2 = xhat_buf + in_dim;
   scalar_t* grad_stage1 = grad_z2 + in_dim;
   __shared__ scalar_t mean_val;
   __shared__ scalar_t invstd_val;
   __shared__ scalar_t mean_dxhat;
   __shared__ scalar_t mean_dxhat_xhat;

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      const int64_t slot = i / emb;
      const int64_t dim = i - slot * emb;
      const int64_t node = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
      input[i] = x_ptr[node * emb + dim];
      grad_stage1[i] = static_cast< scalar_t >(0);
   }
   __syncthreads();

   const scalar_t* w10_group = w10_ptr + group * hidden1 * in_dim;
   const scalar_t* b10_group = b10_ptr + group * hidden1;
   for(int64_t h = threadIdx.x; h < hidden1; h += blockDim.x) {
      scalar_t acc = b10_group[h];
      const scalar_t* w_row = w10_group + h * in_dim;
      for(int64_t i = 0; i < in_dim; ++i) {
         acc += w_row[i] * input[i];
      }
      z1[h] = acc;
      hidden1_buf[h] = silu_compat(acc);
   }
   __syncthreads();

   const scalar_t* w20_group = w20_ptr + group * in_dim * hidden1;
   const scalar_t* b20_group = b20_ptr + group * in_dim;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      scalar_t acc = b20_group[o];
      const scalar_t* w_row = w20_group + o * hidden1;
      for(int64_t h = 0; h < hidden1; ++h) {
         acc += w_row[h] * hidden1_buf[h];
      }
      stage1_buf[o] = acc;
   }
   __syncthreads();

   const scalar_t* w11_group = w11_ptr + group * hidden2 * in_dim;
   const scalar_t* b11_group = b11_ptr + group * hidden2;
   for(int64_t h = threadIdx.x; h < hidden2; h += blockDim.x) {
      scalar_t acc = b11_group[h];
      const scalar_t* w_row = w11_group + h * in_dim;
      for(int64_t i = 0; i < in_dim; ++i) {
         acc += w_row[i] * stage1_buf[i];
      }
      z2[h] = acc;
      hidden2_buf[h] = silu_compat(acc);
   }
   __syncthreads();

   const scalar_t* w21_group = w21_ptr + group * in_dim * hidden2;
   const scalar_t* b21_group = b21_ptr + group * in_dim;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      scalar_t acc = b21_group[o];
      const scalar_t* w_row = w21_group + o * hidden2;
      for(int64_t h = 0; h < hidden2; ++h) {
         acc += w_row[h] * hidden2_buf[h];
      }
      xhat_buf[o] = acc;
      grad_z2[o] = static_cast< scalar_t >(0);
   }
   __syncthreads();

   if(threadIdx.x == 0) {
      scalar_t mean = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         mean += xhat_buf[o];
      }
      mean /= static_cast< scalar_t >(in_dim);
      scalar_t var = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         const scalar_t centered = xhat_buf[o] - mean;
         var += centered * centered;
      }
      var /= static_cast< scalar_t >(in_dim);
      mean_val = mean;
      invstd_val = static_cast< scalar_t >(1) / sqrt(var + ln_eps);
   }
   __syncthreads();

   const scalar_t* ln_weight_group = has_ln_weight ? (ln_weight_ptr + group * in_dim) : nullptr;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      const scalar_t normed = (xhat_buf[o] - mean_val) * invstd_val;
      xhat_buf[o] = normed;
      const int64_t slot = o / emb;
      const int64_t dim = o - slot * emb;
      const scalar_t grad_y = grad_rel_ptr[(out_base + slot) * emb + dim];
      if(grad_ln_bias_ptr != nullptr) {
         atomic_add_compat(grad_ln_bias_ptr + group * in_dim + o, grad_y);
      }
      if(grad_ln_weight_ptr != nullptr) {
         atomic_add_compat(grad_ln_weight_ptr + group * in_dim + o, grad_y * normed);
      }
      grad_z2[o] = has_ln_weight ? (grad_y * ln_weight_group[o]) : grad_y;
   }
   __syncthreads();

   if(threadIdx.x == 0) {
      scalar_t sum_dxhat = static_cast< scalar_t >(0);
      scalar_t sum_dxhat_xhat = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         sum_dxhat += grad_z2[o];
         sum_dxhat_xhat += grad_z2[o] * xhat_buf[o];
      }
      mean_dxhat = sum_dxhat / static_cast< scalar_t >(in_dim);
      mean_dxhat_xhat = sum_dxhat_xhat / static_cast< scalar_t >(in_dim);
   }
   __syncthreads();

   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      grad_z2[o] =
         invstd_val * (grad_z2[o] - mean_dxhat - xhat_buf[o] * mean_dxhat_xhat);
      atomic_add_compat(grad_b21_ptr + group * in_dim + o, grad_z2[o]);
      const scalar_t go = grad_z2[o];
      for(int64_t h = 0; h < hidden2; ++h) {
         atomic_add_compat(
            grad_w21_ptr + (group * in_dim + o) * hidden2 + h,
            go * hidden2_buf[h]
         );
      }
   }
   __syncthreads();

   for(int64_t h = threadIdx.x; h < hidden2; h += blockDim.x) {
      scalar_t grad_h2 = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         grad_h2 += w21_group[o * hidden2 + h] * grad_z2[o];
      }
      const scalar_t grad_pre2 = grad_h2 * silu_grad_from_pre_activation_compat(z2[h]);
      atomic_add_compat(grad_b11_ptr + group * hidden2 + h, grad_pre2);
      for(int64_t i = 0; i < in_dim; ++i) {
         atomic_add_compat(
            grad_w11_ptr + (group * hidden2 + h) * in_dim + i,
            grad_pre2 * stage1_buf[i]
         );
         atomic_add_compat(grad_stage1 + i, w11_group[h * in_dim + i] * grad_pre2);
      }
   }
   __syncthreads();

   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      const scalar_t gs1 = grad_stage1[o];
      atomic_add_compat(grad_b20_ptr + group * in_dim + o, gs1);
      for(int64_t h = 0; h < hidden1; ++h) {
         atomic_add_compat(
            grad_w20_ptr + (group * in_dim + o) * hidden1 + h,
            gs1 * hidden1_buf[h]
         );
      }
   }
   __syncthreads();

   for(int64_t h = threadIdx.x; h < hidden1; h += blockDim.x) {
      scalar_t grad_h1 = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         grad_h1 += w20_group[o * hidden1 + h] * grad_stage1[o];
      }
      const scalar_t gp1 = grad_h1 * silu_grad_from_pre_activation_compat(z1[h]);
      z1[h] = gp1;
      atomic_add_compat(grad_b10_ptr + group * hidden1 + h, gp1);
      for(int64_t i = 0; i < in_dim; ++i) {
         atomic_add_compat(
            grad_w10_ptr + (group * hidden1 + h) * in_dim + i,
            gp1 * input[i]
         );
      }
   }
   __syncthreads();

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      scalar_t grad_input = static_cast< scalar_t >(0);
      for(int64_t h = 0; h < hidden1; ++h) {
         grad_input += w10_group[h * in_dim + i] * z1[h];
      }
      const int64_t slot = i / emb;
      const int64_t dim = i - slot * emb;
      const scalar_t grad_residual = grad_rel_ptr[(out_base + slot) * emb + dim];
      const int64_t node = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
      atomic_add_compat(grad_x_ptr + node * emb + dim, grad_input + grad_residual);
   }
}

template < typename scalar_t, typename index_t >
void launch_fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_backward(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t arity,
   const Tensor& w10_stack,
   const Tensor& b10_stack,
   const Tensor& w20_stack,
   const Tensor& b20_stack,
   const Tensor& w11_stack,
   const Tensor& b11_stack,
   const Tensor& w21_stack,
   const Tensor& b21_stack,
   const Tensor& ln_weight_stack,
   const Tensor& ln_bias_stack,
   scalar_t ln_eps,
   Tensor& grad_x,
   Tensor& grad_w10,
   Tensor& grad_b10,
   Tensor& grad_w20,
   Tensor& grad_b20,
   Tensor& grad_w11,
   Tensor& grad_b11,
   Tensor& grad_w21,
   Tensor& grad_b21,
   Tensor& grad_ln_weight,
   Tensor& grad_ln_bias,
   cudaStream_t stream
)
{
   const int64_t groups = slot_offsets.size(0);
   if(total_rows <= 0) {
      return;
   }
   const int64_t emb = x.size(1);
   const int64_t hidden1 = w10_stack.size(1);
   const int64_t hidden2 = w11_stack.size(1);
   const int64_t in_dim = arity * emb;
   const size_t shared_bytes =
      static_cast< size_t >(5 * in_dim + 2 * hidden1 + 2 * hidden2) * sizeof(scalar_t);
   const dim3 grid(static_cast< unsigned int >(total_rows));
   fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_backward_cuda_kernel< scalar_t, index_t >
      <<<grid, static_cast< int >(kThreads), shared_bytes, stream>>>(
         grad_rel.data_ptr< scalar_t >(),
         x.data_ptr< scalar_t >(),
         relation_args.data_ptr< index_t >(),
         slot_offsets.data_ptr< int64_t >(),
         row_offsets.data_ptr< int64_t >(),
         out_offsets.data_ptr< int64_t >(),
         groups,
         arity,
         emb,
         w10_stack.data_ptr< scalar_t >(),
         b10_stack.data_ptr< scalar_t >(),
         hidden1,
         w20_stack.data_ptr< scalar_t >(),
         b20_stack.data_ptr< scalar_t >(),
         w11_stack.data_ptr< scalar_t >(),
         b11_stack.data_ptr< scalar_t >(),
         hidden2,
         w21_stack.data_ptr< scalar_t >(),
         b21_stack.data_ptr< scalar_t >(),
         ln_weight_stack.numel() > 0 ? ln_weight_stack.data_ptr< scalar_t >() : nullptr,
         ln_bias_stack.numel() > 0 ? ln_bias_stack.data_ptr< scalar_t >() : nullptr,
         ln_weight_stack.numel() > 0,
         ln_bias_stack.numel() > 0,
         ln_eps,
         grad_x.data_ptr< scalar_t >(),
         grad_w10.data_ptr< scalar_t >(),
         grad_b10.data_ptr< scalar_t >(),
         grad_w20.data_ptr< scalar_t >(),
         grad_b20.data_ptr< scalar_t >(),
         grad_w11.data_ptr< scalar_t >(),
         grad_b11.data_ptr< scalar_t >(),
         grad_w21.data_ptr< scalar_t >(),
         grad_b21.data_ptr< scalar_t >(),
         grad_ln_weight.numel() > 0 ? grad_ln_weight.data_ptr< scalar_t >() : nullptr,
         grad_ln_bias.numel() > 0 ? grad_ln_bias.data_ptr< scalar_t >() : nullptr
      );
   check_kernel_launch(
      "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_backward_cuda_kernel"
   );
}

template < typename scalar_t, typename index_t >
__global__ void fused_two_layer_pointwise_from_indices_backward_cuda_kernel(
   const scalar_t* grad_rel_ptr,
   const scalar_t* x_ptr,
   const index_t* relation_args_ptr,
   const int64_t* slot_offsets_ptr,
   const int64_t* row_offsets_ptr,
   const int64_t* out_offsets_ptr,
   int64_t groups,
   int64_t arity,
   int64_t emb,
   const scalar_t* w1_ptr,
   const scalar_t* b1_ptr,
   bool has_b1,
   int64_t hidden,
   const scalar_t* w2_ptr,
   const scalar_t* b2_ptr,
   bool has_b2,
   int64_t pointwise_code,
   scalar_t* grad_x_ptr,
   scalar_t* grad_w1_ptr,
   scalar_t* grad_b1_ptr,
   scalar_t* grad_w2_ptr,
   scalar_t* grad_b2_ptr
)
{
   const int64_t row = static_cast< int64_t >(blockIdx.x);
   if(row >= row_offsets_ptr[groups]) {
      return;
   }

   int64_t group = 0;
   while(group + 1 < groups && row >= row_offsets_ptr[group + 1]) {
      ++group;
   }
   const int64_t row_in_group = row - row_offsets_ptr[group];
   const int64_t slot_base = slot_offsets_ptr[group] + row_in_group * arity;
   const int64_t out_base = out_offsets_ptr[group] + row_in_group * arity;
   const int64_t in_dim = arity * emb;

   extern __shared__ unsigned char smem_raw[];
   scalar_t* input = reinterpret_cast< scalar_t* >(smem_raw);
   scalar_t* z1 = input + in_dim;
   scalar_t* hidden_buf = z1 + hidden;
   scalar_t* grad_out = hidden_buf + hidden;
   scalar_t* grad_z1 = grad_out + in_dim;

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      const int64_t slot = i / emb;
      const int64_t dim = i - slot * emb;
      const int64_t node = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
      input[i] = x_ptr[node * emb + dim];
      grad_out[i] = grad_rel_ptr[(out_base + slot) * emb + dim];
   }
   __syncthreads();

   const scalar_t* w1_group = w1_ptr + group * hidden * in_dim;
   const scalar_t* b1_group = has_b1 ? (b1_ptr + group * hidden) : nullptr;
   for(int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
      scalar_t acc = has_b1 ? b1_group[h] : static_cast< scalar_t >(0);
      const scalar_t* w_row = w1_group + h * in_dim;
      for(int64_t i = 0; i < in_dim; ++i) {
         acc += w_row[i] * input[i];
      }
      z1[h] = acc;
      hidden_buf[h] = pointwise_apply_compat(pointwise_code, acc);
   }
   __syncthreads();

   const scalar_t* w2_group = w2_ptr + group * in_dim * hidden;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      if(grad_b2_ptr != nullptr) {
         atomic_add_compat(grad_b2_ptr + group * in_dim + o, grad_out[o]);
      }
      const scalar_t go = grad_out[o];
      for(int64_t h = 0; h < hidden; ++h) {
         atomic_add_compat(
            grad_w2_ptr + (group * in_dim + o) * hidden + h,
            go * hidden_buf[h]
         );
      }
   }
   __syncthreads();

   for(int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
      scalar_t grad_h = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         grad_h += w2_group[o * hidden + h] * grad_out[o];
      }
      const scalar_t gz1 = grad_h * pointwise_grad_from_pre_activation_compat(pointwise_code, z1[h]);
      grad_z1[h] = gz1;
      if(grad_b1_ptr != nullptr) {
         atomic_add_compat(grad_b1_ptr + group * hidden + h, gz1);
      }
      const scalar_t* input_ptr = input;
      for(int64_t i = 0; i < in_dim; ++i) {
         atomic_add_compat(
            grad_w1_ptr + (group * hidden + h) * in_dim + i,
            gz1 * input_ptr[i]
         );
      }
   }
   __syncthreads();

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      scalar_t grad_input = grad_out[i];
      for(int64_t h = 0; h < hidden; ++h) {
         grad_input += w1_group[h * in_dim + i] * grad_z1[h];
      }
      const int64_t slot = i / emb;
      const int64_t dim = i - slot * emb;
      const int64_t node = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
      atomic_add_compat(grad_x_ptr + node * emb + dim, grad_input);
   }
}

template < typename scalar_t, typename index_t >
void launch_fused_two_layer_pointwise_from_indices_backward(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t arity,
   const Tensor& w1_stack,
   const Tensor& b1_stack,
   const Tensor& w2_stack,
   const Tensor& b2_stack,
   int64_t pointwise_code,
   Tensor& grad_x,
   Tensor& grad_w1,
   Tensor& grad_b1,
   Tensor& grad_w2,
   Tensor& grad_b2,
   cudaStream_t stream
)
{
   const int64_t groups = slot_offsets.size(0);
   if(total_rows <= 0) {
      return;
   }
   const int64_t emb = x.size(1);
   const int64_t hidden = w1_stack.size(1);
   const int64_t in_dim = arity * emb;
   const size_t shared_bytes =
      static_cast< size_t >(2 * in_dim + 3 * hidden) * sizeof(scalar_t);
   const dim3 grid(static_cast< unsigned int >(total_rows));
   fused_two_layer_pointwise_from_indices_backward_cuda_kernel< scalar_t, index_t >
      <<<grid, static_cast< int >(kThreads), shared_bytes, stream>>>(
         grad_rel.data_ptr< scalar_t >(),
         x.data_ptr< scalar_t >(),
         relation_args.data_ptr< index_t >(),
         slot_offsets.data_ptr< int64_t >(),
         row_offsets.data_ptr< int64_t >(),
         out_offsets.data_ptr< int64_t >(),
         groups,
         arity,
         emb,
         w1_stack.data_ptr< scalar_t >(),
         b1_stack.numel() > 0 ? b1_stack.data_ptr< scalar_t >() : nullptr,
         b1_stack.numel() > 0,
         hidden,
         w2_stack.data_ptr< scalar_t >(),
         b2_stack.numel() > 0 ? b2_stack.data_ptr< scalar_t >() : nullptr,
         b2_stack.numel() > 0,
         pointwise_code,
         grad_x.data_ptr< scalar_t >(),
         grad_w1.data_ptr< scalar_t >(),
         grad_b1.numel() > 0 ? grad_b1.data_ptr< scalar_t >() : nullptr,
         grad_w2.data_ptr< scalar_t >(),
         grad_b2.numel() > 0 ? grad_b2.data_ptr< scalar_t >() : nullptr
      );
   check_kernel_launch("fused_two_layer_pointwise_from_indices_backward_cuda_kernel");
}

}  // namespace

Tensor fanout_scatter_cuda(
   const Tensor& x_cat,
   const Tensor& src_global_idx,
   const Tensor& flat_dst,
   int64_t out_rows
)
{
   c10::cuda::CUDAGuard device_guard(x_cat.device());
   check_same_cuda_device(x_cat, src_global_idx, "x_cat", "src_global_idx");
   check_same_cuda_device(x_cat, flat_dst, "x_cat", "flat_dst");

   const int64_t emb = x_cat.size(1);
   Tensor out = at::zeros({out_rows, emb}, x_cat.options());
   if(src_global_idx.numel() == 0 || out_rows == 0 || emb == 0) {
      return out;
   }

   Tensor x_work = ensure_contiguous(x_cat);
   Tensor src_work = ensure_contiguous(src_global_idx);
   Tensor dst_work = ensure_contiguous(flat_dst);
   cudaStream_t stream = current_cuda_stream(x_work);

   dispatch_float_or_double(
      x_work,
      [&]() { launch_fanout_scatter< float >(x_work, src_work, dst_work, emb, out, stream); },
      [&]() { launch_fanout_scatter< double >(x_work, src_work, dst_work, emb, out, stream); },
      "fanout_scatter_cuda"
   );
   return out;
}

Tensor fanout_scatter_backward_cuda(
   const Tensor& grad_out,
   const Tensor& src_global_idx,
   const Tensor& flat_dst,
   int64_t x_rows
)
{
   c10::cuda::CUDAGuard device_guard(grad_out.device());
   check_same_cuda_device(grad_out, src_global_idx, "grad_out", "src_global_idx");
   check_same_cuda_device(grad_out, flat_dst, "grad_out", "flat_dst");

   const int64_t emb = grad_out.size(1);
   Tensor grad_x = at::zeros({x_rows, emb}, grad_out.options());
   if(src_global_idx.numel() == 0 || x_rows == 0 || emb == 0) {
      return grad_x;
   }

   Tensor grad_out_work = ensure_contiguous(grad_out);
   Tensor src_work = ensure_contiguous(src_global_idx);
   Tensor dst_work = ensure_contiguous(flat_dst);
   cudaStream_t stream = current_cuda_stream(grad_out_work);

   dispatch_float_or_double(
      grad_out_work,
      [&]() {
         launch_fanout_scatter_backward< float >(
            grad_out_work, src_work, dst_work, emb, grad_x, stream
         );
      },
      [&]() {
         launch_fanout_scatter_backward< double >(
            grad_out_work, src_work, dst_work, emb, grad_x, stream
         );
      },
      "fanout_scatter_backward_cuda"
   );
   return grad_x;
}

Tensor fanin_reduce_sum_cuda(
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t dim_size
)
{
   c10::cuda::CUDAGuard device_guard(rel_flat.device());
   check_same_cuda_device(rel_flat, flat_src, "rel_flat", "flat_src");
   check_same_cuda_device(rel_flat, dst_idx, "rel_flat", "dst_idx");

   const int64_t emb = rel_flat.size(1);
   Tensor out = at::zeros({dim_size, emb}, rel_flat.options());
   if(flat_src.numel() == 0 || dim_size == 0 || emb == 0) {
      return out;
   }

   Tensor rel_work = ensure_contiguous(rel_flat);
   Tensor src_work = ensure_contiguous(flat_src);
   Tensor dst_work = ensure_contiguous(dst_idx);
   cudaStream_t stream = current_cuda_stream(rel_work);

   dispatch_float_or_double(
      rel_work,
      [&]() { launch_fanin_reduce_sum< float >(rel_work, src_work, dst_work, emb, out, stream); },
      [&]() { launch_fanin_reduce_sum< double >(rel_work, src_work, dst_work, emb, out, stream); },
      "fanin_reduce_sum_cuda"
   );
   return out;
}

Tensor fanin_reduce_logsumexp_cuda(
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t dim_size
)
{
   c10::cuda::CUDAGuard device_guard(rel_flat.device());
   check_same_cuda_device(rel_flat, flat_src, "rel_flat", "flat_src");
   check_same_cuda_device(rel_flat, dst_idx, "rel_flat", "dst_idx");

   const int64_t emb = rel_flat.size(1);
   Tensor out = at::full(
      {dim_size, emb}, -std::numeric_limits< double >::infinity(), rel_flat.options()
   );
   if(flat_src.numel() == 0 || dim_size == 0 || emb == 0) {
      return out;
   }

   Tensor rel_work = ensure_contiguous(rel_flat);
   Tensor src_work = ensure_contiguous(flat_src);
   Tensor dst_work = ensure_contiguous(dst_idx);
   cudaStream_t stream = current_cuda_stream(rel_work);

   const int64_t num_edges = src_work.size(0);
   if(choose_fanin_logsumexp_segmented(num_edges, dim_size, emb)) {
      auto sorted_pair = at::sort(dst_work, 0, false);
      Tensor sorted_dst = std::get< 0 >(sorted_pair);
      Tensor perm = std::get< 1 >(sorted_pair);
      Tensor sorted_src = src_work.index_select(0, perm);
      Tensor counts = at::bincount(sorted_dst, {}, dim_size);
      if(dtype_of(counts) != at::kLong) {
         counts = counts.to(at::kLong);
      }
      Tensor offsets = at::zeros({dim_size + 1}, counts.options());
      offsets.slice(0, 1, dim_size + 1).copy_(counts.cumsum(0));

      dispatch_float_or_double(
         rel_work,
         [&]() {
            launch_fanin_reduce_logsumexp_segmented< float >(
               rel_work, sorted_src, offsets, dim_size, emb, out, stream
            );
         },
         [&]() {
            launch_fanin_reduce_logsumexp_segmented< double >(
               rel_work, sorted_src, offsets, dim_size, emb, out, stream
            );
         },
         "fanin_reduce_logsumexp_cuda_segmented"
      );
      return out;
   }

   Tensor exp_sums = at::zeros({dim_size, emb}, rel_flat.options());

   dispatch_float_or_double(
      rel_work,
      [&]() {
         launch_fanin_reduce_logsumexp_atomic< float >(
            rel_work, src_work, dst_work, dim_size, emb, out, exp_sums, stream
         );
      },
      [&]() {
         launch_fanin_reduce_logsumexp_atomic< double >(
            rel_work, src_work, dst_work, dim_size, emb, out, exp_sums, stream
         );
      },
      "fanin_reduce_logsumexp_cuda"
   );
   return out;
}

Tensor fanin_reduce_sum_backward_cuda(
   const Tensor& grad_out,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t rel_rows
)
{
   c10::cuda::CUDAGuard device_guard(grad_out.device());
   check_same_cuda_device(grad_out, flat_src, "grad_out", "flat_src");
   check_same_cuda_device(grad_out, dst_idx, "grad_out", "dst_idx");

   const int64_t emb = grad_out.size(1);
   Tensor grad_rel = at::zeros({rel_rows, emb}, grad_out.options());
   if(flat_src.numel() == 0 || rel_rows == 0 || emb == 0) {
      return grad_rel;
   }

   Tensor grad_out_work = ensure_contiguous(grad_out);
   Tensor src_work = ensure_contiguous(flat_src);
   Tensor dst_work = ensure_contiguous(dst_idx);
   cudaStream_t stream = current_cuda_stream(grad_out_work);

   dispatch_float_or_double(
      grad_out_work,
      [&]() {
         launch_fanin_reduce_sum_backward< float >(
            grad_out_work, src_work, dst_work, emb, grad_rel, stream
         );
      },
      [&]() {
         launch_fanin_reduce_sum_backward< double >(
            grad_out_work, src_work, dst_work, emb, grad_rel, stream
         );
      },
      "fanin_reduce_sum_backward_cuda"
   );
   return grad_rel;
}

Tensor fanin_reduce_logsumexp_backward_cuda(
   const Tensor& grad_out,
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   const Tensor& out,
   int64_t rel_rows
)
{
   c10::cuda::CUDAGuard device_guard(grad_out.device());
   check_same_cuda_device(grad_out, rel_flat, "grad_out", "rel_flat");
   check_same_cuda_device(grad_out, flat_src, "grad_out", "flat_src");
   check_same_cuda_device(grad_out, dst_idx, "grad_out", "dst_idx");
   check_same_cuda_device(grad_out, out, "grad_out", "out");

   TORCH_CHECK(
      dtype_of(grad_out) == dtype_of(rel_flat) && dtype_of(grad_out) == dtype_of(out),
      "grad_out, rel_flat, and out must share dtype."
   );

   const int64_t emb = grad_out.size(1);
   Tensor grad_rel = at::zeros({rel_rows, emb}, rel_flat.options());
   if(flat_src.numel() == 0 || rel_rows == 0 || emb == 0) {
      return grad_rel;
   }

   Tensor grad_out_work = ensure_contiguous(grad_out);
   Tensor rel_work = ensure_contiguous(rel_flat);
   Tensor src_work = ensure_contiguous(flat_src);
   Tensor dst_work = ensure_contiguous(dst_idx);
   Tensor out_work = ensure_contiguous(out);
   cudaStream_t stream = current_cuda_stream(grad_out_work);

   dispatch_float_or_double(
      rel_work,
      [&]() {
         launch_fanin_reduce_logsumexp_backward< float >(
            grad_out_work, rel_work, src_work, dst_work, out_work, emb, grad_rel, stream
         );
      },
      [&]() {
         launch_fanin_reduce_logsumexp_backward< double >(
            grad_out_work, rel_work, src_work, dst_work, out_work, emb, grad_rel, stream
         );
      },
      "fanin_reduce_logsumexp_backward_cuda"
   );
   return grad_rel;
}

std::tuple< Tensor, Tensor > fused_two_layer_pointwise_from_indices_cuda(
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t total_slots,
   int64_t arity,
   const Tensor& w1_stack,
   const Tensor& b1_stack,
   const Tensor& w2_stack,
   const Tensor& b2_stack,
   int64_t pointwise_code
)
{
   c10::cuda::CUDAGuard device_guard(x.device());
   check_same_cuda_device(x, relation_args, "x", "relation_args");
   check_same_cuda_device(x, slot_offsets, "x", "slot_offsets");
   check_same_cuda_device(x, row_offsets, "x", "row_offsets");
   check_same_cuda_device(x, out_offsets, "x", "out_offsets");
   check_same_cuda_device(x, w1_stack, "x", "w1_stack");
   check_same_cuda_device(x, w2_stack, "x", "w2_stack");
   if(b1_stack.numel() > 0) {
      check_same_cuda_device(x, b1_stack, "x", "b1_stack");
   }
   if(b2_stack.numel() > 0) {
      check_same_cuda_device(x, b2_stack, "x", "b2_stack");
   }

   const int64_t emb = x.size(1);
   Tensor rel_cat = at::empty({total_slots, emb}, x.options());
   Tensor node_idx = at::empty({total_slots}, relation_args.options().dtype(at::kLong));
   if(total_slots <= 0) {
      return std::make_tuple(rel_cat, node_idx);
   }

   Tensor x_work = ensure_contiguous(x);
   Tensor relation_args_work = ensure_contiguous(relation_args);
   Tensor slot_offsets_work = ensure_contiguous(slot_offsets);
   Tensor row_offsets_work = ensure_contiguous(row_offsets);
   Tensor out_offsets_work = ensure_contiguous(out_offsets);
   Tensor w1_work = ensure_contiguous(w1_stack);
   Tensor b1_work = b1_stack.numel() > 0 ? ensure_contiguous(b1_stack) : b1_stack;
   Tensor w2_work = ensure_contiguous(w2_stack);
   Tensor b2_work = b2_stack.numel() > 0 ? ensure_contiguous(b2_stack) : b2_stack;
   cudaStream_t stream = current_cuda_stream(x_work);

   if(dtype_of(relation_args_work) == at::kInt) {
      dispatch_float_or_double(
         x_work,
         [&]() {
            launch_fused_two_layer_pointwise_from_indices< float, int >(
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w1_work,
               b1_work,
               w2_work,
               b2_work,
               pointwise_code,
               rel_cat,
               node_idx,
               stream
            );
         },
         [&]() {
            launch_fused_two_layer_pointwise_from_indices< double, int >(
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w1_work,
               b1_work,
               w2_work,
               b2_work,
               pointwise_code,
               rel_cat,
               node_idx,
               stream
            );
         },
         "fused_two_layer_pointwise_from_indices_cuda"
      );
      return std::make_tuple(rel_cat, node_idx);
   }

   dispatch_float_or_double(
      x_work,
      [&]() {
         launch_fused_two_layer_pointwise_from_indices< float, int64_t >(
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w1_work,
            b1_work,
            w2_work,
            b2_work,
            pointwise_code,
            rel_cat,
            node_idx,
            stream
         );
      },
      [&]() {
         launch_fused_two_layer_pointwise_from_indices< double, int64_t >(
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w1_work,
            b1_work,
            w2_work,
            b2_work,
            pointwise_code,
            rel_cat,
            node_idx,
            stream
         );
      },
      "fused_two_layer_pointwise_from_indices_cuda"
   );
   return std::make_tuple(rel_cat, node_idx);
}

std::tuple< Tensor, Tensor > fused_program_two_layer_silu_then_two_layer_silu_from_indices_cuda(
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t total_slots,
   int64_t arity,
   const Tensor& w10_stack,
   const Tensor& b10_stack,
   const Tensor& w20_stack,
   const Tensor& b20_stack,
   const Tensor& w11_stack,
   const Tensor& b11_stack,
   const Tensor& w21_stack,
   const Tensor& b21_stack
)
{
   c10::cuda::CUDAGuard device_guard(x.device());
   check_same_cuda_device(x, relation_args, "x", "relation_args");
   check_same_cuda_device(x, slot_offsets, "x", "slot_offsets");
   check_same_cuda_device(x, row_offsets, "x", "row_offsets");
   check_same_cuda_device(x, out_offsets, "x", "out_offsets");
   check_same_cuda_device(x, w10_stack, "x", "w10_stack");
   check_same_cuda_device(x, b10_stack, "x", "b10_stack");
   check_same_cuda_device(x, w20_stack, "x", "w20_stack");
   check_same_cuda_device(x, b20_stack, "x", "b20_stack");
   check_same_cuda_device(x, w11_stack, "x", "w11_stack");
   check_same_cuda_device(x, b11_stack, "x", "b11_stack");
   check_same_cuda_device(x, w21_stack, "x", "w21_stack");
   check_same_cuda_device(x, b21_stack, "x", "b21_stack");

   const int64_t emb = x.size(1);
   Tensor rel_cat = at::empty({total_slots, emb}, x.options());
   Tensor node_idx = at::empty({total_slots}, relation_args.options().dtype(at::kLong));
   if(total_slots <= 0) {
      return std::make_tuple(rel_cat, node_idx);
   }

   Tensor x_work = ensure_contiguous(x);
   Tensor relation_args_work = ensure_contiguous(relation_args);
   Tensor slot_offsets_work = ensure_contiguous(slot_offsets);
   Tensor row_offsets_work = ensure_contiguous(row_offsets);
   Tensor out_offsets_work = ensure_contiguous(out_offsets);
   Tensor w10_work = ensure_contiguous(w10_stack);
   Tensor b10_work = ensure_contiguous(b10_stack);
   Tensor w20_work = ensure_contiguous(w20_stack);
   Tensor b20_work = ensure_contiguous(b20_stack);
   Tensor w11_work = ensure_contiguous(w11_stack);
   Tensor b11_work = ensure_contiguous(b11_stack);
   Tensor w21_work = ensure_contiguous(w21_stack);
   Tensor b21_work = ensure_contiguous(b21_stack);
   cudaStream_t stream = current_cuda_stream(x_work);

   if(dtype_of(relation_args_work) == at::kInt) {
      dispatch_float_or_double(
         x_work,
         [&]() {
            launch_fused_program_two_layer_silu_then_two_layer_silu_from_indices< float, int >(
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w10_work,
               b10_work,
               w20_work,
               b20_work,
               w11_work,
               b11_work,
               w21_work,
               b21_work,
               rel_cat,
               node_idx,
               stream
            );
         },
         [&]() {
            launch_fused_program_two_layer_silu_then_two_layer_silu_from_indices< double, int >(
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w10_work,
               b10_work,
               w20_work,
               b20_work,
               w11_work,
               b11_work,
               w21_work,
               b21_work,
               rel_cat,
               node_idx,
               stream
            );
         },
         "fused_program_two_layer_silu_then_two_layer_silu_from_indices_cuda"
      );
      return std::make_tuple(rel_cat, node_idx);
   }

   dispatch_float_or_double(
      x_work,
      [&]() {
         launch_fused_program_two_layer_silu_then_two_layer_silu_from_indices< float, int64_t >(
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w10_work,
            b10_work,
            w20_work,
            b20_work,
            w11_work,
            b11_work,
            w21_work,
            b21_work,
            rel_cat,
            node_idx,
            stream
         );
      },
      [&]() {
         launch_fused_program_two_layer_silu_then_two_layer_silu_from_indices< double, int64_t >(
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w10_work,
            b10_work,
            w20_work,
            b20_work,
            w11_work,
            b11_work,
            w21_work,
            b21_work,
            rel_cat,
            node_idx,
            stream
         );
      },
      "fused_program_two_layer_silu_then_two_layer_silu_from_indices_cuda"
   );
   return std::make_tuple(rel_cat, node_idx);
}

std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor >
fused_program_two_layer_silu_then_two_layer_silu_from_indices_backward_cuda(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t arity,
   const Tensor& w10_stack,
   const Tensor& b10_stack,
   const Tensor& w20_stack,
   const Tensor& b20_stack,
   const Tensor& w11_stack,
   const Tensor& b11_stack,
   const Tensor& w21_stack,
   const Tensor& b21_stack
)
{
   c10::cuda::CUDAGuard device_guard(x.device());
   check_same_cuda_device(grad_rel, x, "grad_rel", "x");
   check_same_cuda_device(x, relation_args, "x", "relation_args");
   check_same_cuda_device(x, slot_offsets, "x", "slot_offsets");
   check_same_cuda_device(x, row_offsets, "x", "row_offsets");
   check_same_cuda_device(x, out_offsets, "x", "out_offsets");
   check_same_cuda_device(x, w10_stack, "x", "w10_stack");
   check_same_cuda_device(x, b10_stack, "x", "b10_stack");
   check_same_cuda_device(x, w20_stack, "x", "w20_stack");
   check_same_cuda_device(x, b20_stack, "x", "b20_stack");
   check_same_cuda_device(x, w11_stack, "x", "w11_stack");
   check_same_cuda_device(x, b11_stack, "x", "b11_stack");
   check_same_cuda_device(x, w21_stack, "x", "w21_stack");
   check_same_cuda_device(x, b21_stack, "x", "b21_stack");

   Tensor grad_x = at::zeros_like(x);
   Tensor grad_w10 = at::zeros_like(w10_stack);
   Tensor grad_b10 = at::zeros_like(b10_stack);
   Tensor grad_w20 = at::zeros_like(w20_stack);
   Tensor grad_b20 = at::zeros_like(b20_stack);
   Tensor grad_w11 = at::zeros_like(w11_stack);
   Tensor grad_b11 = at::zeros_like(b11_stack);
   Tensor grad_w21 = at::zeros_like(w21_stack);
   Tensor grad_b21 = at::zeros_like(b21_stack);

   if(total_rows <= 0) {
      return std::make_tuple(
         grad_x,
         grad_w10,
         grad_b10,
         grad_w20,
         grad_b20,
         grad_w11,
         grad_b11,
         grad_w21,
         grad_b21
      );
   }

   Tensor grad_rel_work = ensure_contiguous(grad_rel);
   Tensor x_work = ensure_contiguous(x);
   Tensor relation_args_work = ensure_contiguous(relation_args);
   Tensor slot_offsets_work = ensure_contiguous(slot_offsets);
   Tensor row_offsets_work = ensure_contiguous(row_offsets);
   Tensor out_offsets_work = ensure_contiguous(out_offsets);
   Tensor w10_work = ensure_contiguous(w10_stack);
   Tensor b10_work = ensure_contiguous(b10_stack);
   Tensor w20_work = ensure_contiguous(w20_stack);
   Tensor b20_work = ensure_contiguous(b20_stack);
   Tensor w11_work = ensure_contiguous(w11_stack);
   Tensor b11_work = ensure_contiguous(b11_stack);
   Tensor w21_work = ensure_contiguous(w21_stack);
   Tensor b21_work = ensure_contiguous(b21_stack);
   cudaStream_t stream = current_cuda_stream(x_work);

   if(dtype_of(relation_args_work) == at::kInt) {
      dispatch_float_or_double(
         x_work,
         [&]() {
            launch_fused_program_two_layer_silu_then_two_layer_silu_from_indices_backward< float, int >(
               grad_rel_work,
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w10_work,
               b10_work,
               w20_work,
               b20_work,
               w11_work,
               b11_work,
               w21_work,
               b21_work,
               grad_x,
               grad_w10,
               grad_b10,
               grad_w20,
               grad_b20,
               grad_w11,
               grad_b11,
               grad_w21,
               grad_b21,
               stream
            );
         },
         [&]() {
            launch_fused_program_two_layer_silu_then_two_layer_silu_from_indices_backward< double, int >(
               grad_rel_work,
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w10_work,
               b10_work,
               w20_work,
               b20_work,
               w11_work,
               b11_work,
               w21_work,
               b21_work,
               grad_x,
               grad_w10,
               grad_b10,
               grad_w20,
               grad_b20,
               grad_w11,
               grad_b11,
               grad_w21,
               grad_b21,
               stream
            );
         },
         "fused_program_two_layer_silu_then_two_layer_silu_from_indices_backward_cuda"
      );
      return std::make_tuple(
         grad_x,
         grad_w10,
         grad_b10,
         grad_w20,
         grad_b20,
         grad_w11,
         grad_b11,
         grad_w21,
         grad_b21
      );
   }

   dispatch_float_or_double(
      x_work,
      [&]() {
         launch_fused_program_two_layer_silu_then_two_layer_silu_from_indices_backward< float, int64_t >(
            grad_rel_work,
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w10_work,
            b10_work,
            w20_work,
            b20_work,
            w11_work,
            b11_work,
            w21_work,
            b21_work,
            grad_x,
            grad_w10,
            grad_b10,
            grad_w20,
            grad_b20,
            grad_w11,
            grad_b11,
            grad_w21,
            grad_b21,
            stream
         );
      },
      [&]() {
         launch_fused_program_two_layer_silu_then_two_layer_silu_from_indices_backward< double, int64_t >(
            grad_rel_work,
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w10_work,
            b10_work,
            w20_work,
            b20_work,
            w11_work,
            b11_work,
            w21_work,
            b21_work,
            grad_x,
            grad_w10,
            grad_b10,
            grad_w20,
            grad_b20,
            grad_w11,
            grad_b11,
            grad_w21,
            grad_b21,
            stream
         );
      },
      "fused_program_two_layer_silu_then_two_layer_silu_from_indices_backward_cuda"
   );
   return std::make_tuple(
      grad_x,
      grad_w10,
      grad_b10,
      grad_w20,
      grad_b20,
      grad_w11,
      grad_b11,
      grad_w21,
      grad_b21
   );
}

std::tuple< Tensor, Tensor > fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_cuda(
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t total_slots,
   int64_t arity,
   const Tensor& w10_stack,
   const Tensor& b10_stack,
   const Tensor& w20_stack,
   const Tensor& b20_stack,
   const Tensor& w11_stack,
   const Tensor& b11_stack,
   const Tensor& w21_stack,
   const Tensor& b21_stack,
   const Tensor& ln_weight_stack,
   const Tensor& ln_bias_stack,
   double ln_eps
)
{
   c10::cuda::CUDAGuard device_guard(x.device());
   check_same_cuda_device(x, relation_args, "x", "relation_args");
   check_same_cuda_device(x, slot_offsets, "x", "slot_offsets");
   check_same_cuda_device(x, row_offsets, "x", "row_offsets");
   check_same_cuda_device(x, out_offsets, "x", "out_offsets");
   check_same_cuda_device(x, w10_stack, "x", "w10_stack");
   check_same_cuda_device(x, b10_stack, "x", "b10_stack");
   check_same_cuda_device(x, w20_stack, "x", "w20_stack");
   check_same_cuda_device(x, b20_stack, "x", "b20_stack");
   check_same_cuda_device(x, w11_stack, "x", "w11_stack");
   check_same_cuda_device(x, b11_stack, "x", "b11_stack");
   check_same_cuda_device(x, w21_stack, "x", "w21_stack");
   check_same_cuda_device(x, b21_stack, "x", "b21_stack");
   if(ln_weight_stack.numel() > 0) {
      check_same_cuda_device(x, ln_weight_stack, "x", "ln_weight_stack");
   }
   if(ln_bias_stack.numel() > 0) {
      check_same_cuda_device(x, ln_bias_stack, "x", "ln_bias_stack");
   }

   const int64_t emb = x.size(1);
   Tensor rel_cat = at::empty({total_slots, emb}, x.options());
   Tensor node_idx = at::empty({total_slots}, relation_args.options().dtype(at::kLong));
   if(total_slots <= 0) {
      return std::make_tuple(rel_cat, node_idx);
   }

   Tensor x_work = ensure_contiguous(x);
   Tensor relation_args_work = ensure_contiguous(relation_args);
   Tensor slot_offsets_work = ensure_contiguous(slot_offsets);
   Tensor row_offsets_work = ensure_contiguous(row_offsets);
   Tensor out_offsets_work = ensure_contiguous(out_offsets);
   Tensor w10_work = ensure_contiguous(w10_stack);
   Tensor b10_work = ensure_contiguous(b10_stack);
   Tensor w20_work = ensure_contiguous(w20_stack);
   Tensor b20_work = ensure_contiguous(b20_stack);
   Tensor w11_work = ensure_contiguous(w11_stack);
   Tensor b11_work = ensure_contiguous(b11_stack);
   Tensor w21_work = ensure_contiguous(w21_stack);
   Tensor b21_work = ensure_contiguous(b21_stack);
   Tensor ln_weight_work =
      ln_weight_stack.numel() > 0 ? ensure_contiguous(ln_weight_stack) : ln_weight_stack;
   Tensor ln_bias_work = ln_bias_stack.numel() > 0 ? ensure_contiguous(ln_bias_stack) : ln_bias_stack;
   cudaStream_t stream = current_cuda_stream(x_work);

   if(dtype_of(relation_args_work) == at::kInt) {
      dispatch_float_or_double(
         x_work,
         [&]() {
            launch_fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices< float, int >(
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w10_work,
               b10_work,
               w20_work,
               b20_work,
               w11_work,
               b11_work,
               w21_work,
               b21_work,
               ln_weight_work,
               ln_bias_work,
               static_cast< float >(ln_eps),
               rel_cat,
               node_idx,
               stream
            );
         },
         [&]() {
            launch_fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices< double, int >(
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w10_work,
               b10_work,
               w20_work,
               b20_work,
               w11_work,
               b11_work,
               w21_work,
               b21_work,
               ln_weight_work,
               ln_bias_work,
               static_cast< double >(ln_eps),
               rel_cat,
               node_idx,
               stream
            );
         },
         "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_cuda"
      );
      return std::make_tuple(rel_cat, node_idx);
   }

   dispatch_float_or_double(
      x_work,
      [&]() {
         launch_fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices< float, int64_t >(
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w10_work,
            b10_work,
            w20_work,
            b20_work,
            w11_work,
            b11_work,
            w21_work,
            b21_work,
            ln_weight_work,
            ln_bias_work,
            static_cast< float >(ln_eps),
            rel_cat,
            node_idx,
            stream
         );
      },
      [&]() {
         launch_fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices< double, int64_t >(
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w10_work,
            b10_work,
            w20_work,
            b20_work,
            w11_work,
            b11_work,
            w21_work,
            b21_work,
            ln_weight_work,
            ln_bias_work,
            static_cast< double >(ln_eps),
            rel_cat,
            node_idx,
            stream
         );
      },
      "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_cuda"
   );
   return std::make_tuple(rel_cat, node_idx);
}

std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor >
fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_backward_cuda(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t arity,
   const Tensor& w10_stack,
   const Tensor& b10_stack,
   const Tensor& w20_stack,
   const Tensor& b20_stack,
   const Tensor& w11_stack,
   const Tensor& b11_stack,
   const Tensor& w21_stack,
   const Tensor& b21_stack,
   const Tensor& ln_weight_stack,
   const Tensor& ln_bias_stack,
   double ln_eps
)
{
   c10::cuda::CUDAGuard device_guard(x.device());
   check_same_cuda_device(grad_rel, x, "grad_rel", "x");
   check_same_cuda_device(x, relation_args, "x", "relation_args");
   check_same_cuda_device(x, slot_offsets, "x", "slot_offsets");
   check_same_cuda_device(x, row_offsets, "x", "row_offsets");
   check_same_cuda_device(x, out_offsets, "x", "out_offsets");
   check_same_cuda_device(x, w10_stack, "x", "w10_stack");
   check_same_cuda_device(x, b10_stack, "x", "b10_stack");
   check_same_cuda_device(x, w20_stack, "x", "w20_stack");
   check_same_cuda_device(x, b20_stack, "x", "b20_stack");
   check_same_cuda_device(x, w11_stack, "x", "w11_stack");
   check_same_cuda_device(x, b11_stack, "x", "b11_stack");
   check_same_cuda_device(x, w21_stack, "x", "w21_stack");
   check_same_cuda_device(x, b21_stack, "x", "b21_stack");
   if(ln_weight_stack.numel() > 0) {
      check_same_cuda_device(x, ln_weight_stack, "x", "ln_weight_stack");
   }
   if(ln_bias_stack.numel() > 0) {
      check_same_cuda_device(x, ln_bias_stack, "x", "ln_bias_stack");
   }

   Tensor grad_x = at::zeros_like(x);
   Tensor grad_w10 = at::zeros_like(w10_stack);
   Tensor grad_b10 = at::zeros_like(b10_stack);
   Tensor grad_w20 = at::zeros_like(w20_stack);
   Tensor grad_b20 = at::zeros_like(b20_stack);
   Tensor grad_w11 = at::zeros_like(w11_stack);
   Tensor grad_b11 = at::zeros_like(b11_stack);
   Tensor grad_w21 = at::zeros_like(w21_stack);
   Tensor grad_b21 = at::zeros_like(b21_stack);
   Tensor grad_ln_weight =
      ln_weight_stack.numel() > 0 ? at::zeros_like(ln_weight_stack) : at::zeros_like(ln_weight_stack);
   Tensor grad_ln_bias =
      ln_bias_stack.numel() > 0 ? at::zeros_like(ln_bias_stack) : at::zeros_like(ln_bias_stack);
   if(total_rows <= 0) {
      return std::make_tuple(
         grad_x,
         grad_w10,
         grad_b10,
         grad_w20,
         grad_b20,
         grad_w11,
         grad_b11,
         grad_w21,
         grad_b21,
         grad_ln_weight,
         grad_ln_bias
      );
   }

   Tensor grad_rel_work = ensure_contiguous(grad_rel);
   Tensor x_work = ensure_contiguous(x);
   Tensor relation_args_work = ensure_contiguous(relation_args);
   Tensor slot_offsets_work = ensure_contiguous(slot_offsets);
   Tensor row_offsets_work = ensure_contiguous(row_offsets);
   Tensor out_offsets_work = ensure_contiguous(out_offsets);
   Tensor w10_work = ensure_contiguous(w10_stack);
   Tensor b10_work = ensure_contiguous(b10_stack);
   Tensor w20_work = ensure_contiguous(w20_stack);
   Tensor b20_work = ensure_contiguous(b20_stack);
   Tensor w11_work = ensure_contiguous(w11_stack);
   Tensor b11_work = ensure_contiguous(b11_stack);
   Tensor w21_work = ensure_contiguous(w21_stack);
   Tensor b21_work = ensure_contiguous(b21_stack);
   Tensor ln_weight_work =
      ln_weight_stack.numel() > 0 ? ensure_contiguous(ln_weight_stack) : ln_weight_stack;
   Tensor ln_bias_work = ln_bias_stack.numel() > 0 ? ensure_contiguous(ln_bias_stack) : ln_bias_stack;
   cudaStream_t stream = current_cuda_stream(x_work);

   if(dtype_of(relation_args_work) == at::kInt) {
      dispatch_float_or_double(
         x_work,
         [&]() {
            launch_fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_backward<
               float,
               int >(
               grad_rel_work,
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w10_work,
               b10_work,
               w20_work,
               b20_work,
               w11_work,
               b11_work,
               w21_work,
               b21_work,
               ln_weight_work,
               ln_bias_work,
               static_cast< float >(ln_eps),
               grad_x,
               grad_w10,
               grad_b10,
               grad_w20,
               grad_b20,
               grad_w11,
               grad_b11,
               grad_w21,
               grad_b21,
               grad_ln_weight,
               grad_ln_bias,
               stream
            );
         },
         [&]() {
            launch_fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_backward<
               double,
               int >(
               grad_rel_work,
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w10_work,
               b10_work,
               w20_work,
               b20_work,
               w11_work,
               b11_work,
               w21_work,
               b21_work,
               ln_weight_work,
               ln_bias_work,
               static_cast< double >(ln_eps),
               grad_x,
               grad_w10,
               grad_b10,
               grad_w20,
               grad_b20,
               grad_w11,
               grad_b11,
               grad_w21,
               grad_b21,
               grad_ln_weight,
               grad_ln_bias,
               stream
            );
         },
         "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_backward_cuda"
      );
      return std::make_tuple(
         grad_x,
         grad_w10,
         grad_b10,
         grad_w20,
         grad_b20,
         grad_w11,
         grad_b11,
         grad_w21,
         grad_b21,
         grad_ln_weight,
         grad_ln_bias
      );
   }

   dispatch_float_or_double(
      x_work,
      [&]() {
         launch_fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_backward<
            float,
            int64_t >(
            grad_rel_work,
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w10_work,
            b10_work,
            w20_work,
            b20_work,
            w11_work,
            b11_work,
            w21_work,
            b21_work,
            ln_weight_work,
            ln_bias_work,
            static_cast< float >(ln_eps),
            grad_x,
            grad_w10,
            grad_b10,
            grad_w20,
            grad_b20,
            grad_w11,
            grad_b11,
            grad_w21,
            grad_b21,
            grad_ln_weight,
            grad_ln_bias,
            stream
         );
      },
      [&]() {
         launch_fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_backward<
            double,
            int64_t >(
            grad_rel_work,
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w10_work,
            b10_work,
            w20_work,
            b20_work,
            w11_work,
            b11_work,
            w21_work,
            b21_work,
            ln_weight_work,
            ln_bias_work,
            static_cast< double >(ln_eps),
            grad_x,
            grad_w10,
            grad_b10,
            grad_w20,
            grad_b20,
            grad_w11,
            grad_b11,
            grad_w21,
            grad_b21,
            grad_ln_weight,
            grad_ln_bias,
            stream
         );
      },
      "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_backward_cuda"
   );
   return std::make_tuple(
      grad_x,
      grad_w10,
      grad_b10,
      grad_w20,
      grad_b20,
      grad_w11,
      grad_b11,
      grad_w21,
      grad_b21,
      grad_ln_weight,
      grad_ln_bias
   );
}

std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor >
fused_two_layer_pointwise_from_indices_backward_cuda(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t arity,
   const Tensor& w1_stack,
   const Tensor& b1_stack,
   const Tensor& w2_stack,
   const Tensor& b2_stack,
   int64_t pointwise_code
)
{
   c10::cuda::CUDAGuard device_guard(grad_rel.device());
   check_same_cuda_device(grad_rel, x, "grad_rel", "x");
   check_same_cuda_device(grad_rel, relation_args, "grad_rel", "relation_args");
   check_same_cuda_device(grad_rel, slot_offsets, "grad_rel", "slot_offsets");
   check_same_cuda_device(grad_rel, row_offsets, "grad_rel", "row_offsets");
   check_same_cuda_device(grad_rel, out_offsets, "grad_rel", "out_offsets");
   check_same_cuda_device(grad_rel, w1_stack, "grad_rel", "w1_stack");
   check_same_cuda_device(grad_rel, w2_stack, "grad_rel", "w2_stack");
   if(b1_stack.numel() > 0) {
      check_same_cuda_device(grad_rel, b1_stack, "grad_rel", "b1_stack");
   }
   if(b2_stack.numel() > 0) {
      check_same_cuda_device(grad_rel, b2_stack, "grad_rel", "b2_stack");
   }

   Tensor grad_x = at::zeros_like(x);
   Tensor grad_w1 = at::zeros_like(w1_stack);
   Tensor grad_b1 = b1_stack.numel() > 0 ? at::zeros_like(b1_stack) : at::zeros_like(b1_stack);
   Tensor grad_w2 = at::zeros_like(w2_stack);
   Tensor grad_b2 = b2_stack.numel() > 0 ? at::zeros_like(b2_stack) : at::zeros_like(b2_stack);
   if(total_rows <= 0) {
      return std::make_tuple(grad_x, grad_w1, grad_b1, grad_w2, grad_b2);
   }

   Tensor grad_rel_work = ensure_contiguous(grad_rel);
   Tensor x_work = ensure_contiguous(x);
   Tensor relation_args_work = ensure_contiguous(relation_args);
   Tensor slot_offsets_work = ensure_contiguous(slot_offsets);
   Tensor row_offsets_work = ensure_contiguous(row_offsets);
   Tensor out_offsets_work = ensure_contiguous(out_offsets);
   Tensor w1_work = ensure_contiguous(w1_stack);
   Tensor b1_work = b1_stack.numel() > 0 ? ensure_contiguous(b1_stack) : b1_stack;
   Tensor w2_work = ensure_contiguous(w2_stack);
   Tensor b2_work = b2_stack.numel() > 0 ? ensure_contiguous(b2_stack) : b2_stack;
   cudaStream_t stream = current_cuda_stream(grad_rel_work);

   if(dtype_of(relation_args_work) == at::kInt) {
      dispatch_float_or_double(
         grad_rel_work,
         [&]() {
            launch_fused_two_layer_pointwise_from_indices_backward< float, int >(
               grad_rel_work,
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w1_work,
               b1_work,
               w2_work,
               b2_work,
               pointwise_code,
               grad_x,
               grad_w1,
               grad_b1,
               grad_w2,
               grad_b2,
               stream
            );
         },
         [&]() {
            launch_fused_two_layer_pointwise_from_indices_backward< double, int >(
               grad_rel_work,
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w1_work,
               b1_work,
               w2_work,
               b2_work,
               pointwise_code,
               grad_x,
               grad_w1,
               grad_b1,
               grad_w2,
               grad_b2,
               stream
            );
         },
         "fused_two_layer_pointwise_from_indices_backward_cuda"
      );
      return std::make_tuple(grad_x, grad_w1, grad_b1, grad_w2, grad_b2);
   }

   dispatch_float_or_double(
      grad_rel_work,
      [&]() {
         launch_fused_two_layer_pointwise_from_indices_backward< float, int64_t >(
            grad_rel_work,
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w1_work,
            b1_work,
            w2_work,
            b2_work,
            pointwise_code,
            grad_x,
            grad_w1,
            grad_b1,
            grad_w2,
            grad_b2,
            stream
         );
      },
      [&]() {
         launch_fused_two_layer_pointwise_from_indices_backward< double, int64_t >(
            grad_rel_work,
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w1_work,
            b1_work,
            w2_work,
            b2_work,
            pointwise_code,
            grad_x,
            grad_w1,
            grad_b1,
            grad_w2,
            grad_b2,
            stream
         );
      },
      "fused_two_layer_pointwise_from_indices_backward_cuda"
   );
   return std::make_tuple(grad_x, grad_w1, grad_b1, grad_w2, grad_b2);
}

template < typename scalar_t, typename index_t >
__global__ void fused_postnorm_two_layer_pointwise_layernorm_from_indices_cuda_kernel(
   const scalar_t* x_ptr,
   const index_t* relation_args_ptr,
   const int64_t* slot_offsets_ptr,
   const int64_t* row_offsets_ptr,
   const int64_t* out_offsets_ptr,
   int64_t groups,
   int64_t arity,
   int64_t emb,
   const scalar_t* w1_ptr,
   const scalar_t* b1_ptr,
   bool has_b1,
   int64_t hidden,
   const scalar_t* w2_ptr,
   const scalar_t* b2_ptr,
   bool has_b2,
   const scalar_t* ln_weight_ptr,
   const scalar_t* ln_bias_ptr,
   bool has_ln_weight,
   bool has_ln_bias,
   scalar_t ln_eps,
   int64_t pointwise_code,
   scalar_t* out_ptr,
   int64_t* node_idx_ptr
)
{
   const int64_t row = static_cast< int64_t >(blockIdx.x);
   if(row >= row_offsets_ptr[groups]) {
      return;
   }

   int64_t group = 0;
   while(group + 1 < groups && row >= row_offsets_ptr[group + 1]) {
      ++group;
   }
   const int64_t row_in_group = row - row_offsets_ptr[group];
   const int64_t slot_base = slot_offsets_ptr[group] + row_in_group * arity;
   const int64_t out_base = out_offsets_ptr[group] + row_in_group * arity;
   const int64_t in_dim = arity * emb;

   extern __shared__ unsigned char smem_raw[];
   scalar_t* input = reinterpret_cast< scalar_t* >(smem_raw);
   scalar_t* hidden_buf = input + in_dim;
   scalar_t* norm_buf = hidden_buf + hidden;
   __shared__ scalar_t mean_val;
   __shared__ scalar_t invstd_val;

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      const int64_t slot = i / emb;
      const int64_t dim = i - slot * emb;
      const int64_t node = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
      input[i] = x_ptr[node * emb + dim];
   }
   for(int64_t slot = threadIdx.x; slot < arity; slot += blockDim.x) {
      node_idx_ptr[out_base + slot] = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
   }
   __syncthreads();

   const scalar_t* w1_group = w1_ptr + group * hidden * in_dim;
   const scalar_t* b1_group = has_b1 ? (b1_ptr + group * hidden) : nullptr;
   for(int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
      scalar_t acc = has_b1 ? b1_group[h] : static_cast< scalar_t >(0);
      const scalar_t* w_row = w1_group + h * in_dim;
      for(int64_t i = 0; i < in_dim; ++i) {
         acc += w_row[i] * input[i];
      }
      hidden_buf[h] = pointwise_apply_compat(pointwise_code, acc);
   }
   __syncthreads();

   const scalar_t* w2_group = w2_ptr + group * in_dim * hidden;
   const scalar_t* b2_group = has_b2 ? (b2_ptr + group * in_dim) : nullptr;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      scalar_t acc = has_b2 ? b2_group[o] : static_cast< scalar_t >(0);
      const scalar_t* w_row = w2_group + o * hidden;
      for(int64_t h = 0; h < hidden; ++h) {
         acc += w_row[h] * hidden_buf[h];
      }
      norm_buf[o] = acc;
   }
   __syncthreads();

   if(threadIdx.x == 0) {
      scalar_t mean = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         mean += norm_buf[o];
      }
      mean /= static_cast< scalar_t >(in_dim);
      scalar_t var = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         const scalar_t centered = norm_buf[o] - mean;
         var += centered * centered;
      }
      var /= static_cast< scalar_t >(in_dim);
      mean_val = mean;
      invstd_val = static_cast< scalar_t >(1) / sqrt(var + ln_eps);
   }
   __syncthreads();

   const scalar_t* ln_weight_group = has_ln_weight ? (ln_weight_ptr + group * in_dim) : nullptr;
   const scalar_t* ln_bias_group = has_ln_bias ? (ln_bias_ptr + group * in_dim) : nullptr;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      scalar_t xhat = (norm_buf[o] - mean_val) * invstd_val;
      if(has_ln_weight) {
         xhat *= ln_weight_group[o];
      }
      if(has_ln_bias) {
         xhat += ln_bias_group[o];
      }
      const int64_t slot = o / emb;
      const int64_t dim = o - slot * emb;
      out_ptr[(out_base + slot) * emb + dim] = input[o] + xhat;
   }
}

template < typename scalar_t, typename index_t >
void launch_fused_postnorm_two_layer_pointwise_layernorm_from_indices(
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t arity,
   const Tensor& w1_stack,
   const Tensor& b1_stack,
   const Tensor& w2_stack,
   const Tensor& b2_stack,
   const Tensor& ln_weight_stack,
   const Tensor& ln_bias_stack,
   scalar_t ln_eps,
   int64_t pointwise_code,
   Tensor& rel_cat,
   Tensor& node_idx,
   cudaStream_t stream
)
{
   const int64_t groups = slot_offsets.size(0);
   if(total_rows <= 0) {
      return;
   }
   const int64_t emb = x.size(1);
   const int64_t hidden = w1_stack.size(1);
   const int64_t in_dim = arity * emb;
   const size_t shared_bytes =
      static_cast< size_t >(2 * in_dim + hidden) * sizeof(scalar_t);
   const dim3 grid(static_cast< unsigned int >(total_rows));
   fused_postnorm_two_layer_pointwise_layernorm_from_indices_cuda_kernel< scalar_t, index_t >
      <<<grid, static_cast< int >(kThreads), shared_bytes, stream>>>(
         x.data_ptr< scalar_t >(),
         relation_args.data_ptr< index_t >(),
         slot_offsets.data_ptr< int64_t >(),
         row_offsets.data_ptr< int64_t >(),
         out_offsets.data_ptr< int64_t >(),
         groups,
         arity,
         emb,
         w1_stack.data_ptr< scalar_t >(),
         b1_stack.numel() > 0 ? b1_stack.data_ptr< scalar_t >() : nullptr,
         b1_stack.numel() > 0,
         hidden,
         w2_stack.data_ptr< scalar_t >(),
         b2_stack.numel() > 0 ? b2_stack.data_ptr< scalar_t >() : nullptr,
         b2_stack.numel() > 0,
         ln_weight_stack.numel() > 0 ? ln_weight_stack.data_ptr< scalar_t >() : nullptr,
         ln_bias_stack.numel() > 0 ? ln_bias_stack.data_ptr< scalar_t >() : nullptr,
         ln_weight_stack.numel() > 0,
         ln_bias_stack.numel() > 0,
         ln_eps,
         pointwise_code,
         rel_cat.data_ptr< scalar_t >(),
         node_idx.data_ptr< int64_t >()
      );
   check_kernel_launch("fused_postnorm_two_layer_pointwise_layernorm_from_indices_cuda_kernel");
}

template < typename scalar_t, typename index_t >
__global__ void fused_postnorm_two_layer_pointwise_layernorm_from_indices_backward_cuda_kernel(
   const scalar_t* grad_rel_ptr,
   const scalar_t* x_ptr,
   const index_t* relation_args_ptr,
   const int64_t* slot_offsets_ptr,
   const int64_t* row_offsets_ptr,
   const int64_t* out_offsets_ptr,
   int64_t groups,
   int64_t arity,
   int64_t emb,
   const scalar_t* w1_ptr,
   const scalar_t* b1_ptr,
   bool has_b1,
   int64_t hidden,
   const scalar_t* w2_ptr,
   const scalar_t* b2_ptr,
   bool has_b2,
   const scalar_t* ln_weight_ptr,
   const scalar_t* ln_bias_ptr,
   bool has_ln_weight,
   bool has_ln_bias,
   scalar_t ln_eps,
   int64_t pointwise_code,
   scalar_t* grad_x_ptr,
   scalar_t* grad_w1_ptr,
   scalar_t* grad_b1_ptr,
   scalar_t* grad_w2_ptr,
   scalar_t* grad_b2_ptr,
   scalar_t* grad_ln_weight_ptr,
   scalar_t* grad_ln_bias_ptr
)
{
   const int64_t row = static_cast< int64_t >(blockIdx.x);
   if(row >= row_offsets_ptr[groups]) {
      return;
   }

   int64_t group = 0;
   while(group + 1 < groups && row >= row_offsets_ptr[group + 1]) {
      ++group;
   }
   const int64_t row_in_group = row - row_offsets_ptr[group];
   const int64_t slot_base = slot_offsets_ptr[group] + row_in_group * arity;
   const int64_t out_base = out_offsets_ptr[group] + row_in_group * arity;
   const int64_t in_dim = arity * emb;

   extern __shared__ unsigned char smem_raw[];
   scalar_t* input = reinterpret_cast< scalar_t* >(smem_raw);
   scalar_t* z1 = input + in_dim;
   scalar_t* hidden_buf = z1 + hidden;
   scalar_t* xhat_buf = hidden_buf + hidden;
   scalar_t* grad_z2 = xhat_buf + in_dim;
   scalar_t* grad_z1 = grad_z2 + in_dim;
   __shared__ scalar_t mean_val;
   __shared__ scalar_t invstd_val;
   __shared__ scalar_t mean_dxhat;
   __shared__ scalar_t mean_dxhat_xhat;

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      const int64_t slot = i / emb;
      const int64_t dim = i - slot * emb;
      const int64_t node = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
      input[i] = x_ptr[node * emb + dim];
   }
   __syncthreads();

   const scalar_t* w1_group = w1_ptr + group * hidden * in_dim;
   const scalar_t* b1_group = has_b1 ? (b1_ptr + group * hidden) : nullptr;
   for(int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
      scalar_t acc = has_b1 ? b1_group[h] : static_cast< scalar_t >(0);
      const scalar_t* w_row = w1_group + h * in_dim;
      for(int64_t i = 0; i < in_dim; ++i) {
         acc += w_row[i] * input[i];
      }
      z1[h] = acc;
      hidden_buf[h] = pointwise_apply_compat(pointwise_code, acc);
   }
   __syncthreads();

   const scalar_t* w2_group = w2_ptr + group * in_dim * hidden;
   const scalar_t* b2_group = has_b2 ? (b2_ptr + group * in_dim) : nullptr;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      scalar_t acc = has_b2 ? b2_group[o] : static_cast< scalar_t >(0);
      const scalar_t* w_row = w2_group + o * hidden;
      for(int64_t h = 0; h < hidden; ++h) {
         acc += w_row[h] * hidden_buf[h];
      }
      xhat_buf[o] = acc;
   }
   __syncthreads();

   if(threadIdx.x == 0) {
      scalar_t mean = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         mean += xhat_buf[o];
      }
      mean /= static_cast< scalar_t >(in_dim);
      scalar_t var = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         const scalar_t centered = xhat_buf[o] - mean;
         var += centered * centered;
      }
      var /= static_cast< scalar_t >(in_dim);
      mean_val = mean;
      invstd_val = static_cast< scalar_t >(1) / sqrt(var + ln_eps);
   }
   __syncthreads();

   const scalar_t* ln_weight_group = has_ln_weight ? (ln_weight_ptr + group * in_dim) : nullptr;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      const scalar_t normed = (xhat_buf[o] - mean_val) * invstd_val;
      xhat_buf[o] = normed;
      const int64_t slot = o / emb;
      const int64_t dim = o - slot * emb;
      const scalar_t grad_y = grad_rel_ptr[(out_base + slot) * emb + dim];
      if(grad_ln_bias_ptr != nullptr) {
         atomic_add_compat(grad_ln_bias_ptr + group * in_dim + o, grad_y);
      }
      if(grad_ln_weight_ptr != nullptr) {
         atomic_add_compat(grad_ln_weight_ptr + group * in_dim + o, grad_y * normed);
      }
      grad_z2[o] = has_ln_weight ? (grad_y * ln_weight_group[o]) : grad_y;
   }
   __syncthreads();

   if(threadIdx.x == 0) {
      scalar_t sum_dxhat = static_cast< scalar_t >(0);
      scalar_t sum_dxhat_xhat = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         sum_dxhat += grad_z2[o];
         sum_dxhat_xhat += grad_z2[o] * xhat_buf[o];
      }
      mean_dxhat = sum_dxhat / static_cast< scalar_t >(in_dim);
      mean_dxhat_xhat = sum_dxhat_xhat / static_cast< scalar_t >(in_dim);
   }
   __syncthreads();

   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      grad_z2[o] =
         invstd_val * (grad_z2[o] - mean_dxhat - xhat_buf[o] * mean_dxhat_xhat);
      if(grad_b2_ptr != nullptr) {
         atomic_add_compat(grad_b2_ptr + group * in_dim + o, grad_z2[o]);
      }
      const scalar_t go = grad_z2[o];
      for(int64_t h = 0; h < hidden; ++h) {
         atomic_add_compat(
            grad_w2_ptr + (group * in_dim + o) * hidden + h,
            go * hidden_buf[h]
         );
      }
   }
   __syncthreads();

   for(int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
      scalar_t grad_h = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         grad_h += w2_group[o * hidden + h] * grad_z2[o];
      }
      const scalar_t gz1 = grad_h * pointwise_grad_from_pre_activation_compat(pointwise_code, z1[h]);
      grad_z1[h] = gz1;
      if(grad_b1_ptr != nullptr) {
         atomic_add_compat(grad_b1_ptr + group * hidden + h, gz1);
      }
      const scalar_t* input_ptr = input;
      for(int64_t i = 0; i < in_dim; ++i) {
         atomic_add_compat(
            grad_w1_ptr + (group * hidden + h) * in_dim + i,
            gz1 * input_ptr[i]
         );
      }
   }
   __syncthreads();

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      scalar_t grad_in = static_cast< scalar_t >(0);
      for(int64_t h = 0; h < hidden; ++h) {
         grad_in += w1_group[h * in_dim + i] * grad_z1[h];
      }
      const int64_t slot = i / emb;
      const int64_t dim = i - slot * emb;
      const scalar_t grad_residual = grad_rel_ptr[(out_base + slot) * emb + dim];
      const int64_t node = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
      atomic_add_compat(grad_x_ptr + node * emb + dim, grad_in + grad_residual);
   }
}

template < typename scalar_t, typename index_t >
void launch_fused_postnorm_two_layer_pointwise_layernorm_from_indices_backward(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t arity,
   const Tensor& w1_stack,
   const Tensor& b1_stack,
   const Tensor& w2_stack,
   const Tensor& b2_stack,
   const Tensor& ln_weight_stack,
   const Tensor& ln_bias_stack,
   scalar_t ln_eps,
   int64_t pointwise_code,
   Tensor& grad_x,
   Tensor& grad_w1,
   Tensor& grad_b1,
   Tensor& grad_w2,
   Tensor& grad_b2,
   Tensor& grad_ln_weight,
   Tensor& grad_ln_bias,
   cudaStream_t stream
)
{
   const int64_t groups = slot_offsets.size(0);
   if(total_rows <= 0) {
      return;
   }
   const int64_t emb = x.size(1);
   const int64_t hidden = w1_stack.size(1);
   const int64_t in_dim = arity * emb;
   const size_t shared_bytes =
      static_cast< size_t >(3 * in_dim + 3 * hidden) * sizeof(scalar_t);
   const dim3 grid(static_cast< unsigned int >(total_rows));
   fused_postnorm_two_layer_pointwise_layernorm_from_indices_backward_cuda_kernel< scalar_t, index_t >
      <<<grid, static_cast< int >(kThreads), shared_bytes, stream>>>(
         grad_rel.data_ptr< scalar_t >(),
         x.data_ptr< scalar_t >(),
         relation_args.data_ptr< index_t >(),
         slot_offsets.data_ptr< int64_t >(),
         row_offsets.data_ptr< int64_t >(),
         out_offsets.data_ptr< int64_t >(),
         groups,
         arity,
         emb,
         w1_stack.data_ptr< scalar_t >(),
         b1_stack.numel() > 0 ? b1_stack.data_ptr< scalar_t >() : nullptr,
         b1_stack.numel() > 0,
         hidden,
         w2_stack.data_ptr< scalar_t >(),
         b2_stack.numel() > 0 ? b2_stack.data_ptr< scalar_t >() : nullptr,
         b2_stack.numel() > 0,
         ln_weight_stack.numel() > 0 ? ln_weight_stack.data_ptr< scalar_t >() : nullptr,
         ln_bias_stack.numel() > 0 ? ln_bias_stack.data_ptr< scalar_t >() : nullptr,
         ln_weight_stack.numel() > 0,
         ln_bias_stack.numel() > 0,
         ln_eps,
         pointwise_code,
         grad_x.data_ptr< scalar_t >(),
         grad_w1.data_ptr< scalar_t >(),
         grad_b1.numel() > 0 ? grad_b1.data_ptr< scalar_t >() : nullptr,
         grad_w2.data_ptr< scalar_t >(),
         grad_b2.numel() > 0 ? grad_b2.data_ptr< scalar_t >() : nullptr,
         grad_ln_weight.numel() > 0 ? grad_ln_weight.data_ptr< scalar_t >() : nullptr,
         grad_ln_bias.numel() > 0 ? grad_ln_bias.data_ptr< scalar_t >() : nullptr
      );
   check_kernel_launch(
      "fused_postnorm_two_layer_pointwise_layernorm_from_indices_backward_cuda_kernel"
   );
}

std::tuple< Tensor, Tensor > fused_postnorm_two_layer_pointwise_layernorm_from_indices_cuda(
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t total_slots,
   int64_t arity,
   const Tensor& w1_stack,
   const Tensor& b1_stack,
   const Tensor& w2_stack,
   const Tensor& b2_stack,
   const Tensor& ln_weight_stack,
   const Tensor& ln_bias_stack,
   double ln_eps,
   int64_t pointwise_code
)
{
   c10::cuda::CUDAGuard device_guard(x.device());
   check_same_cuda_device(x, relation_args, "x", "relation_args");
   check_same_cuda_device(x, slot_offsets, "x", "slot_offsets");
   check_same_cuda_device(x, row_offsets, "x", "row_offsets");
   check_same_cuda_device(x, out_offsets, "x", "out_offsets");
   check_same_cuda_device(x, w1_stack, "x", "w1_stack");
   check_same_cuda_device(x, w2_stack, "x", "w2_stack");
   if(b1_stack.numel() > 0) {
      check_same_cuda_device(x, b1_stack, "x", "b1_stack");
   }
   if(b2_stack.numel() > 0) {
      check_same_cuda_device(x, b2_stack, "x", "b2_stack");
   }
   if(ln_weight_stack.numel() > 0) {
      check_same_cuda_device(x, ln_weight_stack, "x", "ln_weight_stack");
   }
   if(ln_bias_stack.numel() > 0) {
      check_same_cuda_device(x, ln_bias_stack, "x", "ln_bias_stack");
   }

   const int64_t emb = x.size(1);
   Tensor rel_cat = at::empty({total_slots, emb}, x.options());
   Tensor node_idx = at::empty({total_slots}, relation_args.options().dtype(at::kLong));
   if(total_slots <= 0) {
      return std::make_tuple(rel_cat, node_idx);
   }

   Tensor x_work = ensure_contiguous(x);
   Tensor relation_args_work = ensure_contiguous(relation_args);
   Tensor slot_offsets_work = ensure_contiguous(slot_offsets);
   Tensor row_offsets_work = ensure_contiguous(row_offsets);
   Tensor out_offsets_work = ensure_contiguous(out_offsets);
   Tensor w1_work = ensure_contiguous(w1_stack);
   Tensor b1_work = b1_stack.numel() > 0 ? ensure_contiguous(b1_stack) : b1_stack;
   Tensor w2_work = ensure_contiguous(w2_stack);
   Tensor b2_work = b2_stack.numel() > 0 ? ensure_contiguous(b2_stack) : b2_stack;
   Tensor ln_weight_work =
      ln_weight_stack.numel() > 0 ? ensure_contiguous(ln_weight_stack) : ln_weight_stack;
   Tensor ln_bias_work = ln_bias_stack.numel() > 0 ? ensure_contiguous(ln_bias_stack) : ln_bias_stack;
   cudaStream_t stream = current_cuda_stream(x_work);

   if(dtype_of(relation_args_work) == at::kInt) {
      dispatch_float_or_double(
         x_work,
         [&]() {
            launch_fused_postnorm_two_layer_pointwise_layernorm_from_indices< float, int >(
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w1_work,
               b1_work,
               w2_work,
               b2_work,
               ln_weight_work,
               ln_bias_work,
               static_cast< float >(ln_eps),
               pointwise_code,
               rel_cat,
               node_idx,
               stream
            );
         },
         [&]() {
            launch_fused_postnorm_two_layer_pointwise_layernorm_from_indices< double, int >(
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w1_work,
               b1_work,
               w2_work,
               b2_work,
               ln_weight_work,
               ln_bias_work,
               static_cast< double >(ln_eps),
               pointwise_code,
               rel_cat,
               node_idx,
               stream
            );
         },
         "fused_postnorm_two_layer_pointwise_layernorm_from_indices_cuda"
      );
      return std::make_tuple(rel_cat, node_idx);
   }

   dispatch_float_or_double(
      x_work,
      [&]() {
         launch_fused_postnorm_two_layer_pointwise_layernorm_from_indices< float, int64_t >(
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w1_work,
            b1_work,
            w2_work,
            b2_work,
            ln_weight_work,
            ln_bias_work,
            static_cast< float >(ln_eps),
            pointwise_code,
            rel_cat,
            node_idx,
            stream
         );
      },
      [&]() {
         launch_fused_postnorm_two_layer_pointwise_layernorm_from_indices< double, int64_t >(
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w1_work,
            b1_work,
            w2_work,
            b2_work,
            ln_weight_work,
            ln_bias_work,
            static_cast< double >(ln_eps),
            pointwise_code,
            rel_cat,
            node_idx,
            stream
         );
      },
      "fused_postnorm_two_layer_pointwise_layernorm_from_indices_cuda"
   );
   return std::make_tuple(rel_cat, node_idx);
}

std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor >
fused_postnorm_two_layer_pointwise_layernorm_from_indices_backward_cuda(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t arity,
   const Tensor& w1_stack,
   const Tensor& b1_stack,
   const Tensor& w2_stack,
   const Tensor& b2_stack,
   const Tensor& ln_weight_stack,
   const Tensor& ln_bias_stack,
   double ln_eps,
   int64_t pointwise_code
)
{
   c10::cuda::CUDAGuard device_guard(grad_rel.device());
   check_same_cuda_device(grad_rel, x, "grad_rel", "x");
   check_same_cuda_device(grad_rel, relation_args, "grad_rel", "relation_args");
   check_same_cuda_device(grad_rel, slot_offsets, "grad_rel", "slot_offsets");
   check_same_cuda_device(grad_rel, row_offsets, "grad_rel", "row_offsets");
   check_same_cuda_device(grad_rel, out_offsets, "grad_rel", "out_offsets");
   check_same_cuda_device(grad_rel, w1_stack, "grad_rel", "w1_stack");
   check_same_cuda_device(grad_rel, w2_stack, "grad_rel", "w2_stack");
   if(b1_stack.numel() > 0) {
      check_same_cuda_device(grad_rel, b1_stack, "grad_rel", "b1_stack");
   }
   if(b2_stack.numel() > 0) {
      check_same_cuda_device(grad_rel, b2_stack, "grad_rel", "b2_stack");
   }
   if(ln_weight_stack.numel() > 0) {
      check_same_cuda_device(grad_rel, ln_weight_stack, "grad_rel", "ln_weight_stack");
   }
   if(ln_bias_stack.numel() > 0) {
      check_same_cuda_device(grad_rel, ln_bias_stack, "grad_rel", "ln_bias_stack");
   }

   Tensor grad_x = at::zeros_like(x);
   Tensor grad_w1 = at::zeros_like(w1_stack);
   Tensor grad_b1 = b1_stack.numel() > 0 ? at::zeros_like(b1_stack) : at::zeros_like(b1_stack);
   Tensor grad_w2 = at::zeros_like(w2_stack);
   Tensor grad_b2 = b2_stack.numel() > 0 ? at::zeros_like(b2_stack) : at::zeros_like(b2_stack);
   Tensor grad_ln_weight =
      ln_weight_stack.numel() > 0 ? at::zeros_like(ln_weight_stack) : at::zeros_like(ln_weight_stack);
   Tensor grad_ln_bias =
      ln_bias_stack.numel() > 0 ? at::zeros_like(ln_bias_stack) : at::zeros_like(ln_bias_stack);
   if(total_rows <= 0) {
      return std::make_tuple(
         grad_x, grad_w1, grad_b1, grad_w2, grad_b2, grad_ln_weight, grad_ln_bias
      );
   }

   Tensor grad_rel_work = ensure_contiguous(grad_rel);
   Tensor x_work = ensure_contiguous(x);
   Tensor relation_args_work = ensure_contiguous(relation_args);
   Tensor slot_offsets_work = ensure_contiguous(slot_offsets);
   Tensor row_offsets_work = ensure_contiguous(row_offsets);
   Tensor out_offsets_work = ensure_contiguous(out_offsets);
   Tensor w1_work = ensure_contiguous(w1_stack);
   Tensor b1_work = b1_stack.numel() > 0 ? ensure_contiguous(b1_stack) : b1_stack;
   Tensor w2_work = ensure_contiguous(w2_stack);
   Tensor b2_work = b2_stack.numel() > 0 ? ensure_contiguous(b2_stack) : b2_stack;
   Tensor ln_weight_work =
      ln_weight_stack.numel() > 0 ? ensure_contiguous(ln_weight_stack) : ln_weight_stack;
   Tensor ln_bias_work = ln_bias_stack.numel() > 0 ? ensure_contiguous(ln_bias_stack) : ln_bias_stack;
   cudaStream_t stream = current_cuda_stream(grad_rel_work);

   if(dtype_of(relation_args_work) == at::kInt) {
      dispatch_float_or_double(
         grad_rel_work,
         [&]() {
            launch_fused_postnorm_two_layer_pointwise_layernorm_from_indices_backward<
               float,
               int >(
               grad_rel_work,
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w1_work,
               b1_work,
               w2_work,
               b2_work,
               ln_weight_work,
               ln_bias_work,
               static_cast< float >(ln_eps),
               pointwise_code,
               grad_x,
               grad_w1,
               grad_b1,
               grad_w2,
               grad_b2,
               grad_ln_weight,
               grad_ln_bias,
               stream
            );
         },
         [&]() {
            launch_fused_postnorm_two_layer_pointwise_layernorm_from_indices_backward<
               double,
               int >(
               grad_rel_work,
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               w1_work,
               b1_work,
               w2_work,
               b2_work,
               ln_weight_work,
               ln_bias_work,
               static_cast< double >(ln_eps),
               pointwise_code,
               grad_x,
               grad_w1,
               grad_b1,
               grad_w2,
               grad_b2,
               grad_ln_weight,
               grad_ln_bias,
               stream
            );
         },
         "fused_postnorm_two_layer_pointwise_layernorm_from_indices_backward_cuda"
      );
      return std::make_tuple(
         grad_x, grad_w1, grad_b1, grad_w2, grad_b2, grad_ln_weight, grad_ln_bias
      );
   }

   dispatch_float_or_double(
      grad_rel_work,
      [&]() {
         launch_fused_postnorm_two_layer_pointwise_layernorm_from_indices_backward<
            float,
            int64_t >(
            grad_rel_work,
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w1_work,
            b1_work,
            w2_work,
            b2_work,
            ln_weight_work,
            ln_bias_work,
            static_cast< float >(ln_eps),
            pointwise_code,
            grad_x,
            grad_w1,
            grad_b1,
            grad_w2,
            grad_b2,
            grad_ln_weight,
            grad_ln_bias,
            stream
         );
      },
      [&]() {
         launch_fused_postnorm_two_layer_pointwise_layernorm_from_indices_backward<
            double,
            int64_t >(
            grad_rel_work,
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            w1_work,
            b1_work,
            w2_work,
            b2_work,
            ln_weight_work,
            ln_bias_work,
            static_cast< double >(ln_eps),
            pointwise_code,
            grad_x,
            grad_w1,
            grad_b1,
            grad_w2,
            grad_b2,
            grad_ln_weight,
            grad_ln_bias,
            stream
         );
      },
      "fused_postnorm_two_layer_pointwise_layernorm_from_indices_backward_cuda"
   );
   return std::make_tuple(
      grad_x, grad_w1, grad_b1, grad_w2, grad_b2, grad_ln_weight, grad_ln_bias
   );
}

template < typename scalar_t, typename index_t >
__global__ void fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_cuda_kernel(
   const scalar_t* x_ptr,
   const index_t* relation_args_ptr,
   const int64_t* slot_offsets_ptr,
   const int64_t* row_offsets_ptr,
   const int64_t* out_offsets_ptr,
   int64_t groups,
   int64_t arity,
   int64_t emb,
   const scalar_t* rms_weight_ptr,
   bool has_rms_weight,
   scalar_t rms_eps,
   const scalar_t* w1_ptr,
   const scalar_t* b1_ptr,
   bool has_b1,
   int64_t hidden,
   const scalar_t* w2_ptr,
   const scalar_t* b2_ptr,
   bool has_b2,
   int64_t pointwise_code,
   scalar_t* out_ptr,
   int64_t* node_idx_ptr
)
{
   const int64_t row = static_cast< int64_t >(blockIdx.x);
   if(row >= row_offsets_ptr[groups]) {
      return;
   }

   int64_t group = 0;
   while(group + 1 < groups && row >= row_offsets_ptr[group + 1]) {
      ++group;
   }
   const int64_t row_in_group = row - row_offsets_ptr[group];
   const int64_t slot_base = slot_offsets_ptr[group] + row_in_group * arity;
   const int64_t out_base = out_offsets_ptr[group] + row_in_group * arity;
   const int64_t in_dim = arity * emb;

   extern __shared__ unsigned char smem_raw[];
   scalar_t* input = reinterpret_cast< scalar_t* >(smem_raw);
   scalar_t* norm_buf = input + in_dim;
   scalar_t* hidden_buf = norm_buf + in_dim;
   __shared__ scalar_t inv_rms_val;

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      const int64_t slot = i / emb;
      const int64_t dim = i - slot * emb;
      const int64_t node = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
      input[i] = x_ptr[node * emb + dim];
   }
   for(int64_t slot = threadIdx.x; slot < arity; slot += blockDim.x) {
      node_idx_ptr[out_base + slot] = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
   }
   __syncthreads();

   if(threadIdx.x == 0) {
      scalar_t sq_mean = static_cast< scalar_t >(0);
      for(int64_t i = 0; i < in_dim; ++i) {
         sq_mean += input[i] * input[i];
      }
      sq_mean /= static_cast< scalar_t >(in_dim);
      inv_rms_val = static_cast< scalar_t >(1) / sqrt(sq_mean + rms_eps);
   }
   __syncthreads();

   const scalar_t* rms_weight_group = has_rms_weight ? (rms_weight_ptr + group * in_dim) : nullptr;
   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      scalar_t normed = input[i] * inv_rms_val;
      if(has_rms_weight) {
         normed *= rms_weight_group[i];
      }
      norm_buf[i] = normed;
   }
   __syncthreads();

   const scalar_t* w1_group = w1_ptr + group * hidden * in_dim;
   const scalar_t* b1_group = has_b1 ? (b1_ptr + group * hidden) : nullptr;
   for(int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
      scalar_t acc = has_b1 ? b1_group[h] : static_cast< scalar_t >(0);
      const scalar_t* w_row = w1_group + h * in_dim;
      for(int64_t i = 0; i < in_dim; ++i) {
         acc += w_row[i] * norm_buf[i];
      }
      hidden_buf[h] = pointwise_apply_compat(pointwise_code, acc);
   }
   __syncthreads();

   const scalar_t* w2_group = w2_ptr + group * in_dim * hidden;
   const scalar_t* b2_group = has_b2 ? (b2_ptr + group * in_dim) : nullptr;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      scalar_t acc = has_b2 ? b2_group[o] : static_cast< scalar_t >(0);
      const scalar_t* w_row = w2_group + o * hidden;
      for(int64_t h = 0; h < hidden; ++h) {
         acc += w_row[h] * hidden_buf[h];
      }
      const int64_t slot = o / emb;
      const int64_t dim = o - slot * emb;
      out_ptr[(out_base + slot) * emb + dim] = input[o] + acc;
   }
}

template < typename scalar_t, typename index_t >
void launch_fused_prenorm_two_layer_pointwise_rmsnorm_from_indices(
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t arity,
   const Tensor& rms_weight_stack,
   scalar_t rms_eps,
   const Tensor& w1_stack,
   const Tensor& b1_stack,
   const Tensor& w2_stack,
   const Tensor& b2_stack,
   int64_t pointwise_code,
   Tensor& rel_cat,
   Tensor& node_idx,
   cudaStream_t stream
)
{
   const int64_t groups = slot_offsets.size(0);
   if(total_rows <= 0) {
      return;
   }
   const int64_t emb = x.size(1);
   const int64_t hidden = w1_stack.size(1);
   const int64_t in_dim = arity * emb;
   const size_t shared_bytes =
      static_cast< size_t >(2 * in_dim + hidden) * sizeof(scalar_t);
   const dim3 grid(static_cast< unsigned int >(total_rows));
   fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_cuda_kernel< scalar_t, index_t >
      <<<grid, static_cast< int >(kThreads), shared_bytes, stream>>>(
         x.data_ptr< scalar_t >(),
         relation_args.data_ptr< index_t >(),
         slot_offsets.data_ptr< int64_t >(),
         row_offsets.data_ptr< int64_t >(),
         out_offsets.data_ptr< int64_t >(),
         groups,
         arity,
         emb,
         rms_weight_stack.numel() > 0 ? rms_weight_stack.data_ptr< scalar_t >() : nullptr,
         rms_weight_stack.numel() > 0,
         rms_eps,
         w1_stack.data_ptr< scalar_t >(),
         b1_stack.numel() > 0 ? b1_stack.data_ptr< scalar_t >() : nullptr,
         b1_stack.numel() > 0,
         hidden,
         w2_stack.data_ptr< scalar_t >(),
         b2_stack.numel() > 0 ? b2_stack.data_ptr< scalar_t >() : nullptr,
         b2_stack.numel() > 0,
         pointwise_code,
         rel_cat.data_ptr< scalar_t >(),
         node_idx.data_ptr< int64_t >()
      );
   check_kernel_launch("fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_cuda_kernel");
}

template < typename scalar_t, typename index_t >
__global__ void fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_backward_cuda_kernel(
   const scalar_t* grad_rel_ptr,
   const scalar_t* x_ptr,
   const index_t* relation_args_ptr,
   const int64_t* slot_offsets_ptr,
   const int64_t* row_offsets_ptr,
   const int64_t* out_offsets_ptr,
   int64_t groups,
   int64_t arity,
   int64_t emb,
   const scalar_t* rms_weight_ptr,
   bool has_rms_weight,
   scalar_t rms_eps,
   const scalar_t* w1_ptr,
   const scalar_t* b1_ptr,
   bool has_b1,
   int64_t hidden,
   const scalar_t* w2_ptr,
   const scalar_t* b2_ptr,
   bool has_b2,
   int64_t pointwise_code,
   scalar_t* grad_x_ptr,
   scalar_t* grad_rms_weight_ptr,
   scalar_t* grad_w1_ptr,
   scalar_t* grad_b1_ptr,
   scalar_t* grad_w2_ptr,
   scalar_t* grad_b2_ptr
)
{
   const int64_t row = static_cast< int64_t >(blockIdx.x);
   if(row >= row_offsets_ptr[groups]) {
      return;
   }

   int64_t group = 0;
   while(group + 1 < groups && row >= row_offsets_ptr[group + 1]) {
      ++group;
   }
   const int64_t row_in_group = row - row_offsets_ptr[group];
   const int64_t slot_base = slot_offsets_ptr[group] + row_in_group * arity;
   const int64_t out_base = out_offsets_ptr[group] + row_in_group * arity;
   const int64_t in_dim = arity * emb;

   extern __shared__ unsigned char smem_raw[];
   scalar_t* input = reinterpret_cast< scalar_t* >(smem_raw);
   scalar_t* norm_buf = input + in_dim;
   scalar_t* z1 = norm_buf + in_dim;
   scalar_t* hidden_buf = z1 + hidden;
   scalar_t* grad_norm = hidden_buf + hidden;
   scalar_t* grad_z1 = grad_norm + in_dim;
   __shared__ scalar_t inv_rms_val;
   __shared__ scalar_t mean_grad_norm_xhat;

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      const int64_t slot = i / emb;
      const int64_t dim = i - slot * emb;
      const int64_t node = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
      input[i] = x_ptr[node * emb + dim];
   }
   __syncthreads();

   if(threadIdx.x == 0) {
      scalar_t sq_mean = static_cast< scalar_t >(0);
      for(int64_t i = 0; i < in_dim; ++i) {
         sq_mean += input[i] * input[i];
      }
      sq_mean /= static_cast< scalar_t >(in_dim);
      inv_rms_val = static_cast< scalar_t >(1) / sqrt(sq_mean + rms_eps);
   }
   __syncthreads();

   const scalar_t* rms_weight_group = has_rms_weight ? (rms_weight_ptr + group * in_dim) : nullptr;
   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      scalar_t xhat = input[i] * inv_rms_val;
      norm_buf[i] = xhat;
      scalar_t normed = has_rms_weight ? (xhat * rms_weight_group[i]) : xhat;
      norm_buf[i] = normed;
   }
   __syncthreads();

   const scalar_t* w1_group = w1_ptr + group * hidden * in_dim;
   const scalar_t* b1_group = has_b1 ? (b1_ptr + group * hidden) : nullptr;
   for(int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
      scalar_t acc = has_b1 ? b1_group[h] : static_cast< scalar_t >(0);
      const scalar_t* w_row = w1_group + h * in_dim;
      for(int64_t i = 0; i < in_dim; ++i) {
         acc += w_row[i] * norm_buf[i];
      }
      z1[h] = acc;
      hidden_buf[h] = pointwise_apply_compat(pointwise_code, acc);
   }
   __syncthreads();

   const scalar_t* w2_group = w2_ptr + group * in_dim * hidden;
   for(int64_t o = threadIdx.x; o < in_dim; o += blockDim.x) {
      if(grad_b2_ptr != nullptr) {
         const int64_t slot = o / emb;
         const int64_t dim = o - slot * emb;
         atomic_add_compat(
            grad_b2_ptr + group * in_dim + o,
            grad_rel_ptr[(out_base + slot) * emb + dim]
         );
      }
      const int64_t slot = o / emb;
      const int64_t dim = o - slot * emb;
      const scalar_t go = grad_rel_ptr[(out_base + slot) * emb + dim];
      for(int64_t h = 0; h < hidden; ++h) {
         atomic_add_compat(
            grad_w2_ptr + (group * in_dim + o) * hidden + h,
            go * hidden_buf[h]
         );
      }
   }
   __syncthreads();

   for(int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
      scalar_t grad_h = static_cast< scalar_t >(0);
      for(int64_t o = 0; o < in_dim; ++o) {
         const int64_t slot = o / emb;
         const int64_t dim = o - slot * emb;
         grad_h += w2_group[o * hidden + h] * grad_rel_ptr[(out_base + slot) * emb + dim];
      }
      const scalar_t gz1 = grad_h * pointwise_grad_from_pre_activation_compat(pointwise_code, z1[h]);
      grad_z1[h] = gz1;
      if(grad_b1_ptr != nullptr) {
         atomic_add_compat(grad_b1_ptr + group * hidden + h, gz1);
      }
      for(int64_t i = 0; i < in_dim; ++i) {
         atomic_add_compat(
            grad_w1_ptr + (group * hidden + h) * in_dim + i,
            gz1 * norm_buf[i]
         );
      }
   }
   __syncthreads();

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      scalar_t grad_i = static_cast< scalar_t >(0);
      for(int64_t h = 0; h < hidden; ++h) {
         grad_i += w1_group[h * in_dim + i] * grad_z1[h];
      }
      if(grad_rms_weight_ptr != nullptr) {
         atomic_add_compat(grad_rms_weight_ptr + group * in_dim + i, grad_i * input[i] * inv_rms_val);
      }
      grad_norm[i] = has_rms_weight ? (grad_i * rms_weight_group[i]) : grad_i;
   }
   __syncthreads();

   if(threadIdx.x == 0) {
      scalar_t accum = static_cast< scalar_t >(0);
      for(int64_t i = 0; i < in_dim; ++i) {
         const scalar_t xhat = input[i] * inv_rms_val;
         accum += grad_norm[i] * xhat;
      }
      mean_grad_norm_xhat = accum / static_cast< scalar_t >(in_dim);
   }
   __syncthreads();

   for(int64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
      const scalar_t xhat = input[i] * inv_rms_val;
      const scalar_t grad_in_norm =
         inv_rms_val * (grad_norm[i] - xhat * mean_grad_norm_xhat);
      const int64_t slot = i / emb;
      const int64_t dim = i - slot * emb;
      const scalar_t grad_residual = grad_rel_ptr[(out_base + slot) * emb + dim];
      const int64_t node = static_cast< int64_t >(relation_args_ptr[slot_base + slot]);
      atomic_add_compat(grad_x_ptr + node * emb + dim, grad_in_norm + grad_residual);
   }
}

template < typename scalar_t, typename index_t >
void launch_fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_backward(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t arity,
   const Tensor& rms_weight_stack,
   scalar_t rms_eps,
   const Tensor& w1_stack,
   const Tensor& b1_stack,
   const Tensor& w2_stack,
   const Tensor& b2_stack,
   int64_t pointwise_code,
   Tensor& grad_x,
   Tensor& grad_rms_weight,
   Tensor& grad_w1,
   Tensor& grad_b1,
   Tensor& grad_w2,
   Tensor& grad_b2,
   cudaStream_t stream
)
{
   const int64_t groups = slot_offsets.size(0);
   if(total_rows <= 0) {
      return;
   }
   const int64_t emb = x.size(1);
   const int64_t hidden = w1_stack.size(1);
   const int64_t in_dim = arity * emb;
   const size_t shared_bytes =
      static_cast< size_t >(4 * in_dim + 2 * hidden) * sizeof(scalar_t);
   const dim3 grid(static_cast< unsigned int >(total_rows));
   fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_backward_cuda_kernel< scalar_t, index_t >
      <<<grid, static_cast< int >(kThreads), shared_bytes, stream>>>(
         grad_rel.data_ptr< scalar_t >(),
         x.data_ptr< scalar_t >(),
         relation_args.data_ptr< index_t >(),
         slot_offsets.data_ptr< int64_t >(),
         row_offsets.data_ptr< int64_t >(),
         out_offsets.data_ptr< int64_t >(),
         groups,
         arity,
         emb,
         rms_weight_stack.numel() > 0 ? rms_weight_stack.data_ptr< scalar_t >() : nullptr,
         rms_weight_stack.numel() > 0,
         rms_eps,
         w1_stack.data_ptr< scalar_t >(),
         b1_stack.numel() > 0 ? b1_stack.data_ptr< scalar_t >() : nullptr,
         b1_stack.numel() > 0,
         hidden,
         w2_stack.data_ptr< scalar_t >(),
         b2_stack.numel() > 0 ? b2_stack.data_ptr< scalar_t >() : nullptr,
         b2_stack.numel() > 0,
         pointwise_code,
         grad_x.data_ptr< scalar_t >(),
         grad_rms_weight.numel() > 0 ? grad_rms_weight.data_ptr< scalar_t >() : nullptr,
         grad_w1.data_ptr< scalar_t >(),
         grad_b1.numel() > 0 ? grad_b1.data_ptr< scalar_t >() : nullptr,
         grad_w2.data_ptr< scalar_t >(),
         grad_b2.numel() > 0 ? grad_b2.data_ptr< scalar_t >() : nullptr
      );
   check_kernel_launch("fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_backward_cuda_kernel");
}

std::tuple< Tensor, Tensor > fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_cuda(
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t total_slots,
   int64_t arity,
   const Tensor& rms_weight_stack,
   double rms_eps,
   const Tensor& w1_stack,
   const Tensor& b1_stack,
   const Tensor& w2_stack,
   const Tensor& b2_stack,
   int64_t pointwise_code
)
{
   c10::cuda::CUDAGuard device_guard(x.device());
   check_same_cuda_device(x, relation_args, "x", "relation_args");
   check_same_cuda_device(x, slot_offsets, "x", "slot_offsets");
   check_same_cuda_device(x, row_offsets, "x", "row_offsets");
   check_same_cuda_device(x, out_offsets, "x", "out_offsets");
   check_same_cuda_device(x, w1_stack, "x", "w1_stack");
   check_same_cuda_device(x, w2_stack, "x", "w2_stack");
   if(rms_weight_stack.numel() > 0) {
      check_same_cuda_device(x, rms_weight_stack, "x", "rms_weight_stack");
   }
   if(b1_stack.numel() > 0) {
      check_same_cuda_device(x, b1_stack, "x", "b1_stack");
   }
   if(b2_stack.numel() > 0) {
      check_same_cuda_device(x, b2_stack, "x", "b2_stack");
   }

   const int64_t emb = x.size(1);
   Tensor rel_cat = at::empty({total_slots, emb}, x.options());
   Tensor node_idx = at::empty({total_slots}, relation_args.options().dtype(at::kLong));
   if(total_slots <= 0) {
      return std::make_tuple(rel_cat, node_idx);
   }

   Tensor x_work = ensure_contiguous(x);
   Tensor relation_args_work = ensure_contiguous(relation_args);
   Tensor slot_offsets_work = ensure_contiguous(slot_offsets);
   Tensor row_offsets_work = ensure_contiguous(row_offsets);
   Tensor out_offsets_work = ensure_contiguous(out_offsets);
   Tensor rms_weight_work =
      rms_weight_stack.numel() > 0 ? ensure_contiguous(rms_weight_stack) : rms_weight_stack;
   Tensor w1_work = ensure_contiguous(w1_stack);
   Tensor b1_work = b1_stack.numel() > 0 ? ensure_contiguous(b1_stack) : b1_stack;
   Tensor w2_work = ensure_contiguous(w2_stack);
   Tensor b2_work = b2_stack.numel() > 0 ? ensure_contiguous(b2_stack) : b2_stack;
   cudaStream_t stream = current_cuda_stream(x_work);

   if(dtype_of(relation_args_work) == at::kInt) {
      dispatch_float_or_double(
         x_work,
         [&]() {
            launch_fused_prenorm_two_layer_pointwise_rmsnorm_from_indices< float, int >(
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               rms_weight_work,
               static_cast< float >(rms_eps),
               w1_work,
               b1_work,
               w2_work,
               b2_work,
               pointwise_code,
               rel_cat,
               node_idx,
               stream
            );
         },
         [&]() {
            launch_fused_prenorm_two_layer_pointwise_rmsnorm_from_indices< double, int >(
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               rms_weight_work,
               static_cast< double >(rms_eps),
               w1_work,
               b1_work,
               w2_work,
               b2_work,
               pointwise_code,
               rel_cat,
               node_idx,
               stream
            );
         },
         "fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_cuda"
      );
      return std::make_tuple(rel_cat, node_idx);
   }

   dispatch_float_or_double(
      x_work,
      [&]() {
         launch_fused_prenorm_two_layer_pointwise_rmsnorm_from_indices< float, int64_t >(
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            rms_weight_work,
            static_cast< float >(rms_eps),
            w1_work,
            b1_work,
            w2_work,
            b2_work,
            pointwise_code,
            rel_cat,
            node_idx,
            stream
         );
      },
      [&]() {
         launch_fused_prenorm_two_layer_pointwise_rmsnorm_from_indices< double, int64_t >(
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            rms_weight_work,
            static_cast< double >(rms_eps),
            w1_work,
            b1_work,
            w2_work,
            b2_work,
            pointwise_code,
            rel_cat,
            node_idx,
            stream
         );
      },
      "fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_cuda"
   );
   return std::make_tuple(rel_cat, node_idx);
}

std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor, Tensor >
fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_backward_cuda(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const Tensor& slot_offsets,
   const Tensor& row_offsets,
   const Tensor& out_offsets,
   int64_t total_rows,
   int64_t arity,
   const Tensor& rms_weight_stack,
   double rms_eps,
   const Tensor& w1_stack,
   const Tensor& b1_stack,
   const Tensor& w2_stack,
   const Tensor& b2_stack,
   int64_t pointwise_code
)
{
   c10::cuda::CUDAGuard device_guard(grad_rel.device());
   check_same_cuda_device(grad_rel, x, "grad_rel", "x");
   check_same_cuda_device(grad_rel, relation_args, "grad_rel", "relation_args");
   check_same_cuda_device(grad_rel, slot_offsets, "grad_rel", "slot_offsets");
   check_same_cuda_device(grad_rel, row_offsets, "grad_rel", "row_offsets");
   check_same_cuda_device(grad_rel, out_offsets, "grad_rel", "out_offsets");
   check_same_cuda_device(grad_rel, w1_stack, "grad_rel", "w1_stack");
   check_same_cuda_device(grad_rel, w2_stack, "grad_rel", "w2_stack");
   if(rms_weight_stack.numel() > 0) {
      check_same_cuda_device(grad_rel, rms_weight_stack, "grad_rel", "rms_weight_stack");
   }
   if(b1_stack.numel() > 0) {
      check_same_cuda_device(grad_rel, b1_stack, "grad_rel", "b1_stack");
   }
   if(b2_stack.numel() > 0) {
      check_same_cuda_device(grad_rel, b2_stack, "grad_rel", "b2_stack");
   }

   Tensor grad_x = at::zeros_like(x);
   Tensor grad_rms_weight =
      rms_weight_stack.numel() > 0 ? at::zeros_like(rms_weight_stack) : at::zeros_like(rms_weight_stack);
   Tensor grad_w1 = at::zeros_like(w1_stack);
   Tensor grad_b1 = b1_stack.numel() > 0 ? at::zeros_like(b1_stack) : at::zeros_like(b1_stack);
   Tensor grad_w2 = at::zeros_like(w2_stack);
   Tensor grad_b2 = b2_stack.numel() > 0 ? at::zeros_like(b2_stack) : at::zeros_like(b2_stack);
   if(total_rows <= 0) {
      return std::make_tuple(grad_x, grad_rms_weight, grad_w1, grad_b1, grad_w2, grad_b2);
   }

   Tensor grad_rel_work = ensure_contiguous(grad_rel);
   Tensor x_work = ensure_contiguous(x);
   Tensor relation_args_work = ensure_contiguous(relation_args);
   Tensor slot_offsets_work = ensure_contiguous(slot_offsets);
   Tensor row_offsets_work = ensure_contiguous(row_offsets);
   Tensor out_offsets_work = ensure_contiguous(out_offsets);
   Tensor rms_weight_work =
      rms_weight_stack.numel() > 0 ? ensure_contiguous(rms_weight_stack) : rms_weight_stack;
   Tensor w1_work = ensure_contiguous(w1_stack);
   Tensor b1_work = b1_stack.numel() > 0 ? ensure_contiguous(b1_stack) : b1_stack;
   Tensor w2_work = ensure_contiguous(w2_stack);
   Tensor b2_work = b2_stack.numel() > 0 ? ensure_contiguous(b2_stack) : b2_stack;
   cudaStream_t stream = current_cuda_stream(grad_rel_work);

   if(dtype_of(relation_args_work) == at::kInt) {
      dispatch_float_or_double(
         grad_rel_work,
         [&]() {
            launch_fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_backward< float, int >(
               grad_rel_work,
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               rms_weight_work,
               static_cast< float >(rms_eps),
               w1_work,
               b1_work,
               w2_work,
               b2_work,
               pointwise_code,
               grad_x,
               grad_rms_weight,
               grad_w1,
               grad_b1,
               grad_w2,
               grad_b2,
               stream
            );
         },
         [&]() {
            launch_fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_backward< double, int >(
               grad_rel_work,
               x_work,
               relation_args_work,
               slot_offsets_work,
               row_offsets_work,
               out_offsets_work,
               total_rows,
               arity,
               rms_weight_work,
               static_cast< double >(rms_eps),
               w1_work,
               b1_work,
               w2_work,
               b2_work,
               pointwise_code,
               grad_x,
               grad_rms_weight,
               grad_w1,
               grad_b1,
               grad_w2,
               grad_b2,
               stream
            );
         },
         "fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_backward_cuda"
      );
      return std::make_tuple(grad_x, grad_rms_weight, grad_w1, grad_b1, grad_w2, grad_b2);
   }

   dispatch_float_or_double(
      grad_rel_work,
      [&]() {
         launch_fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_backward< float, int64_t >(
            grad_rel_work,
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            rms_weight_work,
            static_cast< float >(rms_eps),
            w1_work,
            b1_work,
            w2_work,
            b2_work,
            pointwise_code,
            grad_x,
            grad_rms_weight,
            grad_w1,
            grad_b1,
            grad_w2,
            grad_b2,
            stream
         );
      },
      [&]() {
         launch_fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_backward< double, int64_t >(
            grad_rel_work,
            x_work,
            relation_args_work,
            slot_offsets_work,
            row_offsets_work,
            out_offsets_work,
            total_rows,
            arity,
            rms_weight_work,
            static_cast< double >(rms_eps),
            w1_work,
            b1_work,
            w2_work,
            b2_work,
            pointwise_code,
            grad_x,
            grad_rms_weight,
            grad_w1,
            grad_b1,
            grad_w2,
            grad_b2,
            stream
         );
      },
      "fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_backward_cuda"
   );
   return std::make_tuple(grad_x, grad_rms_weight, grad_w1, grad_b1, grad_w2, grad_b2);
}

}  // namespace relm::mp
