#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
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

}  // namespace

Tensor fanout_scatter_cuda(
   const Tensor& x_cat,
   const Tensor& src_global_idx,
   const Tensor& flat_dst,
   int64_t out_rows
)
{
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

}  // namespace relm::mp
