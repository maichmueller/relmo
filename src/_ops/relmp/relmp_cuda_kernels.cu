#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace relm::relmp {

using torch::stable::Tensor;

namespace {

constexpr int64_t kThreads = 256;

int32_t dtype_of(const Tensor& t)
{
   int32_t dtype = 0;
   TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(t.get(), &dtype));
   return dtype;
}

bool is_fastpath_dtype(int32_t dtype)
{
   return dtype == aoti_torch_dtype_float32() || dtype == aoti_torch_dtype_float64();
}

Tensor ensure_contiguous(const Tensor& t)
{
   return t.is_contiguous() ? t : torch::stable::contiguous(t);
}

void check_same_cuda_device(
   const Tensor& lhs,
   const Tensor& rhs,
   const char* lhs_name,
   const char* rhs_name
)
{
   STD_TORCH_CHECK(
      lhs.get_device_index() == rhs.get_device_index(),
      lhs_name,
      " and ",
      rhs_name,
      " must be on the same CUDA device."
   );
}

cudaStream_t current_cuda_stream(const Tensor& t)
{
   void* raw_stream = nullptr;
   TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(static_cast< int32_t >(t.get_device_index()), &raw_stream)
   );
   return static_cast< cudaStream_t >(raw_stream);
}

void check_kernel_launch(const char* kernel_name)
{
   const cudaError_t err = cudaGetLastError();
   STD_TORCH_CHECK(err == cudaSuccess, kernel_name, " launch failed: ", cudaGetErrorString(err));
}

int grid_for(int64_t total)
{
   const int64_t blocks = (total + kThreads - 1) / kThreads;
   return static_cast< int >(
      std::min< int64_t >(blocks, static_cast< int64_t >(std::numeric_limits< int >::max()))
   );
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
__global__ void fanout_scatter_backward_cuda_kernel(
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
      < < < blocks, static_cast< int >(kThreads), 0, stream > > >(
         x.const_data_ptr< scalar_t >(),
         src.const_data_ptr< int64_t >(),
         dst.const_data_ptr< int64_t >(),
         num_edges,
         emb,
         out.mutable_data_ptr< scalar_t >()
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
   const int64_t total = num_edges * emb;
   const int blocks = grid_for(total);
   fanout_scatter_backward_cuda_kernel< scalar_t >
      < < < blocks, static_cast< int >(kThreads), 0, stream > > >(
         grad_out.const_data_ptr< scalar_t >(),
         src.const_data_ptr< int64_t >(),
         dst.const_data_ptr< int64_t >(),
         num_edges,
         emb,
         grad_x.mutable_data_ptr< scalar_t >()
      );
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
      < < < blocks, static_cast< int >(kThreads), 0, stream > > >(
         rel.const_data_ptr< scalar_t >(),
         src.const_data_ptr< int64_t >(),
         dst.const_data_ptr< int64_t >(),
         num_edges,
         emb,
         out.mutable_data_ptr< scalar_t >()
      );
   check_kernel_launch("fanin_reduce_sum_cuda_kernel");
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
      < < < blocks, static_cast< int >(kThreads), 0, stream > > >(
         grad_out.const_data_ptr< scalar_t >(),
         src.const_data_ptr< int64_t >(),
         dst.const_data_ptr< int64_t >(),
         num_edges,
         emb,
         grad_rel.mutable_data_ptr< scalar_t >()
      );
   check_kernel_launch("fanin_reduce_sum_backward_cuda_kernel");
}

template < typename FnFloat, typename FnDouble >
void dispatch_float_or_double(
   const Tensor& t,
   FnFloat&& fn_float,
   FnDouble&& fn_double,
   const char* opname
)
{
   const int32_t dtype = dtype_of(t);
   STD_TORCH_CHECK(is_fastpath_dtype(dtype), opname, " supports only float32/float64.");
   if(dtype == aoti_torch_dtype_float32()) {
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
   std::vector< int64_t > shape{out_rows, emb};
   Tensor out = torch::stable::new_zeros(x_cat, shape);
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
   std::vector< int64_t > shape{x_rows, emb};
   Tensor grad_x = torch::stable::new_zeros(grad_out, shape);
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
   std::vector< int64_t > shape{dim_size, emb};
   Tensor out = torch::stable::new_zeros(rel_flat, shape);
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
   std::vector< int64_t > shape{rel_rows, emb};
   Tensor grad_rel = torch::stable::new_zeros(grad_out, shape);
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

}  // namespace relm::relmp
