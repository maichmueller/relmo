#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace relm::relmp {

using ::StableIValue;
using torch::stable::Tensor;
using torch::stable::detail::from;
using torch::stable::detail::to;

constexpr int64_t kModeSum = 0;
constexpr int64_t kModeLogSumExp = 1;

template < size_t N >
void call_dispatcher(const char* opname, const char* overload, std::array< StableIValue, N >& stack)
{
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
   TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(opname, overload, stack.data(), TORCH_ABI_VERSION));
#else
   TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher(opname, overload, stack.data()));
#endif
}

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

void check_rank(const Tensor& t, int64_t expected, const char* name)
{
   STD_TORCH_CHECK(
      t.dim() == expected, name, " must be rank ", expected, ", got rank ", t.dim(), "."
   );
}

void check_int64_index(const Tensor& t, const char* name)
{
   check_rank(t, 1, name);
   STD_TORCH_CHECK(dtype_of(t) == aoti_torch_dtype_int64(), name, " must have dtype torch.int64.");
}

void check_in_bounds(int64_t idx, int64_t size, const char* name)
{
   STD_TORCH_CHECK(
      idx >= 0 && idx < size, name, " index out of bounds: ", idx, " not in [0, ", size, ")."
   );
}

Tensor ensure_contiguous(const Tensor& t)
{
   return t.is_contiguous() ? t : torch::stable::contiguous(t);
}

Tensor dispatch_index_select0(const Tensor& self, const Tensor& index)
{
   std::array< StableIValue, 3 > stack{from(self), from(static_cast< int64_t >(0)), from(index)};
   call_dispatcher("aten::index_select", "", stack);
   return to< Tensor >(stack[0]);
}

Tensor dispatch_index_copy0(const Tensor& self, const Tensor& index, const Tensor& source)
{
   std::array< StableIValue, 4 > stack{
      from(self), from(static_cast< int64_t >(0)), from(index), from(source)
   };
   call_dispatcher("aten::index_copy", "", stack);
   return to< Tensor >(stack[0]);
}

Tensor dispatch_expand(const Tensor& self, const std::vector< int64_t >& size)
{
   std::array< StableIValue, 3 > stack{from(self), from(size), from(false)};
   call_dispatcher("aten::expand", "", stack);
   return to< Tensor >(stack[0]);
}

Tensor dispatch_scatter_reduce0(
   const Tensor& self,
   const Tensor& index,
   const Tensor& src,
   const char* reduce,
   bool include_self
)
{
   std::array< StableIValue, 6 > stack{
      from(self),
      from(static_cast< int64_t >(0)),
      from(index),
      from(src),
      from(std::string(reduce)),
      from(include_self)
   };
   call_dispatcher("aten::scatter_reduce", "two", stack);
   return to< Tensor >(stack[0]);
}

Tensor dispatch_exp(const Tensor& self)
{
   std::array< StableIValue, 1 > stack{from(self)};
   call_dispatcher("aten::exp", "", stack);
   return to< Tensor >(stack[0]);
}

Tensor dispatch_log(const Tensor& self)
{
   std::array< StableIValue, 1 > stack{from(self)};
   call_dispatcher("aten::log", "", stack);
   return to< Tensor >(stack[0]);
}

Tensor make_scatter_index(const Tensor& dst_idx, int64_t emb)
{
   Tensor unsqueezed = torch::stable::unsqueeze(dst_idx, 1);
   std::vector< int64_t > expanded_shape{dst_idx.size(0), emb};
   return dispatch_expand(unsqueezed, expanded_shape);
}

#if defined(RELM_RELMP_HAS_CUDA) && RELM_RELMP_HAS_CUDA
Tensor fanout_scatter_cuda(
   const Tensor& x_cat,
   const Tensor& src_global_idx,
   const Tensor& flat_dst,
   int64_t out_rows
);
Tensor fanout_scatter_backward_cuda(
   const Tensor& grad_out,
   const Tensor& src_global_idx,
   const Tensor& flat_dst,
   int64_t x_rows
);
Tensor fanin_reduce_sum_cuda(
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t dim_size
);
Tensor fanin_reduce_sum_backward_cuda(
   const Tensor& grad_out,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t rel_rows
);
#endif

template < typename scalar_t >
void fanout_scatter_cpu_kernel(
   const scalar_t* x_ptr,
   const int64_t* src_ptr,
   const int64_t* dst_ptr,
   int64_t num_edges,
   int64_t x_rows,
   int64_t out_rows,
   int64_t emb,
   scalar_t* out_ptr
)
{
   for(int64_t e = 0; e < num_edges; ++e) {
      const int64_t s = src_ptr[e];
      const int64_t d = dst_ptr[e];
      check_in_bounds(s, x_rows, "src_global_idx");
      check_in_bounds(d, out_rows, "flat_dst");
      std::memcpy(
         out_ptr + d * emb, x_ptr + s * emb, static_cast< size_t >(emb) * sizeof(scalar_t)
      );
   }
}

template < typename scalar_t >
void fanout_scatter_backward_cpu_kernel(
   const scalar_t* grad_out_ptr,
   const int64_t* src_ptr,
   const int64_t* dst_ptr,
   int64_t num_edges,
   int64_t x_rows,
   int64_t grad_out_rows,
   int64_t emb,
   scalar_t* grad_x_ptr
)
{
   for(int64_t e = 0; e < num_edges; ++e) {
      const int64_t s = src_ptr[e];
      const int64_t d = dst_ptr[e];
      check_in_bounds(s, x_rows, "src_global_idx");
      check_in_bounds(d, grad_out_rows, "flat_dst");
      const scalar_t* in_row = grad_out_ptr + d * emb;
      scalar_t* out_row = grad_x_ptr + s * emb;
      for(int64_t k = 0; k < emb; ++k) {
         out_row[k] += in_row[k];
      }
   }
}

template < typename scalar_t >
void fanin_reduce_sum_cpu_kernel(
   const scalar_t* rel_ptr,
   const int64_t* flat_src_ptr,
   const int64_t* dst_idx_ptr,
   int64_t num_edges,
   int64_t rel_rows,
   int64_t dim_size,
   int64_t emb,
   scalar_t* out_ptr
)
{
   for(int64_t e = 0; e < num_edges; ++e) {
      const int64_t s = flat_src_ptr[e];
      const int64_t d = dst_idx_ptr[e];
      check_in_bounds(s, rel_rows, "flat_src");
      check_in_bounds(d, dim_size, "dst_idx");
      const scalar_t* in_row = rel_ptr + s * emb;
      scalar_t* out_row = out_ptr + d * emb;
      for(int64_t k = 0; k < emb; ++k) {
         out_row[k] += in_row[k];
      }
   }
}

template < typename scalar_t >
void fanin_reduce_sum_backward_cpu_kernel(
   const scalar_t* grad_out_ptr,
   const int64_t* flat_src_ptr,
   const int64_t* dst_idx_ptr,
   int64_t num_edges,
   int64_t rel_rows,
   int64_t grad_out_rows,
   int64_t emb,
   scalar_t* grad_rel_ptr
)
{
   for(int64_t e = 0; e < num_edges; ++e) {
      const int64_t s = flat_src_ptr[e];
      const int64_t d = dst_idx_ptr[e];
      check_in_bounds(s, rel_rows, "flat_src");
      check_in_bounds(d, grad_out_rows, "dst_idx");
      const scalar_t* in_row = grad_out_ptr + d * emb;
      scalar_t* out_row = grad_rel_ptr + s * emb;
      for(int64_t k = 0; k < emb; ++k) {
         out_row[k] += in_row[k];
      }
   }
}

Tensor fanout_scatter_fallback(
   const Tensor& x_cat,
   const Tensor& src_global_idx,
   const Tensor& flat_dst,
   int64_t out_rows
)
{
   const int64_t emb = x_cat.size(1);
   std::vector< int64_t > shape{out_rows, emb};
   Tensor args_flat = torch::stable::new_zeros(x_cat, shape);
   if(src_global_idx.numel() == 0 || out_rows == 0) {
      return args_flat;
   }
   Tensor vals = dispatch_index_select0(x_cat, src_global_idx);
   return dispatch_index_copy0(args_flat, flat_dst, vals);
}

Tensor fanout_scatter_backward_fallback(
   const Tensor& grad_out,
   const Tensor& src_global_idx,
   const Tensor& flat_dst,
   int64_t x_rows
)
{
   const int64_t emb = grad_out.size(1);
   std::vector< int64_t > shape{x_rows, emb};
   Tensor grad_x = torch::stable::new_zeros(grad_out, shape);
   if(flat_dst.numel() == 0 || x_rows == 0) {
      return grad_x;
   }
   Tensor gathered = dispatch_index_select0(grad_out, flat_dst);
   Tensor index = make_scatter_index(src_global_idx, emb);
   return dispatch_scatter_reduce0(grad_x, index, gathered, "sum", true);
}

Tensor fanin_reduce_sum_fallback(
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t dim_size
)
{
   const int64_t emb = rel_flat.size(1);
   std::vector< int64_t > shape{dim_size, emb};
   Tensor out = torch::stable::new_zeros(rel_flat, shape);
   if(flat_src.numel() == 0 || dim_size == 0) {
      return out;
   }
   Tensor msgs = dispatch_index_select0(rel_flat, flat_src);
   Tensor index = make_scatter_index(dst_idx, emb);
   return dispatch_scatter_reduce0(out, index, msgs, "sum", true);
}

Tensor fanin_reduce_sum_backward_fallback(
   const Tensor& grad_out,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t rel_rows
)
{
   const int64_t emb = grad_out.size(1);
   std::vector< int64_t > shape{rel_rows, emb};
   Tensor grad_rel = torch::stable::new_zeros(grad_out, shape);
   if(flat_src.numel() == 0 || rel_rows == 0) {
      return grad_rel;
   }
   Tensor gathered = dispatch_index_select0(grad_out, dst_idx);
   Tensor index = make_scatter_index(flat_src, emb);
   return dispatch_scatter_reduce0(grad_rel, index, gathered, "sum", true);
}

Tensor fanin_reduce_logsumexp(
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t dim_size
)
{
   const int64_t emb = rel_flat.size(1);
   std::vector< int64_t > shape{dim_size, emb};
   Tensor max_init = torch::stable::new_zeros(rel_flat, shape);
   torch::stable::fill_(max_init, -std::numeric_limits< double >::infinity());

   if(flat_src.numel() == 0 || dim_size == 0) {
      return max_init;
   }

   Tensor msgs = dispatch_index_select0(rel_flat, flat_src);
   Tensor index = make_scatter_index(dst_idx, emb);
   Tensor max_vals = dispatch_scatter_reduce0(max_init, index, msgs, "amax", true);
   Tensor max_offsets = dispatch_index_select0(max_vals, dst_idx);
   Tensor centered = torch::stable::subtract(msgs, max_offsets, 1.0);
   Tensor exps = dispatch_exp(centered);

   Tensor sum_init = torch::stable::new_zeros(rel_flat, shape);
   Tensor exps_sum = dispatch_scatter_reduce0(sum_init, index, exps, "sum", true);
   Tensor logs = dispatch_log(exps_sum);
   return torch::stable::subtract(logs, max_vals, -1.0);
}

Tensor fanout_scatter(
   const Tensor& x_cat,
   const Tensor& src_global_idx,
   const Tensor& flat_dst,
   int64_t out_rows
)
{
   check_rank(x_cat, 2, "x_cat");
   check_int64_index(src_global_idx, "src_global_idx");
   check_int64_index(flat_dst, "flat_dst");
   STD_TORCH_CHECK(
      src_global_idx.size(0) == flat_dst.size(0),
      "src_global_idx and flat_dst must have equal length."
   );
   STD_TORCH_CHECK(out_rows >= 0, "out_rows must be >= 0.");

   const int64_t emb = x_cat.size(1);
   std::vector< int64_t > shape{out_rows, emb};
   Tensor args_flat = torch::stable::new_zeros(x_cat, shape);
   if(src_global_idx.numel() == 0 || out_rows == 0) {
      return args_flat;
   }

#if defined(RELM_RELMP_HAS_CUDA) && RELM_RELMP_HAS_CUDA
   if(x_cat.is_cuda() && src_global_idx.is_cuda() && flat_dst.is_cuda()
      && is_fastpath_dtype(dtype_of(x_cat))) {
      return fanout_scatter_cuda(x_cat, src_global_idx, flat_dst, out_rows);
   }
#endif

   if(x_cat.is_cpu() && src_global_idx.is_cpu() && flat_dst.is_cpu()
      && is_fastpath_dtype(dtype_of(x_cat))) {
      Tensor x_work = ensure_contiguous(x_cat);
      Tensor src_work = ensure_contiguous(src_global_idx);
      Tensor dst_work = ensure_contiguous(flat_dst);

      const int64_t num_edges = src_work.size(0);
      if(dtype_of(x_work) == aoti_torch_dtype_float32()) {
         fanout_scatter_cpu_kernel< float >(
            x_work.const_data_ptr< float >(),
            src_work.const_data_ptr< int64_t >(),
            dst_work.const_data_ptr< int64_t >(),
            num_edges,
            x_work.size(0),
            out_rows,
            emb,
            args_flat.mutable_data_ptr< float >()
         );
         return args_flat;
      }
      fanout_scatter_cpu_kernel< double >(
         x_work.const_data_ptr< double >(),
         src_work.const_data_ptr< int64_t >(),
         dst_work.const_data_ptr< int64_t >(),
         num_edges,
         x_work.size(0),
         out_rows,
         emb,
         args_flat.mutable_data_ptr< double >()
      );
      return args_flat;
   }

   return fanout_scatter_fallback(x_cat, src_global_idx, flat_dst, out_rows);
}

Tensor fanout_scatter_backward(
   const Tensor& grad_out,
   const Tensor& src_global_idx,
   const Tensor& flat_dst,
   int64_t x_rows
)
{
   check_rank(grad_out, 2, "grad_out");
   check_int64_index(src_global_idx, "src_global_idx");
   check_int64_index(flat_dst, "flat_dst");
   STD_TORCH_CHECK(
      src_global_idx.size(0) == flat_dst.size(0),
      "src_global_idx and flat_dst must have equal length."
   );
   STD_TORCH_CHECK(x_rows >= 0, "x_rows must be >= 0.");

   const int64_t emb = grad_out.size(1);
   std::vector< int64_t > shape{x_rows, emb};
   Tensor grad_x = torch::stable::new_zeros(grad_out, shape);
   if(src_global_idx.numel() == 0 || x_rows == 0) {
      return grad_x;
   }

#if defined(RELM_RELMP_HAS_CUDA) && RELM_RELMP_HAS_CUDA
   if(grad_out.is_cuda() && src_global_idx.is_cuda() && flat_dst.is_cuda()
      && is_fastpath_dtype(dtype_of(grad_out))) {
      return fanout_scatter_backward_cuda(grad_out, src_global_idx, flat_dst, x_rows);
   }
#endif

   if(grad_out.is_cpu() && src_global_idx.is_cpu() && flat_dst.is_cpu()
      && is_fastpath_dtype(dtype_of(grad_out))) {
      Tensor grad_out_work = ensure_contiguous(grad_out);
      Tensor src_work = ensure_contiguous(src_global_idx);
      Tensor dst_work = ensure_contiguous(flat_dst);
      const int64_t num_edges = src_work.size(0);

      if(dtype_of(grad_out_work) == aoti_torch_dtype_float32()) {
         fanout_scatter_backward_cpu_kernel< float >(
            grad_out_work.const_data_ptr< float >(),
            src_work.const_data_ptr< int64_t >(),
            dst_work.const_data_ptr< int64_t >(),
            num_edges,
            x_rows,
            grad_out_work.size(0),
            emb,
            grad_x.mutable_data_ptr< float >()
         );
         return grad_x;
      }
      fanout_scatter_backward_cpu_kernel< double >(
         grad_out_work.const_data_ptr< double >(),
         src_work.const_data_ptr< int64_t >(),
         dst_work.const_data_ptr< int64_t >(),
         num_edges,
         x_rows,
         grad_out_work.size(0),
         emb,
         grad_x.mutable_data_ptr< double >()
      );
      return grad_x;
   }

   return fanout_scatter_backward_fallback(grad_out, src_global_idx, flat_dst, x_rows);
}

Tensor fanin_reduce_sum(
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t dim_size
)
{
   const int64_t emb = rel_flat.size(1);
   std::vector< int64_t > shape{dim_size, emb};
   Tensor out = torch::stable::new_zeros(rel_flat, shape);
   if(flat_src.numel() == 0 || dim_size == 0) {
      return out;
   }

#if defined(RELM_RELMP_HAS_CUDA) && RELM_RELMP_HAS_CUDA
   if(rel_flat.is_cuda() && flat_src.is_cuda() && dst_idx.is_cuda()
      && is_fastpath_dtype(dtype_of(rel_flat))) {
      return fanin_reduce_sum_cuda(rel_flat, flat_src, dst_idx, dim_size);
   }
#endif

   if(rel_flat.is_cpu() && flat_src.is_cpu() && dst_idx.is_cpu()
      && is_fastpath_dtype(dtype_of(rel_flat))) {
      Tensor rel_work = ensure_contiguous(rel_flat);
      Tensor src_work = ensure_contiguous(flat_src);
      Tensor dst_work = ensure_contiguous(dst_idx);
      const int64_t num_edges = src_work.size(0);

      if(dtype_of(rel_work) == aoti_torch_dtype_float32()) {
         fanin_reduce_sum_cpu_kernel< float >(
            rel_work.const_data_ptr< float >(),
            src_work.const_data_ptr< int64_t >(),
            dst_work.const_data_ptr< int64_t >(),
            num_edges,
            rel_work.size(0),
            dim_size,
            emb,
            out.mutable_data_ptr< float >()
         );
         return out;
      }
      fanin_reduce_sum_cpu_kernel< double >(
         rel_work.const_data_ptr< double >(),
         src_work.const_data_ptr< int64_t >(),
         dst_work.const_data_ptr< int64_t >(),
         num_edges,
         rel_work.size(0),
         dim_size,
         emb,
         out.mutable_data_ptr< double >()
      );
      return out;
   }

   return fanin_reduce_sum_fallback(rel_flat, flat_src, dst_idx, dim_size);
}

Tensor fanin_reduce_sum_backward(
   const Tensor& grad_out,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t rel_rows
)
{
   check_rank(grad_out, 2, "grad_out");
   check_int64_index(flat_src, "flat_src");
   check_int64_index(dst_idx, "dst_idx");
   STD_TORCH_CHECK(
      flat_src.size(0) == dst_idx.size(0), "flat_src and dst_idx must have equal length."
   );
   STD_TORCH_CHECK(rel_rows >= 0, "rel_rows must be >= 0.");

   const int64_t emb = grad_out.size(1);
   std::vector< int64_t > shape{rel_rows, emb};
   Tensor grad_rel = torch::stable::new_zeros(grad_out, shape);
   if(flat_src.numel() == 0 || rel_rows == 0) {
      return grad_rel;
   }

#if defined(RELM_RELMP_HAS_CUDA) && RELM_RELMP_HAS_CUDA
   if(grad_out.is_cuda() && flat_src.is_cuda() && dst_idx.is_cuda()
      && is_fastpath_dtype(dtype_of(grad_out))) {
      return fanin_reduce_sum_backward_cuda(grad_out, flat_src, dst_idx, rel_rows);
   }
#endif

   if(grad_out.is_cpu() && flat_src.is_cpu() && dst_idx.is_cpu()
      && is_fastpath_dtype(dtype_of(grad_out))) {
      Tensor grad_out_work = ensure_contiguous(grad_out);
      Tensor src_work = ensure_contiguous(flat_src);
      Tensor dst_work = ensure_contiguous(dst_idx);
      const int64_t num_edges = src_work.size(0);

      if(dtype_of(grad_out_work) == aoti_torch_dtype_float32()) {
         fanin_reduce_sum_backward_cpu_kernel< float >(
            grad_out_work.const_data_ptr< float >(),
            src_work.const_data_ptr< int64_t >(),
            dst_work.const_data_ptr< int64_t >(),
            num_edges,
            rel_rows,
            grad_out_work.size(0),
            emb,
            grad_rel.mutable_data_ptr< float >()
         );
         return grad_rel;
      }
      fanin_reduce_sum_backward_cpu_kernel< double >(
         grad_out_work.const_data_ptr< double >(),
         src_work.const_data_ptr< int64_t >(),
         dst_work.const_data_ptr< int64_t >(),
         num_edges,
         rel_rows,
         grad_out_work.size(0),
         emb,
         grad_rel.mutable_data_ptr< double >()
      );
      return grad_rel;
   }

   return fanin_reduce_sum_backward_fallback(grad_out, flat_src, dst_idx, rel_rows);
}

Tensor fanin_reduce(
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t dim_size,
   int64_t mode
)
{
   check_rank(rel_flat, 2, "rel_flat");
   check_int64_index(flat_src, "flat_src");
   check_int64_index(dst_idx, "dst_idx");
   STD_TORCH_CHECK(
      flat_src.size(0) == dst_idx.size(0), "flat_src and dst_idx must have equal length."
   );
   STD_TORCH_CHECK(dim_size >= 0, "dim_size must be >= 0.");

   if(mode == kModeSum) {
      return fanin_reduce_sum(rel_flat, flat_src, dst_idx, dim_size);
   }
   if(mode == kModeLogSumExp) {
      return fanin_reduce_logsumexp(rel_flat, flat_src, dst_idx, dim_size);
   }
   throw std::invalid_argument("Unsupported fanin_reduce mode. Supported: 0=sum, 1=logsumexp.");
}

std::string build_info()
{
   return std::string("build_torch=") + RELM_RELMP_BUILD_TORCH_VERSION + ";build_cuda_tag="
          + RELM_RELMP_BUILD_CUDA_TAG + ";stable_abi_target=" + RELM_RELMP_STABLE_ABI_TARGET;
}

}  // namespace relm::relmp

STABLE_TORCH_LIBRARY(relm_relmp, m)
{
   m.def(
      "fanout_scatter(Tensor x_cat, Tensor src_global_idx, Tensor flat_dst, int out_rows) -> Tensor"
   );
   m.def(
      "fanout_scatter_backward(Tensor grad_out, Tensor src_global_idx, Tensor flat_dst, int "
      "x_rows) -> Tensor"
   );
   m.def(
      "fanin_reduce(Tensor rel_flat, Tensor flat_src, Tensor dst_idx, int dim_size, int mode) -> "
      "Tensor"
   );
   m.def(
      "fanin_reduce_sum_backward(Tensor grad_out, Tensor flat_src, Tensor dst_idx, int rel_rows) "
      "-> Tensor"
   );
   m.def("build_info() -> str");
}

STABLE_TORCH_LIBRARY_IMPL(relm_relmp, CompositeImplicitAutograd, m)
{
   m.impl("fanout_scatter", TORCH_BOX(relm::relmp::fanout_scatter));
   m.impl("fanout_scatter_backward", TORCH_BOX(relm::relmp::fanout_scatter_backward));
   m.impl("fanin_reduce", TORCH_BOX(relm::relmp::fanin_reduce));
   m.impl("fanin_reduce_sum_backward", TORCH_BOX(relm::relmp::fanin_reduce_sum_backward));
   m.impl("build_info", TORCH_BOX(relm::relmp::build_info));
}
