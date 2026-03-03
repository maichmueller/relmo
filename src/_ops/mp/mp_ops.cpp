#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace relm::mp {

using at::Tensor;

constexpr int64_t kModeSum = 0;
constexpr int64_t kModeLogSumExp = 1;

at::ScalarType dtype_of(const Tensor& t)
{
   return t.scalar_type();
}

bool is_fastpath_dtype(at::ScalarType dtype)
{
   return dtype == at::kFloat || dtype == at::kDouble;
}

void check_rank(const Tensor& t, int64_t expected, const char* name)
{
   TORCH_CHECK(
      t.dim() == expected, name, " must be rank ", expected, ", got rank ", t.dim(), "."
   );
}

void check_int64_index(const Tensor& t, const char* name)
{
   check_rank(t, 1, name);
   TORCH_CHECK(dtype_of(t) == at::kLong, name, " must have dtype torch.int64.");
}

void check_in_bounds(int64_t idx, int64_t size, const char* name)
{
   TORCH_CHECK(
      idx >= 0 && idx < size, name, " index out of bounds: ", idx, " not in [0, ", size, ")."
   );
}

Tensor ensure_contiguous(const Tensor& t)
{
   return t.is_contiguous() ? t : t.contiguous();
}

Tensor make_scatter_index(const Tensor& idx, int64_t emb)
{
   return idx.unsqueeze(1).expand({idx.size(0), emb});
}

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
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
Tensor fanin_reduce_logsumexp_cuda(
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
Tensor fanin_reduce_logsumexp_backward_cuda(
   const Tensor& grad_out,
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   const Tensor& out,
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
   Tensor out = at::zeros({out_rows, emb}, x_cat.options());
   if(src_global_idx.numel() == 0 || out_rows == 0) {
      return out;
   }
   Tensor vals = x_cat.index_select(0, src_global_idx);
   out.index_copy_(0, flat_dst, vals);
   return out;
}

Tensor fanout_scatter_backward_fallback(
   const Tensor& grad_out,
   const Tensor& src_global_idx,
   const Tensor& flat_dst,
   int64_t x_rows
)
{
   const int64_t emb = grad_out.size(1);
   Tensor grad_x = at::zeros({x_rows, emb}, grad_out.options());
   if(flat_dst.numel() == 0 || x_rows == 0) {
      return grad_x;
   }
   Tensor gathered = grad_out.index_select(0, flat_dst);
   grad_x.index_add_(0, src_global_idx, gathered);
   return grad_x;
}

Tensor fanin_reduce_sum_fallback(
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t dim_size
)
{
   const int64_t emb = rel_flat.size(1);
   Tensor out = at::zeros({dim_size, emb}, rel_flat.options());
   if(flat_src.numel() == 0 || dim_size == 0) {
      return out;
   }
   Tensor msgs = rel_flat.index_select(0, flat_src);
   out.index_add_(0, dst_idx, msgs);
   return out;
}

Tensor fanin_reduce_sum_backward_fallback(
   const Tensor& grad_out,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t rel_rows
)
{
   const int64_t emb = grad_out.size(1);
   Tensor grad_rel = at::zeros({rel_rows, emb}, grad_out.options());
   if(flat_src.numel() == 0 || rel_rows == 0) {
      return grad_rel;
   }
   Tensor gathered = grad_out.index_select(0, dst_idx);
   grad_rel.index_add_(0, flat_src, gathered);
   return grad_rel;
}

Tensor fanin_reduce_logsumexp_backward_fallback(
   const Tensor& grad_out,
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   const Tensor& out,
   int64_t rel_rows
)
{
   const int64_t emb = grad_out.size(1);
   Tensor grad_rel = at::zeros({rel_rows, emb}, rel_flat.options());
   if(flat_src.numel() == 0 || rel_rows == 0) {
      return grad_rel;
   }

   Tensor msgs = rel_flat.index_select(0, flat_src);
   Tensor out_sel = out.index_select(0, dst_idx);
   Tensor grad_sel = grad_out.index_select(0, dst_idx);
   Tensor weights = (msgs - out_sel).exp();
   Tensor contrib = grad_sel * weights;
   grad_rel.index_add_(0, flat_src, contrib);
   return grad_rel;
}

Tensor fanin_reduce_logsumexp(
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t dim_size
)
{
#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
   if(rel_flat.is_cuda() && flat_src.is_cuda() && dst_idx.is_cuda()
      && is_fastpath_dtype(dtype_of(rel_flat))) {
      return fanin_reduce_logsumexp_cuda(rel_flat, flat_src, dst_idx, dim_size);
   }
#endif

   const int64_t emb = rel_flat.size(1);
   Tensor max_vals = at::full(
      {dim_size, emb}, -std::numeric_limits< double >::infinity(), rel_flat.options()
   );

   if(flat_src.numel() == 0 || dim_size == 0) {
      return max_vals;
   }

   Tensor msgs = rel_flat.index_select(0, flat_src);
   Tensor index = make_scatter_index(dst_idx, emb);
   max_vals.scatter_reduce_(0, index, msgs, "amax", true);
   Tensor max_offsets = max_vals.index_select(0, dst_idx);
   Tensor exps = (msgs - max_offsets).exp();
   Tensor exps_sum = at::zeros({dim_size, emb}, rel_flat.options());
   exps_sum.scatter_add_(0, index, exps);
   return exps_sum.log() + max_vals;
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
   TORCH_CHECK(
      src_global_idx.size(0) == flat_dst.size(0),
      "src_global_idx and flat_dst must have equal length."
   );
   TORCH_CHECK(out_rows >= 0, "out_rows must be >= 0.");

   const int64_t emb = x_cat.size(1);
   Tensor out = at::zeros({out_rows, emb}, x_cat.options());
   if(src_global_idx.numel() == 0 || out_rows == 0) {
      return out;
   }

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
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

      if(dtype_of(x_work) == at::kFloat) {
         fanout_scatter_cpu_kernel< float >(
            x_work.data_ptr< float >(),
            src_work.data_ptr< int64_t >(),
            dst_work.data_ptr< int64_t >(),
            num_edges,
            x_work.size(0),
            out_rows,
            emb,
            out.data_ptr< float >()
         );
         return out;
      }
      fanout_scatter_cpu_kernel< double >(
         x_work.data_ptr< double >(),
         src_work.data_ptr< int64_t >(),
         dst_work.data_ptr< int64_t >(),
         num_edges,
         x_work.size(0),
         out_rows,
         emb,
         out.data_ptr< double >()
      );
      return out;
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
   TORCH_CHECK(
      src_global_idx.size(0) == flat_dst.size(0),
      "src_global_idx and flat_dst must have equal length."
   );
   TORCH_CHECK(x_rows >= 0, "x_rows must be >= 0.");

   const int64_t emb = grad_out.size(1);
   Tensor grad_x = at::zeros({x_rows, emb}, grad_out.options());
   if(src_global_idx.numel() == 0 || x_rows == 0) {
      return grad_x;
   }

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
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

      if(dtype_of(grad_out_work) == at::kFloat) {
         fanout_scatter_backward_cpu_kernel< float >(
            grad_out_work.data_ptr< float >(),
            src_work.data_ptr< int64_t >(),
            dst_work.data_ptr< int64_t >(),
            num_edges,
            x_rows,
            grad_out_work.size(0),
            emb,
            grad_x.data_ptr< float >()
         );
         return grad_x;
      }
      fanout_scatter_backward_cpu_kernel< double >(
         grad_out_work.data_ptr< double >(),
         src_work.data_ptr< int64_t >(),
         dst_work.data_ptr< int64_t >(),
         num_edges,
         x_rows,
         grad_out_work.size(0),
         emb,
         grad_x.data_ptr< double >()
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
   Tensor out = at::zeros({dim_size, emb}, rel_flat.options());
   if(flat_src.numel() == 0 || dim_size == 0) {
      return out;
   }

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
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

      if(dtype_of(rel_work) == at::kFloat) {
         fanin_reduce_sum_cpu_kernel< float >(
            rel_work.data_ptr< float >(),
            src_work.data_ptr< int64_t >(),
            dst_work.data_ptr< int64_t >(),
            num_edges,
            rel_work.size(0),
            dim_size,
            emb,
            out.data_ptr< float >()
         );
         return out;
      }
      fanin_reduce_sum_cpu_kernel< double >(
         rel_work.data_ptr< double >(),
         src_work.data_ptr< int64_t >(),
         dst_work.data_ptr< int64_t >(),
         num_edges,
         rel_work.size(0),
         dim_size,
         emb,
         out.data_ptr< double >()
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
   TORCH_CHECK(flat_src.size(0) == dst_idx.size(0), "flat_src and dst_idx must have equal length.");
   TORCH_CHECK(rel_rows >= 0, "rel_rows must be >= 0.");

   const int64_t emb = grad_out.size(1);
   Tensor grad_rel = at::zeros({rel_rows, emb}, grad_out.options());
   if(flat_src.numel() == 0 || rel_rows == 0) {
      return grad_rel;
   }

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
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

      if(dtype_of(grad_out_work) == at::kFloat) {
         fanin_reduce_sum_backward_cpu_kernel< float >(
            grad_out_work.data_ptr< float >(),
            src_work.data_ptr< int64_t >(),
            dst_work.data_ptr< int64_t >(),
            num_edges,
            rel_rows,
            grad_out_work.size(0),
            emb,
            grad_rel.data_ptr< float >()
         );
         return grad_rel;
      }
      fanin_reduce_sum_backward_cpu_kernel< double >(
         grad_out_work.data_ptr< double >(),
         src_work.data_ptr< int64_t >(),
         dst_work.data_ptr< int64_t >(),
         num_edges,
         rel_rows,
         grad_out_work.size(0),
         emb,
         grad_rel.data_ptr< double >()
      );
      return grad_rel;
   }

   return fanin_reduce_sum_backward_fallback(grad_out, flat_src, dst_idx, rel_rows);
}

Tensor fanin_reduce_logsumexp_backward(
   const Tensor& grad_out,
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   const Tensor& out,
   int64_t rel_rows
)
{
   check_rank(grad_out, 2, "grad_out");
   check_rank(rel_flat, 2, "rel_flat");
   check_int64_index(flat_src, "flat_src");
   check_int64_index(dst_idx, "dst_idx");
   check_rank(out, 2, "out");

   TORCH_CHECK(flat_src.size(0) == dst_idx.size(0), "flat_src and dst_idx must have equal length.");
   TORCH_CHECK(rel_flat.size(1) == grad_out.size(1), "rel_flat and grad_out must share embedding dim.");
   TORCH_CHECK(out.size(0) == grad_out.size(0), "out and grad_out must share dim_size.");
   TORCH_CHECK(out.size(1) == grad_out.size(1), "out and grad_out must share embedding dim.");
   TORCH_CHECK(rel_rows >= 0, "rel_rows must be >= 0.");

   const int64_t emb = grad_out.size(1);
   Tensor grad_rel = at::zeros({rel_rows, emb}, rel_flat.options());
   if(flat_src.numel() == 0 || rel_rows == 0) {
      return grad_rel;
   }

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
   if(grad_out.is_cuda() && rel_flat.is_cuda() && flat_src.is_cuda() && dst_idx.is_cuda()
      && out.is_cuda() && is_fastpath_dtype(dtype_of(grad_out))) {
      return fanin_reduce_logsumexp_backward_cuda(
         grad_out, rel_flat, flat_src, dst_idx, out, rel_rows
      );
   }
#endif

   return fanin_reduce_logsumexp_backward_fallback(
      grad_out, rel_flat, flat_src, dst_idx, out, rel_rows
   );
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
   TORCH_CHECK(flat_src.size(0) == dst_idx.size(0), "flat_src and dst_idx must have equal length.");
   TORCH_CHECK(dim_size >= 0, "dim_size must be >= 0.");

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
   return std::string("build_torch=") + RELM_MP_BUILD_TORCH_VERSION + ";build_cuda_tag="
          + RELM_MP_BUILD_CUDA_TAG + ";abi_target=" + RELM_MP_ABI_TARGET + ";abi_lane="
          + RELM_MP_ABI_LANE;
}

}  // namespace relm::mp

TORCH_LIBRARY(relm_mp, m)
{
   m.def(
      "fanout_scatter(Tensor x_cat, Tensor src_global_idx, Tensor flat_dst, int out_rows) -> Tensor"
   );
   m.def(
      "fanout_scatter_backward(Tensor grad_out, Tensor src_global_idx, Tensor flat_dst, int x_rows) -> Tensor"
   );
   m.def(
      "fanin_reduce(Tensor rel_flat, Tensor flat_src, Tensor dst_idx, int dim_size, int mode) -> Tensor"
   );
   m.def(
      "fanin_reduce_sum_backward(Tensor grad_out, Tensor flat_src, Tensor dst_idx, int rel_rows) -> Tensor"
   );
   m.def(
      "fanin_reduce_logsumexp_backward(Tensor grad_out, Tensor rel_flat, Tensor flat_src, Tensor dst_idx, Tensor out, int rel_rows) -> Tensor"
   );
   m.def("build_info() -> str");
}

TORCH_LIBRARY_IMPL(relm_mp, CompositeImplicitAutograd, m)
{
   m.impl("fanout_scatter", relm::mp::fanout_scatter);
   m.impl("fanout_scatter_backward", relm::mp::fanout_scatter_backward);
   m.impl("fanin_reduce", relm::mp::fanin_reduce);
   m.impl("fanin_reduce_sum_backward", relm::mp::fanin_reduce_sum_backward);
   m.impl("fanin_reduce_logsumexp_backward", relm::mp::fanin_reduce_logsumexp_backward);
   m.impl("build_info", relm::mp::build_info);
}
