#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "utils.hpp"

namespace relm::mp{

std::tuple< Tensor, Tensor, Tensor > fanin_pack_multi(
   const std::vector< Tensor >& rel_parts,
   const std::vector< Tensor >& flat_src_parts,
   const std::vector< Tensor >& dst_idx_parts
)
{
   TORCH_CHECK(!rel_parts.empty(), "fanin_pack_multi requires at least one relation tensor.");
   TORCH_CHECK(
      rel_parts.size() == flat_src_parts.size() && rel_parts.size() == dst_idx_parts.size(),
      "fanin_pack_multi expects rel_parts, flat_src_parts, and dst_idx_parts with equal lengths."
   );

   const Tensor& ref_rel = rel_parts.front();
   check_rank(ref_rel, 2, "rel_parts[0]");
   const int64_t emb = ref_rel.size(1);
   const auto ref_device = ref_rel.device();
   const auto ref_dtype = dtype_of(ref_rel);

   std::vector< Tensor > rel_cat_parts;
   std::vector< Tensor > src_cat_parts;
   std::vector< Tensor > dst_cat_parts;
   rel_cat_parts.reserve(rel_parts.size());
   src_cat_parts.reserve(rel_parts.size());
   dst_cat_parts.reserve(rel_parts.size());
   int64_t row_offset = 0;

   for(size_t i = 0; i < rel_parts.size(); ++i) {
      const Tensor& rel = rel_parts[i];
      const Tensor& flat_src = flat_src_parts[i];
      const Tensor& dst_idx = dst_idx_parts[i];

      check_rank(rel, 2, "rel_parts[i]");
      check_int64_index(flat_src, "flat_src_parts[i]");
      check_int64_index(dst_idx, "dst_idx_parts[i]");
      TORCH_CHECK(
         rel.device() == ref_device,
         "fanin_pack_multi expects all rel_parts on the same device. Mismatch at part ",
         i,
         "."
      );
      TORCH_CHECK(
         dtype_of(rel) == ref_dtype,
         "fanin_pack_multi expects all rel_parts with the same dtype. Mismatch at part ",
         i,
         "."
      );
      TORCH_CHECK(
         rel.size(1) == emb,
         "fanin_pack_multi expects matching embedding dim across rel_parts. Part ",
         i,
         " has emb=",
         rel.size(1),
         ", expected ",
         emb,
         "."
      );
      TORCH_CHECK(
         flat_src.device() == ref_device && dst_idx.device() == ref_device,
         "fanin_pack_multi expects all index tensors on the same device as rel_parts."
      );
      TORCH_CHECK(
         flat_src.size(0) == dst_idx.size(0),
         "fanin_pack_multi expects flat_src and dst_idx lengths to match for part ",
         i,
         "."
      );

      rel_cat_parts.push_back(rel);
      src_cat_parts.push_back(flat_src + row_offset);
      dst_cat_parts.push_back(dst_idx);
      row_offset += rel.size(0);
   }

   Tensor rel_cat = rel_cat_parts.size() == 1 ? rel_cat_parts[0] : at::cat(rel_cat_parts, 0);
   Tensor flat_src = src_cat_parts.size() == 1 ? src_cat_parts[0] : at::cat(src_cat_parts, 0);
   Tensor dst_idx = dst_cat_parts.size() == 1 ? dst_cat_parts[0] : at::cat(dst_cat_parts, 0);
   return std::make_tuple(rel_cat, flat_src, dst_idx);
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


}  // namespace relm::mp
