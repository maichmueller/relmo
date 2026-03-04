#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace relm::mp {

std::tuple< Tensor, Tensor, Tensor > fanout_pack_multi(
   const std::vector< Tensor >& x_parts,
   const std::vector< Tensor >& src_idx_parts,
   const std::vector< Tensor >& flat_dst_parts
);

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
);

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


Tensor fanout_scatter_fallback(
   const Tensor& x_cat,
   const Tensor& src_global_idx,
   const Tensor& flat_dst,
   int64_t out_rows
);

Tensor fanout_scatter_backward_fallback(
   const Tensor& grad_out,
   const Tensor& src_global_idx,
   const Tensor& flat_dst,
   int64_t x_rows
);

Tensor fanout_scatter(
   const Tensor& x_cat,
   const Tensor& src_global_idx,
   const Tensor& flat_dst,
   int64_t out_rows
);

Tensor fanout_scatter_backward(
   const Tensor& grad_out,
   const Tensor& src_global_idx,
   const Tensor& flat_dst,
   int64_t x_rows
);


}  // namespace relm::mp
