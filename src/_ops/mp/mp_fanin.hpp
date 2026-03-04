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

#include "utils.hpp"


namespace relm::mp {


std::tuple< Tensor, Tensor, Tensor > fanin_pack_multi(
   const std::vector< Tensor >& rel_parts,
   const std::vector< Tensor >& flat_src_parts,
   const std::vector< Tensor >& dst_idx_parts
);

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
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
);

Tensor fanin_reduce_sum_backward_fallback(
   const Tensor& grad_out,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t rel_rows
);

Tensor fanin_reduce_logsumexp_backward_fallback(
   const Tensor& grad_out,
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   const Tensor& out,
   int64_t rel_rows
);

Tensor fanin_reduce_logsumexp(
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t dim_size
);

Tensor fanin_reduce_sum(
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t dim_size
);

Tensor fanin_reduce_sum_backward(
   const Tensor& grad_out,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t rel_rows
);

Tensor fanin_reduce_logsumexp_backward(
   const Tensor& grad_out,
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   const Tensor& out,
   int64_t rel_rows
);

Tensor fanin_reduce(
   const Tensor& rel_flat,
   const Tensor& flat_src,
   const Tensor& dst_idx,
   int64_t dim_size,
   int64_t mode
);

}  // namespace relm::mp
