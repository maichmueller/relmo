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

using at::Tensor;

constexpr int64_t kModeSum = 0;
constexpr int64_t kModeLogSumExp = 1;

constexpr int64_t kPwIdentity = 0;
constexpr int64_t kPwReLU = 1;
constexpr int64_t kPwMish = 2;
constexpr int64_t kPwGeluNone = 3;
constexpr int64_t kPwGeluTanh = 4;
constexpr int64_t kPwSiLU = 5;
constexpr int64_t kPwTanh = 6;

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

Tensor apply_pointwise_code(const Tensor& x, int64_t code)
{
   if(code == kPwIdentity) {
      return x;
   }
   if(code == kPwReLU) {
      return at::relu(x);
   }
   if(code == kPwMish) {
      return at::mish(x);
   }
   if(code == kPwGeluNone) {
      return at::gelu(x, "none");
   }
   if(code == kPwGeluTanh) {
      return at::gelu(x, "tanh");
   }
   if(code == kPwSiLU) {
      return at::silu(x);
   }
   if(code == kPwTanh) {
      return at::tanh(x);
   }
   TORCH_CHECK(false, "Unsupported grouped pointwise code: ", code, ".");
}

std::tuple< Tensor, Tensor, Tensor > fanout_pack_multi(
   const std::vector< Tensor >& x_parts,
   const std::vector< Tensor >& src_idx_parts,
   const std::vector< Tensor >& flat_dst_parts
)
{
   TORCH_CHECK(!x_parts.empty(), "fanout_pack_multi requires at least one source tensor.");
   TORCH_CHECK(
      x_parts.size() == src_idx_parts.size() && x_parts.size() == flat_dst_parts.size(),
      "fanout_pack_multi expects x_parts, src_idx_parts, and flat_dst_parts with equal lengths."
   );

   const Tensor& ref_x = x_parts.front();
   check_rank(ref_x, 2, "x_parts[0]");
   const int64_t emb = ref_x.size(1);
   const auto ref_device = ref_x.device();
   const auto ref_dtype = dtype_of(ref_x);

   std::vector< Tensor > src_global_parts;
   std::vector< Tensor > x_cat_parts;
   std::vector< Tensor > flat_cat_parts;
   src_global_parts.reserve(x_parts.size());
   x_cat_parts.reserve(x_parts.size());
   flat_cat_parts.reserve(x_parts.size());
   int64_t row_offset = 0;

   for(size_t i = 0; i < x_parts.size(); ++i) {
      const Tensor& x = x_parts[i];
      const Tensor& src_idx = src_idx_parts[i];
      const Tensor& flat_dst = flat_dst_parts[i];

      check_rank(x, 2, "x_parts[i]");
      check_int64_index(src_idx, "src_idx_parts[i]");
      check_int64_index(flat_dst, "flat_dst_parts[i]");
      TORCH_CHECK(
         x.device() == ref_device,
         "fanout_pack_multi expects all x_parts on the same device. Mismatch at part ",
         i,
         "."
      );
      TORCH_CHECK(
         dtype_of(x) == ref_dtype,
         "fanout_pack_multi expects all x_parts with the same dtype. Mismatch at part ",
         i,
         "."
      );
      TORCH_CHECK(
         x.size(1) == emb,
         "fanout_pack_multi expects matching embedding dim across x_parts. Part ",
         i,
         " has emb=",
         x.size(1),
         ", expected ",
         emb,
         "."
      );
      TORCH_CHECK(
         src_idx.device() == ref_device && flat_dst.device() == ref_device,
         "fanout_pack_multi expects all index tensors on the same device as x_parts."
      );
      TORCH_CHECK(
         src_idx.size(0) == flat_dst.size(0),
         "fanout_pack_multi expects src_idx and flat_dst lengths to match for part ",
         i,
         "."
      );

      x_cat_parts.push_back(x);
      src_global_parts.push_back(src_idx + row_offset);
      flat_cat_parts.push_back(flat_dst);
      row_offset += x.size(0);
   }

   Tensor x_cat = x_cat_parts.size() == 1 ? x_cat_parts[0] : at::cat(x_cat_parts, 0);
   Tensor src_global =
      src_global_parts.size() == 1 ? src_global_parts[0] : at::cat(src_global_parts, 0);
   Tensor flat_dst =
      flat_cat_parts.size() == 1 ? flat_cat_parts[0] : at::cat(flat_cat_parts, 0);
   return std::make_tuple(x_cat, src_global, flat_dst);
}

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

Tensor grouped_stack_from_flat(
   const Tensor& flat,
   const std::vector< int64_t >& slot_offsets,
   const std::vector< int64_t >& row_sizes,
   int64_t arity
)
{
   check_rank(flat, 2, "flat");
   TORCH_CHECK(
      slot_offsets.size() == row_sizes.size(),
      "grouped_stack_from_flat expects slot_offsets and row_sizes with equal length."
   );
   TORCH_CHECK(arity > 0, "grouped_stack_from_flat expects arity > 0.");

   const int64_t groups = static_cast< int64_t >(slot_offsets.size());
   const int64_t emb = flat.size(1);
   const int64_t in_dim = emb * arity;
   int64_t max_rows = 0;
   for(const int64_t n : row_sizes) {
      TORCH_CHECK(n >= 0, "grouped_stack_from_flat row_sizes must be >= 0.");
      if(n > max_rows) {
         max_rows = n;
      }
   }

   Tensor out = at::zeros({groups, max_rows, in_dim}, flat.options());
   if(groups == 0 || max_rows == 0) {
      return out;
   }

   for(int64_t i = 0; i < groups; ++i) {
      const int64_t n = row_sizes[static_cast< size_t >(i)];
      if(n <= 0) {
         continue;
      }
      const int64_t start = slot_offsets[static_cast< size_t >(i)];
      const int64_t len = n * arity;
      TORCH_CHECK(start >= 0, "grouped_stack_from_flat slot_offsets must be >= 0.");
      TORCH_CHECK(
         start + len <= flat.size(0),
         "grouped_stack_from_flat slice out of bounds at group ",
         i,
         ": start=",
         start,
         " len=",
         len,
         " flat_rows=",
         flat.size(0),
         "."
      );
      Tensor src = flat.narrow(0, start, len).view({n, in_dim});
      out[i].narrow(0, 0, n).copy_(src);
   }
   return out;
}

std::tuple< Tensor, Tensor > grouped_residual_mlp_from_flat(
   const Tensor& flat,
   const std::vector< int64_t >& slot_offsets,
   const std::vector< int64_t >& row_sizes,
   int64_t arity,
   const std::vector< Tensor >& weight_stacks,
   const std::vector< Tensor >& bias_stacks,
   const std::vector< int64_t >& op_kinds,
   const std::vector< int64_t >& op_indices,
   const std::vector< int64_t >& pointwise_codes,
   int64_t truncated_dim,
   bool truncate_right
)
{
   check_rank(flat, 2, "flat");
   TORCH_CHECK(
      slot_offsets.size() == row_sizes.size(),
      "grouped_residual_mlp_from_flat expects slot_offsets and row_sizes with equal lengths."
   );
   TORCH_CHECK(
      op_kinds.size() == op_indices.size(),
      "grouped_residual_mlp_from_flat expects op_kinds and op_indices with equal lengths."
   );
   TORCH_CHECK(arity > 0, "grouped_residual_mlp_from_flat expects arity > 0.");

   const int64_t groups = static_cast< int64_t >(slot_offsets.size());
   const int64_t emb = flat.size(1);
   const int64_t in_dim = emb * arity;
   int64_t max_rows = 0;
   for(const int64_t n : row_sizes) {
      TORCH_CHECK(n >= 0, "grouped_residual_mlp_from_flat row_sizes must be >= 0.");
      if(n > max_rows) {
         max_rows = n;
      }
   }

   for(size_t i = 0; i < weight_stacks.size(); ++i) {
      const Tensor& w = weight_stacks[i];
      check_rank(w, 3, "weight_stacks[i]");
      TORCH_CHECK(
         w.size(0) == groups,
         "grouped_residual_mlp_from_flat weight_stacks[",
         i,
         "] first dim must equal group count."
      );
   }
   for(size_t i = 0; i < bias_stacks.size(); ++i) {
      const Tensor& b = bias_stacks[i];
      TORCH_CHECK(
         b.dim() <= 2,
         "grouped_residual_mlp_from_flat bias_stacks[",
         i,
         "] must be rank <= 2."
      );
      if(b.numel() == 0) {
         continue;
      }
      TORCH_CHECK(
         b.dim() == 2 && b.size(0) == groups,
         "grouped_residual_mlp_from_flat bias_stacks[",
         i,
         "] must have shape [groups, out_features] when non-empty."
      );
   }

   Tensor x_stack = at::zeros({groups, max_rows, in_dim}, flat.options());
   if(groups > 0 && max_rows > 0) {
      for(int64_t i = 0; i < groups; ++i) {
         const int64_t n = row_sizes[static_cast< size_t >(i)];
         if(n <= 0) {
            continue;
         }
         const int64_t start = slot_offsets[static_cast< size_t >(i)];
         const int64_t len = n * arity;
         TORCH_CHECK(
            start >= 0,
            "grouped_residual_mlp_from_flat expects non-negative slot offsets."
         );
         TORCH_CHECK(
            start + len <= flat.size(0),
            "grouped_residual_mlp_from_flat slice out of bounds at group ",
            i,
            ": start=",
            start,
            " len=",
            len,
            " flat_rows=",
            flat.size(0),
            "."
         );
         Tensor src = flat.narrow(0, start, len).view({n, in_dim});
         x_stack[i].narrow(0, 0, n).copy_(src);
      }
   }

   Tensor out_stack = x_stack;
   for(size_t op_i = 0; op_i < op_kinds.size(); ++op_i) {
      const int64_t kind = op_kinds[op_i];
      const int64_t idx = op_indices[op_i];
      if(kind == 0) {
         TORCH_CHECK(
            idx >= 0 && static_cast< size_t >(idx) < weight_stacks.size(),
            "grouped_residual_mlp_from_flat linear op index out of range: ",
            idx,
            "."
         );
         const Tensor& w = weight_stacks[static_cast< size_t >(idx)];
         out_stack = at::matmul(out_stack, w.transpose(1, 2));
         if(static_cast< size_t >(idx) < bias_stacks.size()) {
            const Tensor& b = bias_stacks[static_cast< size_t >(idx)];
            if(b.numel() > 0) {
               out_stack = out_stack + b.unsqueeze(1);
            }
         }
         continue;
      }
      if(kind == 1) {
         TORCH_CHECK(
            idx >= 0 && static_cast< size_t >(idx) < pointwise_codes.size(),
            "grouped_residual_mlp_from_flat pointwise op index out of range: ",
            idx,
            "."
         );
         const int64_t code = pointwise_codes[static_cast< size_t >(idx)];
         out_stack = apply_pointwise_code(out_stack, code);
         continue;
      }
      TORCH_CHECK(
         false,
         "grouped_residual_mlp_from_flat unsupported op kind: ",
         kind,
         "."
      );
   }

   Tensor residual;
   if(truncated_dim >= 0 && x_stack.size(-1) != truncated_dim) {
      if(truncate_right) {
         residual = x_stack.narrow(-1, 0, truncated_dim);
      } else {
         residual = x_stack.narrow(-1, x_stack.size(-1) - truncated_dim, truncated_dim);
      }
   } else {
      residual = x_stack;
   }
   out_stack = residual + out_stack;

   std::vector< Tensor > rel_parts;
   std::vector< Tensor > flat_parts;
   rel_parts.reserve(groups);
   flat_parts.reserve(groups);
   auto idx_options = flat.options().dtype(at::kLong);
   for(int64_t i = 0; i < groups; ++i) {
      const int64_t n = row_sizes[static_cast< size_t >(i)];
      if(n <= 0) {
         continue;
      }
      const int64_t slot = slot_offsets[static_cast< size_t >(i)];
      Tensor rel_i = out_stack[i].narrow(0, 0, n).contiguous().view({n * arity, emb});
      Tensor flat_i = at::arange(n * arity, idx_options) + slot;
      rel_parts.push_back(rel_i);
      flat_parts.push_back(flat_i);
   }
   if(rel_parts.empty()) {
      return std::make_tuple(
         flat.new_empty({0, emb}), at::empty({0}, flat.options().dtype(at::kLong))
      );
   }
   Tensor rel_cat = rel_parts.size() == 1 ? rel_parts[0] : at::cat(rel_parts, 0);
   Tensor flat_idx = flat_parts.size() == 1 ? flat_parts[0] : at::cat(flat_parts, 0);
   return std::make_tuple(rel_cat, flat_idx);
}

std::tuple< Tensor, Tensor, Tensor > fanout_pack_from_edges(
   const std::vector< Tensor >& x_parts,
   const std::vector< Tensor >& edge_src_parts,
   const std::vector< Tensor >& edge_dst_parts,
   const std::vector< int64_t >& src_part_ids,
   const std::vector< int64_t >& arity_parts,
   const std::vector< int64_t >& pos_parts,
   const std::vector< int64_t >& slot_offset_parts
)
{
   TORCH_CHECK(!x_parts.empty(), "fanout_pack_from_edges requires at least one source tensor.");
   const int64_t n_edge_parts = static_cast< int64_t >(edge_src_parts.size());
   TORCH_CHECK(
      static_cast< int64_t >(edge_dst_parts.size()) == n_edge_parts,
      "fanout_pack_from_edges expects edge_src_parts and edge_dst_parts with equal lengths."
   );
   TORCH_CHECK(
      static_cast< int64_t >(src_part_ids.size()) == n_edge_parts
         && static_cast< int64_t >(arity_parts.size()) == n_edge_parts
         && static_cast< int64_t >(pos_parts.size()) == n_edge_parts
         && static_cast< int64_t >(slot_offset_parts.size()) == n_edge_parts,
      "fanout_pack_from_edges expects metadata arrays to match number of edge parts."
   );

   const Tensor& ref_x = x_parts.front();
   check_rank(ref_x, 2, "x_parts[0]");
   const int64_t emb = ref_x.size(1);
   const auto ref_device = ref_x.device();
   const auto ref_dtype = dtype_of(ref_x);

   std::vector< int64_t > x_offsets(x_parts.size(), 0);
   int64_t row_offset = 0;
   for(size_t i = 0; i < x_parts.size(); ++i) {
      const Tensor& x = x_parts[i];
      check_rank(x, 2, "x_parts[i]");
      TORCH_CHECK(
         x.device() == ref_device,
         "fanout_pack_from_edges expects all x_parts on the same device. Mismatch at part ",
         i,
         "."
      );
      TORCH_CHECK(
         dtype_of(x) == ref_dtype,
         "fanout_pack_from_edges expects all x_parts with the same dtype. Mismatch at part ",
         i,
         "."
      );
      TORCH_CHECK(
         x.size(1) == emb,
         "fanout_pack_from_edges expects matching embedding dim across x_parts. Part ",
         i,
         " has emb=",
         x.size(1),
         ", expected ",
         emb,
         "."
      );
      x_offsets[i] = row_offset;
      row_offset += x.size(0);
   }

   Tensor x_cat = x_parts.size() == 1 ? x_parts[0] : at::cat(x_parts, 0);

   if(n_edge_parts == 0) {
      auto idx_opts = at::TensorOptions().device(ref_device).dtype(at::kLong);
      Tensor empty_idx = at::empty({0}, idx_opts);
      return std::make_tuple(x_cat, empty_idx, empty_idx.clone());
   }

   std::vector< Tensor > src_global_parts;
   std::vector< Tensor > flat_dst_parts;
   src_global_parts.reserve(edge_src_parts.size());
   flat_dst_parts.reserve(edge_src_parts.size());

   for(int64_t i = 0; i < n_edge_parts; ++i) {
      const Tensor& src_idx = edge_src_parts[static_cast< size_t >(i)];
      const Tensor& dst_idx = edge_dst_parts[static_cast< size_t >(i)];
      check_int64_index(src_idx, "edge_src_parts[i]");
      check_int64_index(dst_idx, "edge_dst_parts[i]");
      TORCH_CHECK(
         src_idx.device() == ref_device && dst_idx.device() == ref_device,
         "fanout_pack_from_edges expects all edge index tensors on the same device as x_parts."
      );
      TORCH_CHECK(
         src_idx.size(0) == dst_idx.size(0),
         "fanout_pack_from_edges expects edge src/dst lengths to match for part ",
         i,
         "."
      );

      const int64_t src_part = src_part_ids[static_cast< size_t >(i)];
      TORCH_CHECK(
         src_part >= 0 && src_part < static_cast< int64_t >(x_parts.size()),
         "fanout_pack_from_edges src_part_ids[",
         i,
         "] out of range: ",
         src_part,
         "."
      );
      const int64_t arity = arity_parts[static_cast< size_t >(i)];
      const int64_t pos = pos_parts[static_cast< size_t >(i)];
      const int64_t slot_offset = slot_offset_parts[static_cast< size_t >(i)];
      TORCH_CHECK(arity > 0, "fanout_pack_from_edges requires arity > 0 for edge part ", i, ".");
      TORCH_CHECK(
         pos >= 0 && pos < arity,
         "fanout_pack_from_edges pos out of range for edge part ",
         i,
         ": pos=",
         pos,
         " arity=",
         arity,
         "."
      );

      const int64_t src_offset = x_offsets[static_cast< size_t >(src_part)];
      src_global_parts.push_back(src_idx + src_offset);
      flat_dst_parts.push_back(slot_offset + dst_idx * arity + pos);
   }

   Tensor src_global =
      src_global_parts.size() == 1 ? src_global_parts[0] : at::cat(src_global_parts, 0);
   Tensor flat_dst =
      flat_dst_parts.size() == 1 ? flat_dst_parts[0] : at::cat(flat_dst_parts, 0);
   return std::make_tuple(x_cat, src_global, flat_dst);
}

std::tuple< Tensor, Tensor, Tensor > fanin_pack_from_edges(
   const std::vector< Tensor >& rel_parts,
   const std::vector< Tensor >& edge_src_parts,
   const std::vector< Tensor >& edge_dst_parts,
   const std::vector< int64_t >& rel_part_ids,
   const std::vector< int64_t >& arity_parts,
   const std::vector< int64_t >& pos_parts,
   int64_t mode
)
{
   TORCH_CHECK(!rel_parts.empty(), "fanin_pack_from_edges requires at least one relation tensor.");
   TORCH_CHECK(
      mode == 0 || mode == 1,
      "fanin_pack_from_edges mode must be 0 (relation) or 1 (label), got ",
      mode,
      "."
   );
   const int64_t n_edge_parts = static_cast< int64_t >(edge_src_parts.size());
   TORCH_CHECK(
      static_cast< int64_t >(edge_dst_parts.size()) == n_edge_parts,
      "fanin_pack_from_edges expects edge_src_parts and edge_dst_parts with equal lengths."
   );
   TORCH_CHECK(
      static_cast< int64_t >(rel_part_ids.size()) == n_edge_parts
         && static_cast< int64_t >(arity_parts.size()) == n_edge_parts
         && static_cast< int64_t >(pos_parts.size()) == n_edge_parts,
      "fanin_pack_from_edges expects metadata arrays to match number of edge parts."
   );

   const Tensor& ref_rel = rel_parts.front();
   check_rank(ref_rel, 2, "rel_parts[0]");
   const int64_t emb = ref_rel.size(1);
   const auto ref_device = ref_rel.device();
   const auto ref_dtype = dtype_of(ref_rel);

   std::vector< int64_t > rel_offsets(rel_parts.size(), 0);
   int64_t row_offset = 0;
   for(size_t i = 0; i < rel_parts.size(); ++i) {
      const Tensor& rel = rel_parts[i];
      check_rank(rel, 2, "rel_parts[i]");
      TORCH_CHECK(
         rel.device() == ref_device,
         "fanin_pack_from_edges expects all rel_parts on the same device. Mismatch at part ",
         i,
         "."
      );
      TORCH_CHECK(
         dtype_of(rel) == ref_dtype,
         "fanin_pack_from_edges expects all rel_parts with the same dtype. Mismatch at part ",
         i,
         "."
      );
      TORCH_CHECK(
         rel.size(1) == emb,
         "fanin_pack_from_edges expects matching embedding dim across rel_parts. Part ",
         i,
         " has emb=",
         rel.size(1),
         ", expected ",
         emb,
         "."
      );
      rel_offsets[i] = row_offset;
      row_offset += rel.size(0);
   }

   Tensor rel_cat = rel_parts.size() == 1 ? rel_parts[0] : at::cat(rel_parts, 0);

   if(n_edge_parts == 0) {
      auto idx_opts = at::TensorOptions().device(ref_device).dtype(at::kLong);
      Tensor empty_idx = at::empty({0}, idx_opts);
      return std::make_tuple(rel_cat, empty_idx, empty_idx.clone());
   }

   std::vector< Tensor > flat_src_parts;
   std::vector< Tensor > dst_cat_parts;
   flat_src_parts.reserve(edge_src_parts.size());
   dst_cat_parts.reserve(edge_src_parts.size());

   for(int64_t i = 0; i < n_edge_parts; ++i) {
      const Tensor& src_idx = edge_src_parts[static_cast< size_t >(i)];
      const Tensor& dst_idx = edge_dst_parts[static_cast< size_t >(i)];
      check_int64_index(src_idx, "edge_src_parts[i]");
      check_int64_index(dst_idx, "edge_dst_parts[i]");
      TORCH_CHECK(
         src_idx.device() == ref_device && dst_idx.device() == ref_device,
         "fanin_pack_from_edges expects all edge index tensors on the same device as rel_parts."
      );
      TORCH_CHECK(
         src_idx.size(0) == dst_idx.size(0),
         "fanin_pack_from_edges expects edge src/dst lengths to match for part ",
         i,
         "."
      );

      const int64_t rel_part = rel_part_ids[static_cast< size_t >(i)];
      TORCH_CHECK(
         rel_part >= 0 && rel_part < static_cast< int64_t >(rel_parts.size()),
         "fanin_pack_from_edges rel_part_ids[",
         i,
         "] out of range: ",
         rel_part,
         "."
      );
      const int64_t rel_offset = rel_offsets[static_cast< size_t >(rel_part)];
      if(mode == 1) {
         flat_src_parts.push_back(src_idx + rel_offset);
      } else {
         const int64_t arity = arity_parts[static_cast< size_t >(i)];
         const int64_t pos = pos_parts[static_cast< size_t >(i)];
         TORCH_CHECK(
            arity > 0,
            "fanin_pack_from_edges requires arity > 0 in relation mode for edge part ",
            i,
            "."
         );
         TORCH_CHECK(
            pos >= 0 && pos < arity,
            "fanin_pack_from_edges pos out of range for edge part ",
            i,
            ": pos=",
            pos,
            " arity=",
            arity,
            "."
         );
         flat_src_parts.push_back((src_idx * arity + pos) + rel_offset);
      }
      dst_cat_parts.push_back(dst_idx);
   }

   Tensor flat_src =
      flat_src_parts.size() == 1 ? flat_src_parts[0] : at::cat(flat_src_parts, 0);
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
      "fanout_pack_multi(Tensor[] x_parts, Tensor[] src_idx_parts, Tensor[] flat_dst_parts) -> (Tensor, Tensor, Tensor)"
   );
   m.def(
      "fanout_pack_from_edges(Tensor[] x_parts, Tensor[] edge_src_parts, Tensor[] edge_dst_parts, int[] src_part_ids, int[] arity_parts, int[] pos_parts, int[] slot_offset_parts) -> (Tensor, Tensor, Tensor)"
   );
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
   m.def(
      "fanin_pack_multi(Tensor[] rel_parts, Tensor[] flat_src_parts, Tensor[] dst_idx_parts) -> (Tensor, Tensor, Tensor)"
   );
   m.def(
      "fanin_pack_from_edges(Tensor[] rel_parts, Tensor[] edge_src_parts, Tensor[] edge_dst_parts, int[] rel_part_ids, int[] arity_parts, int[] pos_parts, int mode) -> (Tensor, Tensor, Tensor)"
   );
   m.def(
      "grouped_stack_from_flat(Tensor flat, int[] slot_offsets, int[] row_sizes, int arity) -> Tensor"
   );
   m.def(
      "grouped_residual_mlp_from_flat(Tensor flat, int[] slot_offsets, int[] row_sizes, int arity, Tensor[] weight_stacks, Tensor[] bias_stacks, int[] op_kinds, int[] op_indices, int[] pointwise_codes, int truncated_dim, bool truncate_right) -> (Tensor, Tensor)"
   );
   m.def("build_info() -> str");
}

TORCH_LIBRARY_IMPL(relm_mp, CompositeImplicitAutograd, m)
{
   m.impl("fanout_pack_multi", relm::mp::fanout_pack_multi);
   m.impl("fanout_pack_from_edges", relm::mp::fanout_pack_from_edges);
   m.impl("fanout_scatter", relm::mp::fanout_scatter);
   m.impl("fanout_scatter_backward", relm::mp::fanout_scatter_backward);
   m.impl("fanin_reduce", relm::mp::fanin_reduce);
   m.impl("fanin_reduce_sum_backward", relm::mp::fanin_reduce_sum_backward);
   m.impl("fanin_reduce_logsumexp_backward", relm::mp::fanin_reduce_logsumexp_backward);
   m.impl("fanin_pack_multi", relm::mp::fanin_pack_multi);
   m.impl("fanin_pack_from_edges", relm::mp::fanin_pack_from_edges);
   m.impl("grouped_stack_from_flat", relm::mp::grouped_stack_from_flat);
   m.impl("grouped_residual_mlp_from_flat", relm::mp::grouped_residual_mlp_from_flat);
   m.impl("build_info", relm::mp::build_info);
}
