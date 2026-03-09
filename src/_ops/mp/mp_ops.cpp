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

void check_int32_or_int64_index(const Tensor& t, const char* name)
{
   check_rank(t, 1, name);
   TORCH_CHECK(
      dtype_of(t) == at::kLong || dtype_of(t) == at::kInt,
      name,
      " must have dtype torch.int64 or torch.int32."
   );
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

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
std::tuple< Tensor, Tensor > block_pointwise_cuda(
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
);
std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor >
block_pointwise_backward_cuda(
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
);
std::tuple< Tensor, Tensor > block_postnorm_ln_cuda(
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
);
std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor >
block_postnorm_ln_backward_cuda(
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
);
std::tuple< Tensor, Tensor > block_prenorm_rms_cuda(
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
);
std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor, Tensor >
block_prenorm_rms_backward_cuda(
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
);
std::tuple< Tensor, Tensor > program_silu_pair_cuda(
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
);
std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor >
program_silu_pair_backward_cuda(
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
);
std::tuple< Tensor, Tensor > program_silu_postnorm_cuda(
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
);
std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor >
program_silu_postnorm_backward_cuda(
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
);
std::tuple< Tensor, Tensor >
program_rmsnorm_silu_cuda(
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
   const Tensor& w10_stack,
   const Tensor& b10_stack,
   const Tensor& w20_stack,
   const Tensor& b20_stack,
   const Tensor& w11_stack,
   const Tensor& b11_stack,
   const Tensor& w21_stack,
   const Tensor& b21_stack
);
std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor >
program_rmsnorm_silu_backward_cuda(
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
   const Tensor& w10_stack,
   const Tensor& b10_stack,
   const Tensor& w20_stack,
   const Tensor& b20_stack,
   const Tensor& w11_stack,
   const Tensor& b11_stack,
   const Tensor& w21_stack,
   const Tensor& b21_stack
);
#endif

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

std::tuple< Tensor, Tensor > block_pointwise(
   const Tensor& x,
   const Tensor& relation_args,
   const std::vector< int64_t >& slot_offsets,
   const std::vector< int64_t >& row_sizes,
   int64_t arity,
   const Tensor& w1_stack,
   const Tensor& b1_stack,
   const Tensor& w2_stack,
   const Tensor& b2_stack,
   int64_t pointwise_code
)
{
   check_rank(x, 2, "x");
   check_int32_or_int64_index(relation_args, "relation_args");
   TORCH_CHECK(
      x.device() == relation_args.device(),
      "block_pointwise expects x and relation_args on the same device."
   );
   TORCH_CHECK(
      slot_offsets.size() == row_sizes.size(),
      "block_pointwise expects slot_offsets and row_sizes with equal lengths."
   );
   TORCH_CHECK(arity > 0, "block_pointwise expects arity > 0.");
   check_rank(w1_stack, 3, "w1_stack");
   check_rank(w2_stack, 3, "w2_stack");
   TORCH_CHECK(
      b1_stack.dim() <= 2,
      "block_pointwise expects b1_stack rank <= 2."
   );
   TORCH_CHECK(
      b2_stack.dim() <= 2,
      "block_pointwise expects b2_stack rank <= 2."
   );

   const int64_t groups = static_cast< int64_t >(slot_offsets.size());
   TORCH_CHECK(
      w1_stack.size(0) == groups && w2_stack.size(0) == groups,
      "block_pointwise weight stacks must have first dim equal to group count."
   );
   if(b1_stack.numel() > 0) {
      TORCH_CHECK(
         b1_stack.dim() == 2 && b1_stack.size(0) == groups,
         "block_pointwise b1_stack must have shape [groups, hidden] when non-empty."
      );
   }
   if(b2_stack.numel() > 0) {
      TORCH_CHECK(
         b2_stack.dim() == 2 && b2_stack.size(0) == groups,
         "block_pointwise b2_stack must have shape [groups, out_dim] when non-empty."
      );
   }

   const int64_t emb = x.size(1);
   const int64_t in_dim = emb * arity;
   TORCH_CHECK(
      w1_stack.size(2) == in_dim,
      "block_pointwise expects w1_stack.shape[-1] == arity * emb, got ",
      w1_stack.size(2),
      " vs ",
      in_dim,
      "."
   );
   const int64_t hidden = w1_stack.size(1);
   TORCH_CHECK(
      w2_stack.size(2) == hidden,
      "block_pointwise expects w2_stack.shape[-1] == hidden."
   );
   TORCH_CHECK(
      w2_stack.size(1) == in_dim,
      "block_pointwise expects w2_stack.shape[1] == arity * emb."
   );
   if(b1_stack.numel() > 0) {
      TORCH_CHECK(
         b1_stack.size(1) == hidden,
         "block_pointwise expects b1_stack.shape[1] == hidden."
      );
   }
   if(b2_stack.numel() > 0) {
      TORCH_CHECK(
         b2_stack.size(1) == in_dim,
         "block_pointwise expects b2_stack.shape[1] == arity * emb."
      );
   }

   Tensor relation_args_i64 =
      dtype_of(relation_args) == at::kLong ? relation_args : relation_args.to(at::kLong);

   std::vector< int64_t > row_offsets;
   std::vector< int64_t > out_offsets;
   row_offsets.reserve(static_cast< size_t >(groups + 1));
   out_offsets.reserve(static_cast< size_t >(groups + 1));
   row_offsets.push_back(0);
   out_offsets.push_back(0);
   for(int64_t i = 0; i < groups; ++i) {
      const int64_t n = row_sizes[static_cast< size_t >(i)];
      TORCH_CHECK(
         n >= 0,
         "block_pointwise row_sizes must be >= 0 at group ",
         i,
         "."
      );
      const int64_t start = slot_offsets[static_cast< size_t >(i)];
      const int64_t len = n * arity;
      TORCH_CHECK(
         start >= 0,
         "block_pointwise expects non-negative slot offsets."
      );
      TORCH_CHECK(
         start + len <= relation_args_i64.size(0),
         "block_pointwise slice out of bounds at group ",
         i,
         ": start=",
         start,
         " len=",
         len,
         " relation_args_rows=",
         relation_args_i64.size(0),
         "."
      );
      row_offsets.push_back(row_offsets.back() + n);
      out_offsets.push_back(out_offsets.back() + len);
   }

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
   if(x.is_cuda() && relation_args_i64.is_cuda() && w1_stack.is_cuda() && w2_stack.is_cuda()
      && (b1_stack.numel() == 0 || b1_stack.is_cuda()) && (b2_stack.numel() == 0 || b2_stack.is_cuda())
      && is_fastpath_dtype(dtype_of(x))) {
      Tensor slot_offsets_t = at::tensor(slot_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor row_offsets_t = at::tensor(row_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor out_offsets_t = at::tensor(out_offsets, relation_args_i64.options().dtype(at::kLong));
      return block_pointwise_cuda(
         x,
         relation_args_i64,
         slot_offsets_t,
         row_offsets_t,
         out_offsets_t,
         row_offsets.back(),
         out_offsets.back(),
         arity,
         w1_stack,
         b1_stack,
         w2_stack,
         b2_stack,
         pointwise_code
      );
   }
#endif

   Tensor rel_cat = x.new_empty({out_offsets.back(), emb});
   Tensor node_idx_cat = at::empty({out_offsets.back()}, relation_args_i64.options());
   for(int64_t i = 0; i < groups; ++i) {
      const int64_t row_start = row_offsets[static_cast< size_t >(i)];
      const int64_t row_end = row_offsets[static_cast< size_t >(i + 1)];
      const int64_t out_start = out_offsets[static_cast< size_t >(i)];
      const int64_t out_end = out_offsets[static_cast< size_t >(i + 1)];
      const int64_t n = row_end - row_start;
      const int64_t len = out_end - out_start;
      if(n <= 0 || len <= 0) {
         continue;
      }
      const int64_t slot = slot_offsets[static_cast< size_t >(i)];
      Tensor node_idx_i = relation_args_i64.narrow(0, slot, len);
      Tensor arg_emb_i = x.index_select(0, node_idx_i);
      Tensor x_i = arg_emb_i.view({n, in_dim});
      Tensor hidden_i;
      if(b1_stack.numel() > 0) {
         hidden_i = at::addmm(b1_stack.select(0, i), x_i, w1_stack.select(0, i).transpose(0, 1));
      } else {
         hidden_i = at::mm(x_i, w1_stack.select(0, i).transpose(0, 1));
      }
      hidden_i = apply_pointwise_code(hidden_i, pointwise_code);
      Tensor out_i;
      if(b2_stack.numel() > 0) {
         out_i = at::addmm(b2_stack.select(0, i), hidden_i, w2_stack.select(0, i).transpose(0, 1));
      } else {
         out_i = at::mm(hidden_i, w2_stack.select(0, i).transpose(0, 1));
      }
      Tensor rel_i = (x_i + out_i).contiguous().view({len, emb});
      rel_cat.narrow(0, out_start, len).copy_(rel_i);
      node_idx_cat.narrow(0, out_start, len).copy_(node_idx_i);
   }
   return std::make_tuple(rel_cat, node_idx_cat);
}

std::tuple< Tensor, Tensor > program_silu_pair(
   const Tensor& x,
   const Tensor& relation_args,
   const std::vector< int64_t >& slot_offsets,
   const std::vector< int64_t >& row_sizes,
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
   check_rank(x, 2, "x");
   check_int32_or_int64_index(relation_args, "relation_args");
   TORCH_CHECK(
      x.device() == relation_args.device(),
      "program_silu_pair expects x and relation_args on the same device."
   );
   TORCH_CHECK(
      slot_offsets.size() == row_sizes.size(),
      "program_silu_pair expects slot_offsets and row_sizes with equal lengths."
   );
   TORCH_CHECK(
      arity > 0,
      "program_silu_pair expects arity > 0."
   );
   check_rank(w10_stack, 3, "w10_stack");
   check_rank(b10_stack, 2, "b10_stack");
   check_rank(w20_stack, 3, "w20_stack");
   check_rank(b20_stack, 2, "b20_stack");
   check_rank(w11_stack, 3, "w11_stack");
   check_rank(b11_stack, 2, "b11_stack");
   check_rank(w21_stack, 3, "w21_stack");
   check_rank(b21_stack, 2, "b21_stack");

   const int64_t groups = static_cast< int64_t >(slot_offsets.size());
   TORCH_CHECK(
      w10_stack.size(0) == groups && b10_stack.size(0) == groups && w20_stack.size(0) == groups
         && b20_stack.size(0) == groups && w11_stack.size(0) == groups
         && b11_stack.size(0) == groups && w21_stack.size(0) == groups
         && b21_stack.size(0) == groups,
      "program_silu_pair parameter stacks must match group count."
   );

   const int64_t emb = x.size(1);
   const int64_t in_dim = emb * arity;
   TORCH_CHECK(
      w10_stack.size(2) == in_dim && w20_stack.size(1) == in_dim && b20_stack.size(1) == in_dim,
      "program_silu_pair stage-1 dims must match arity * emb."
   );
   TORCH_CHECK(
      w11_stack.size(2) == in_dim && w21_stack.size(1) == in_dim && b21_stack.size(1) == in_dim,
      "program_silu_pair stage-2 dims must match arity * emb."
   );
   TORCH_CHECK(
      b10_stack.size(1) == w10_stack.size(1) && w20_stack.size(2) == w10_stack.size(1),
      "program_silu_pair stage-1 hidden dims must match."
   );
   TORCH_CHECK(
      b11_stack.size(1) == w11_stack.size(1) && w21_stack.size(2) == w11_stack.size(1),
      "program_silu_pair stage-2 hidden dims must match."
   );

   Tensor relation_args_i64 =
      dtype_of(relation_args) == at::kLong ? relation_args : relation_args.to(at::kLong);

   std::vector< int64_t > row_offsets;
   std::vector< int64_t > out_offsets;
   row_offsets.reserve(static_cast< size_t >(groups + 1));
   out_offsets.reserve(static_cast< size_t >(groups + 1));
   row_offsets.push_back(0);
   out_offsets.push_back(0);
   for(int64_t i = 0; i < groups; ++i) {
      const int64_t n = row_sizes[static_cast< size_t >(i)];
      TORCH_CHECK(
         n >= 0,
         "program_silu_pair row_sizes must be >= 0 at group ",
         i,
         "."
      );
      const int64_t start = slot_offsets[static_cast< size_t >(i)];
      const int64_t len = n * arity;
      TORCH_CHECK(
         start >= 0,
         "program_silu_pair expects non-negative slot offsets."
      );
      TORCH_CHECK(
         start + len <= relation_args_i64.size(0),
         "program_silu_pair slice out of bounds at group ",
         i,
         "."
      );
      row_offsets.push_back(row_offsets.back() + n);
      out_offsets.push_back(out_offsets.back() + len);
   }

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
   if(x.is_cuda() && relation_args_i64.is_cuda() && w10_stack.is_cuda() && b10_stack.is_cuda()
      && w20_stack.is_cuda() && b20_stack.is_cuda() && w11_stack.is_cuda() && b11_stack.is_cuda()
      && w21_stack.is_cuda() && b21_stack.is_cuda() && is_fastpath_dtype(dtype_of(x))) {
      Tensor slot_offsets_t = at::tensor(slot_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor row_offsets_t = at::tensor(row_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor out_offsets_t = at::tensor(out_offsets, relation_args_i64.options().dtype(at::kLong));
      return program_silu_pair_cuda(
         x,
         relation_args_i64,
         slot_offsets_t,
         row_offsets_t,
         out_offsets_t,
         row_offsets.back(),
         out_offsets.back(),
         arity,
         w10_stack,
         b10_stack,
         w20_stack,
         b20_stack,
         w11_stack,
         b11_stack,
         w21_stack,
         b21_stack
      );
   }
#endif

   Tensor rel_cat = x.new_empty({out_offsets.back(), emb});
   Tensor node_idx_cat = at::empty({out_offsets.back()}, relation_args_i64.options());
   for(int64_t i = 0; i < groups; ++i) {
      const int64_t row_start = row_offsets[static_cast< size_t >(i)];
      const int64_t row_end = row_offsets[static_cast< size_t >(i + 1)];
      const int64_t out_start = out_offsets[static_cast< size_t >(i)];
      const int64_t out_end = out_offsets[static_cast< size_t >(i + 1)];
      const int64_t n = row_end - row_start;
      const int64_t len = out_end - out_start;
      if(n <= 0 || len <= 0) {
         continue;
      }
      const int64_t slot = slot_offsets[static_cast< size_t >(i)];
      Tensor node_idx_i = relation_args_i64.narrow(0, slot, len);
      Tensor arg_emb_i = x.index_select(0, node_idx_i);
      Tensor x_i = arg_emb_i.view({n, in_dim});
      Tensor stage1 = at::addmm(b10_stack.select(0, i), x_i, w10_stack.select(0, i).transpose(0, 1));
      stage1 = at::silu(stage1);
      stage1 = at::addmm(b20_stack.select(0, i), stage1, w20_stack.select(0, i).transpose(0, 1));
      Tensor stage2 = at::addmm(b11_stack.select(0, i), stage1, w11_stack.select(0, i).transpose(0, 1));
      stage2 = at::silu(stage2);
      stage2 = at::addmm(b21_stack.select(0, i), stage2, w21_stack.select(0, i).transpose(0, 1));
      Tensor rel_i = (x_i + stage2).contiguous().view({len, emb});
      rel_cat.narrow(0, out_start, len).copy_(rel_i);
      node_idx_cat.narrow(0, out_start, len).copy_(node_idx_i);
   }
   return std::make_tuple(rel_cat, node_idx_cat);
}

std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor >
program_silu_pair_backward(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const std::vector< int64_t >& slot_offsets,
   const std::vector< int64_t >& row_sizes,
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
   check_rank(grad_rel, 2, "grad_rel");
   check_rank(x, 2, "x");
   check_int32_or_int64_index(relation_args, "relation_args");
   TORCH_CHECK(
      grad_rel.device() == x.device() && x.device() == relation_args.device(),
      "program_silu_pair_backward expects grad_rel, x, and relation_args on the same device."
   );
   TORCH_CHECK(
      slot_offsets.size() == row_sizes.size(),
      "program_silu_pair_backward expects slot_offsets and row_sizes with equal lengths."
   );
   TORCH_CHECK(
      arity > 0,
      "program_silu_pair_backward expects arity > 0."
   );
   const int64_t groups = static_cast< int64_t >(slot_offsets.size());
   const int64_t emb = x.size(1);
   const int64_t in_dim = emb * arity;
   TORCH_CHECK(
      grad_rel.size(1) == emb,
      "program_silu_pair_backward expects grad_rel.shape[1] == emb."
   );
   TORCH_CHECK(
      w10_stack.size(0) == groups && b10_stack.size(0) == groups && w20_stack.size(0) == groups
         && b20_stack.size(0) == groups && w11_stack.size(0) == groups
         && b11_stack.size(0) == groups && w21_stack.size(0) == groups
         && b21_stack.size(0) == groups,
      "program_silu_pair_backward parameter stacks must match group count."
   );
   TORCH_CHECK(
      w10_stack.size(2) == in_dim && w20_stack.size(1) == in_dim && b20_stack.size(1) == in_dim
         && w11_stack.size(2) == in_dim && w21_stack.size(1) == in_dim && b21_stack.size(1) == in_dim,
      "program_silu_pair_backward stack dims do not match arity * emb."
   );

   Tensor relation_args_i64 =
      dtype_of(relation_args) == at::kLong ? relation_args : relation_args.to(at::kLong);

   std::vector< int64_t > row_offsets;
   std::vector< int64_t > out_offsets;
   row_offsets.reserve(static_cast< size_t >(groups + 1));
   out_offsets.reserve(static_cast< size_t >(groups + 1));
   row_offsets.push_back(0);
   out_offsets.push_back(0);
   for(int64_t i = 0; i < groups; ++i) {
      const int64_t n = row_sizes[static_cast< size_t >(i)];
      TORCH_CHECK(
         n >= 0,
         "program_silu_pair_backward row_sizes must be >= 0 at group ",
         i,
         "."
      );
      const int64_t start = slot_offsets[static_cast< size_t >(i)];
      const int64_t len = n * arity;
      TORCH_CHECK(
         start >= 0,
         "program_silu_pair_backward expects non-negative slot offsets."
      );
      TORCH_CHECK(
         start + len <= relation_args_i64.size(0),
         "program_silu_pair_backward slice out of bounds at group ",
         i,
         "."
      );
      row_offsets.push_back(row_offsets.back() + n);
      out_offsets.push_back(out_offsets.back() + len);
   }
   TORCH_CHECK(
      grad_rel.size(0) == out_offsets.back(),
      "program_silu_pair_backward expects grad_rel rows to match packed slot count."
   );

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
   if(grad_rel.is_cuda() && x.is_cuda() && relation_args_i64.is_cuda() && w10_stack.is_cuda()
      && b10_stack.is_cuda() && w20_stack.is_cuda() && b20_stack.is_cuda()
      && w11_stack.is_cuda() && b11_stack.is_cuda() && w21_stack.is_cuda() && b21_stack.is_cuda()
      && is_fastpath_dtype(dtype_of(x))) {
      Tensor slot_offsets_t = at::tensor(slot_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor row_offsets_t = at::tensor(row_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor out_offsets_t = at::tensor(out_offsets, relation_args_i64.options().dtype(at::kLong));
      return program_silu_pair_backward_cuda(
         grad_rel,
         x,
         relation_args_i64,
         slot_offsets_t,
         row_offsets_t,
         out_offsets_t,
         row_offsets.back(),
         arity,
         w10_stack,
         b10_stack,
         w20_stack,
         b20_stack,
         w11_stack,
         b11_stack,
         w21_stack,
         b21_stack
      );
   }
#endif
   TORCH_CHECK(
      false,
      "program_silu_pair_backward currently supports only CUDA float32/float64 tensors."
   );
}

std::tuple< Tensor, Tensor > program_silu_postnorm(
   const Tensor& x,
   const Tensor& relation_args,
   const std::vector< int64_t >& slot_offsets,
   const std::vector< int64_t >& row_sizes,
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
   check_rank(x, 2, "x");
   check_int32_or_int64_index(relation_args, "relation_args");
   TORCH_CHECK(
      x.device() == relation_args.device(),
      "program_silu_postnorm expects x and relation_args on the same device."
   );
   TORCH_CHECK(
      slot_offsets.size() == row_sizes.size(),
      "program_silu_postnorm expects slot_offsets and row_sizes with equal lengths."
   );
   TORCH_CHECK(
      arity > 0,
      "program_silu_postnorm expects arity > 0."
   );
   check_rank(w10_stack, 3, "w10_stack");
   check_rank(b10_stack, 2, "b10_stack");
   check_rank(w20_stack, 3, "w20_stack");
   check_rank(b20_stack, 2, "b20_stack");
   check_rank(w11_stack, 3, "w11_stack");
   check_rank(b11_stack, 2, "b11_stack");
   check_rank(w21_stack, 3, "w21_stack");
   check_rank(b21_stack, 2, "b21_stack");

   const int64_t groups = static_cast< int64_t >(slot_offsets.size());
   TORCH_CHECK(
      w10_stack.size(0) == groups && b10_stack.size(0) == groups && w20_stack.size(0) == groups
         && b20_stack.size(0) == groups && w11_stack.size(0) == groups
         && b11_stack.size(0) == groups && w21_stack.size(0) == groups
         && b21_stack.size(0) == groups,
      "program_silu_postnorm parameter stacks must match group count."
   );

   const int64_t emb = x.size(1);
   const int64_t in_dim = emb * arity;
   TORCH_CHECK(
      w10_stack.size(2) == in_dim && w20_stack.size(1) == in_dim && b20_stack.size(1) == in_dim,
      "program_silu_postnorm stage-1 dims must match arity * emb."
   );
   TORCH_CHECK(
      w11_stack.size(2) == in_dim && w21_stack.size(1) == in_dim && b21_stack.size(1) == in_dim,
      "program_silu_postnorm stage-2 dims must match arity * emb."
   );
   TORCH_CHECK(
      b10_stack.size(1) == w10_stack.size(1) && w20_stack.size(2) == w10_stack.size(1),
      "program_silu_postnorm stage-1 hidden dims must match."
   );
   TORCH_CHECK(
      b11_stack.size(1) == w11_stack.size(1) && w21_stack.size(2) == w11_stack.size(1),
      "program_silu_postnorm stage-2 hidden dims must match."
   );
   if(ln_weight_stack.numel() > 0) {
      check_rank(ln_weight_stack, 2, "ln_weight_stack");
      TORCH_CHECK(
         ln_weight_stack.size(0) == groups && ln_weight_stack.size(1) == in_dim,
         "program_silu_postnorm ln_weight_stack must have shape [groups, arity * emb]."
      );
   }
   if(ln_bias_stack.numel() > 0) {
      check_rank(ln_bias_stack, 2, "ln_bias_stack");
      TORCH_CHECK(
         ln_bias_stack.size(0) == groups && ln_bias_stack.size(1) == in_dim,
         "program_silu_postnorm ln_bias_stack must have shape [groups, arity * emb]."
      );
   }

   Tensor relation_args_i64 =
      dtype_of(relation_args) == at::kLong ? relation_args : relation_args.to(at::kLong);

   std::vector< int64_t > row_offsets;
   std::vector< int64_t > out_offsets;
   row_offsets.reserve(static_cast< size_t >(groups + 1));
   out_offsets.reserve(static_cast< size_t >(groups + 1));
   row_offsets.push_back(0);
   out_offsets.push_back(0);
   for(int64_t i = 0; i < groups; ++i) {
      const int64_t n = row_sizes[static_cast< size_t >(i)];
      TORCH_CHECK(
         n >= 0,
         "program_silu_postnorm row_sizes must be >= 0 at group ",
         i,
         "."
      );
      const int64_t start = slot_offsets[static_cast< size_t >(i)];
      const int64_t len = n * arity;
      TORCH_CHECK(
         start >= 0,
         "program_silu_postnorm expects non-negative slot offsets."
      );
      TORCH_CHECK(
         start + len <= relation_args_i64.size(0),
         "program_silu_postnorm slice out of bounds at group ",
         i,
         "."
      );
      row_offsets.push_back(row_offsets.back() + n);
      out_offsets.push_back(out_offsets.back() + len);
   }

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
   if(x.is_cuda() && relation_args_i64.is_cuda() && w10_stack.is_cuda() && b10_stack.is_cuda()
      && w20_stack.is_cuda() && b20_stack.is_cuda() && w11_stack.is_cuda() && b11_stack.is_cuda()
      && w21_stack.is_cuda() && b21_stack.is_cuda()
      && (ln_weight_stack.numel() == 0 || ln_weight_stack.is_cuda())
      && (ln_bias_stack.numel() == 0 || ln_bias_stack.is_cuda())
      && is_fastpath_dtype(dtype_of(x))) {
      Tensor slot_offsets_t = at::tensor(slot_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor row_offsets_t = at::tensor(row_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor out_offsets_t = at::tensor(out_offsets, relation_args_i64.options().dtype(at::kLong));
      return program_silu_postnorm_cuda(
         x,
         relation_args_i64,
         slot_offsets_t,
         row_offsets_t,
         out_offsets_t,
         row_offsets.back(),
         out_offsets.back(),
         arity,
         w10_stack,
         b10_stack,
         w20_stack,
         b20_stack,
         w11_stack,
         b11_stack,
         w21_stack,
         b21_stack,
         ln_weight_stack,
         ln_bias_stack,
         ln_eps
      );
   }
#endif
   TORCH_CHECK(
      false,
      "program_silu_postnorm currently supports only CUDA float32/float64 tensors."
   );
}

std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor >
program_silu_postnorm_backward(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const std::vector< int64_t >& slot_offsets,
   const std::vector< int64_t >& row_sizes,
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
   check_rank(grad_rel, 2, "grad_rel");
   check_rank(x, 2, "x");
   check_int32_or_int64_index(relation_args, "relation_args");
   TORCH_CHECK(
      grad_rel.device() == x.device() && x.device() == relation_args.device(),
      "program_silu_postnorm_backward expects grad_rel, x, and relation_args on the same device."
   );
   TORCH_CHECK(
      slot_offsets.size() == row_sizes.size(),
      "program_silu_postnorm_backward expects slot_offsets and row_sizes with equal lengths."
   );
   TORCH_CHECK(
      arity > 0,
      "program_silu_postnorm_backward expects arity > 0."
   );

   const int64_t groups = static_cast< int64_t >(slot_offsets.size());
   const int64_t emb = x.size(1);
   const int64_t in_dim = emb * arity;
   TORCH_CHECK(
      grad_rel.size(1) == emb,
      "program_silu_postnorm_backward expects grad_rel.shape[1] == emb."
   );
   TORCH_CHECK(
      w10_stack.size(0) == groups && b10_stack.size(0) == groups && w20_stack.size(0) == groups
         && b20_stack.size(0) == groups && w11_stack.size(0) == groups
         && b11_stack.size(0) == groups && w21_stack.size(0) == groups
         && b21_stack.size(0) == groups,
      "program_silu_postnorm_backward parameter stacks must match group count."
   );
   TORCH_CHECK(
      w10_stack.size(2) == in_dim && w20_stack.size(1) == in_dim && b20_stack.size(1) == in_dim
         && w11_stack.size(2) == in_dim && w21_stack.size(1) == in_dim && b21_stack.size(1) == in_dim,
      "program_silu_postnorm_backward stack dims do not match arity * emb."
   );
   if(ln_weight_stack.numel() > 0) {
      TORCH_CHECK(
         ln_weight_stack.size(0) == groups && ln_weight_stack.size(1) == in_dim,
         "program_silu_postnorm_backward ln_weight_stack must have shape [groups, arity * emb]."
      );
   }
   if(ln_bias_stack.numel() > 0) {
      TORCH_CHECK(
         ln_bias_stack.size(0) == groups && ln_bias_stack.size(1) == in_dim,
         "program_silu_postnorm_backward ln_bias_stack must have shape [groups, arity * emb]."
      );
   }

   Tensor relation_args_i64 =
      dtype_of(relation_args) == at::kLong ? relation_args : relation_args.to(at::kLong);

   std::vector< int64_t > row_offsets;
   std::vector< int64_t > out_offsets;
   row_offsets.reserve(static_cast< size_t >(groups + 1));
   out_offsets.reserve(static_cast< size_t >(groups + 1));
   row_offsets.push_back(0);
   out_offsets.push_back(0);
   for(int64_t i = 0; i < groups; ++i) {
      const int64_t n = row_sizes[static_cast< size_t >(i)];
      TORCH_CHECK(
         n >= 0,
         "program_silu_postnorm_backward row_sizes must be >= 0 at group ",
         i,
         "."
      );
      const int64_t start = slot_offsets[static_cast< size_t >(i)];
      const int64_t len = n * arity;
      TORCH_CHECK(
         start >= 0,
         "program_silu_postnorm_backward expects non-negative slot offsets."
      );
      TORCH_CHECK(
         start + len <= relation_args_i64.size(0),
         "program_silu_postnorm_backward slice out of bounds at group ",
         i,
         "."
      );
      row_offsets.push_back(row_offsets.back() + n);
      out_offsets.push_back(out_offsets.back() + len);
   }
   TORCH_CHECK(
      grad_rel.size(0) == out_offsets.back(),
      "program_silu_postnorm_backward expects grad_rel rows to match packed slot count."
   );

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
   if(grad_rel.is_cuda() && x.is_cuda() && relation_args_i64.is_cuda() && w10_stack.is_cuda()
      && b10_stack.is_cuda() && w20_stack.is_cuda() && b20_stack.is_cuda()
      && w11_stack.is_cuda() && b11_stack.is_cuda() && w21_stack.is_cuda() && b21_stack.is_cuda()
      && (ln_weight_stack.numel() == 0 || ln_weight_stack.is_cuda())
      && (ln_bias_stack.numel() == 0 || ln_bias_stack.is_cuda())
      && is_fastpath_dtype(dtype_of(x))) {
      Tensor slot_offsets_t = at::tensor(slot_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor row_offsets_t = at::tensor(row_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor out_offsets_t = at::tensor(out_offsets, relation_args_i64.options().dtype(at::kLong));
      return program_silu_postnorm_backward_cuda(
         grad_rel,
         x,
         relation_args_i64,
         slot_offsets_t,
         row_offsets_t,
         out_offsets_t,
         row_offsets.back(),
         arity,
         w10_stack,
         b10_stack,
         w20_stack,
         b20_stack,
         w11_stack,
         b11_stack,
         w21_stack,
         b21_stack,
         ln_weight_stack,
         ln_bias_stack,
         ln_eps
      );
   }
#endif
   TORCH_CHECK(
      false,
      "program_silu_postnorm_backward currently supports only CUDA float32/float64 tensors."
   );
}

std::tuple< Tensor, Tensor >
program_rmsnorm_silu(
   const Tensor& x,
   const Tensor& relation_args,
   const std::vector< int64_t >& slot_offsets,
   const std::vector< int64_t >& row_sizes,
   int64_t arity,
   const Tensor& rms_weight_stack,
   double rms_eps,
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
   check_rank(x, 2, "x");
   check_int32_or_int64_index(relation_args, "relation_args");
   TORCH_CHECK(
      x.device() == relation_args.device(),
      "program_rmsnorm_silu expects x and relation_args on the same device."
   );
   TORCH_CHECK(
      slot_offsets.size() == row_sizes.size(),
      "program_rmsnorm_silu expects slot_offsets and row_sizes with equal lengths."
   );
   TORCH_CHECK(
      arity > 0,
      "program_rmsnorm_silu expects arity > 0."
   );
   check_rank(w10_stack, 3, "w10_stack");
   check_rank(b10_stack, 2, "b10_stack");
   check_rank(w20_stack, 3, "w20_stack");
   check_rank(b20_stack, 2, "b20_stack");
   check_rank(w11_stack, 3, "w11_stack");
   check_rank(b11_stack, 2, "b11_stack");
   check_rank(w21_stack, 3, "w21_stack");
   check_rank(b21_stack, 2, "b21_stack");

   const int64_t groups = static_cast< int64_t >(slot_offsets.size());
   TORCH_CHECK(
      w10_stack.size(0) == groups && b10_stack.size(0) == groups && w20_stack.size(0) == groups
         && b20_stack.size(0) == groups && w11_stack.size(0) == groups
         && b11_stack.size(0) == groups && w21_stack.size(0) == groups
         && b21_stack.size(0) == groups,
      "program_rmsnorm_silu parameter stacks must match group count."
   );

   const int64_t emb = x.size(1);
   const int64_t in_dim = emb * arity;
   TORCH_CHECK(
      w10_stack.size(2) == in_dim && w20_stack.size(1) == in_dim && b20_stack.size(1) == in_dim,
      "program_rmsnorm_silu stage-1 dims must match arity * emb."
   );
   TORCH_CHECK(
      w11_stack.size(2) == in_dim && w21_stack.size(1) == in_dim && b21_stack.size(1) == in_dim,
      "program_rmsnorm_silu stage-2 dims must match arity * emb."
   );
   TORCH_CHECK(
      b10_stack.size(1) == w10_stack.size(1) && w20_stack.size(2) == w10_stack.size(1),
      "program_rmsnorm_silu stage-1 hidden dims must match."
   );
   TORCH_CHECK(
      b11_stack.size(1) == w11_stack.size(1) && w21_stack.size(2) == w11_stack.size(1),
      "program_rmsnorm_silu stage-2 hidden dims must match."
   );
   if(rms_weight_stack.numel() > 0) {
      check_rank(rms_weight_stack, 2, "rms_weight_stack");
      TORCH_CHECK(
         rms_weight_stack.size(0) == groups && rms_weight_stack.size(1) == in_dim,
         "program_rmsnorm_silu rms_weight_stack must have shape [groups, arity * emb]."
      );
   }

   Tensor relation_args_i64 =
      dtype_of(relation_args) == at::kLong ? relation_args : relation_args.to(at::kLong);

   std::vector< int64_t > row_offsets;
   std::vector< int64_t > out_offsets;
   row_offsets.reserve(static_cast< size_t >(groups + 1));
   out_offsets.reserve(static_cast< size_t >(groups + 1));
   row_offsets.push_back(0);
   out_offsets.push_back(0);
   for(int64_t i = 0; i < groups; ++i) {
      const int64_t n = row_sizes[static_cast< size_t >(i)];
      TORCH_CHECK(
         n >= 0,
         "program_rmsnorm_silu row_sizes must be >= 0 at group ",
         i,
         "."
      );
      const int64_t start = slot_offsets[static_cast< size_t >(i)];
      const int64_t len = n * arity;
      TORCH_CHECK(
         start >= 0,
         "program_rmsnorm_silu expects non-negative slot offsets."
      );
      TORCH_CHECK(
         start + len <= relation_args_i64.size(0),
         "program_rmsnorm_silu slice out of bounds at group ",
         i,
         "."
      );
      row_offsets.push_back(row_offsets.back() + n);
      out_offsets.push_back(out_offsets.back() + len);
   }

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
   if(x.is_cuda() && relation_args_i64.is_cuda() && w10_stack.is_cuda() && b10_stack.is_cuda()
      && w20_stack.is_cuda() && b20_stack.is_cuda() && w11_stack.is_cuda() && b11_stack.is_cuda()
      && w21_stack.is_cuda() && b21_stack.is_cuda()
      && (rms_weight_stack.numel() == 0 || rms_weight_stack.is_cuda())
      && is_fastpath_dtype(dtype_of(x))) {
      Tensor slot_offsets_t = at::tensor(slot_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor row_offsets_t = at::tensor(row_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor out_offsets_t = at::tensor(out_offsets, relation_args_i64.options().dtype(at::kLong));
      return program_rmsnorm_silu_cuda(
         x,
         relation_args_i64,
         slot_offsets_t,
         row_offsets_t,
         out_offsets_t,
         row_offsets.back(),
         out_offsets.back(),
         arity,
         rms_weight_stack,
         rms_eps,
         w10_stack,
         b10_stack,
         w20_stack,
         b20_stack,
         w11_stack,
         b11_stack,
         w21_stack,
         b21_stack
      );
   }
#endif
   TORCH_CHECK(
      false,
      "program_rmsnorm_silu currently supports only CUDA float32/float64 tensors."
   );
}

std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor >
program_rmsnorm_silu_backward(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const std::vector< int64_t >& slot_offsets,
   const std::vector< int64_t >& row_sizes,
   int64_t arity,
   const Tensor& rms_weight_stack,
   double rms_eps,
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
   check_rank(grad_rel, 2, "grad_rel");
   check_rank(x, 2, "x");
   check_int32_or_int64_index(relation_args, "relation_args");
   TORCH_CHECK(
      grad_rel.device() == x.device() && x.device() == relation_args.device(),
      "program_rmsnorm_silu_backward expects grad_rel, x, and relation_args on the same device."
   );
   TORCH_CHECK(
      slot_offsets.size() == row_sizes.size(),
      "program_rmsnorm_silu_backward expects slot_offsets and row_sizes with equal lengths."
   );
   TORCH_CHECK(
      arity > 0,
      "program_rmsnorm_silu_backward expects arity > 0."
   );

   const int64_t groups = static_cast< int64_t >(slot_offsets.size());
   const int64_t emb = x.size(1);
   const int64_t in_dim = emb * arity;
   TORCH_CHECK(
      grad_rel.size(1) == emb,
      "program_rmsnorm_silu_backward expects grad_rel.shape[1] == emb."
   );
   TORCH_CHECK(
      w10_stack.size(0) == groups && b10_stack.size(0) == groups && w20_stack.size(0) == groups
         && b20_stack.size(0) == groups && w11_stack.size(0) == groups
         && b11_stack.size(0) == groups && w21_stack.size(0) == groups
         && b21_stack.size(0) == groups,
      "program_rmsnorm_silu_backward parameter stacks must match group count."
   );
   TORCH_CHECK(
      w10_stack.size(2) == in_dim && w20_stack.size(1) == in_dim && b20_stack.size(1) == in_dim
         && w11_stack.size(2) == in_dim && w21_stack.size(1) == in_dim && b21_stack.size(1) == in_dim,
      "program_rmsnorm_silu_backward stack dims do not match arity * emb."
   );
   if(rms_weight_stack.numel() > 0) {
      TORCH_CHECK(
         rms_weight_stack.size(0) == groups && rms_weight_stack.size(1) == in_dim,
         "program_rmsnorm_silu_backward rms_weight_stack must have shape [groups, arity * emb]."
      );
   }

   Tensor relation_args_i64 =
      dtype_of(relation_args) == at::kLong ? relation_args : relation_args.to(at::kLong);

   std::vector< int64_t > row_offsets;
   std::vector< int64_t > out_offsets;
   row_offsets.reserve(static_cast< size_t >(groups + 1));
   out_offsets.reserve(static_cast< size_t >(groups + 1));
   row_offsets.push_back(0);
   out_offsets.push_back(0);
   for(int64_t i = 0; i < groups; ++i) {
      const int64_t n = row_sizes[static_cast< size_t >(i)];
      TORCH_CHECK(
         n >= 0,
         "program_rmsnorm_silu_backward row_sizes must be >= 0 at group ",
         i,
         "."
      );
      const int64_t start = slot_offsets[static_cast< size_t >(i)];
      const int64_t len = n * arity;
      TORCH_CHECK(
         start >= 0,
         "program_rmsnorm_silu_backward expects non-negative slot offsets."
      );
      TORCH_CHECK(
         start + len <= relation_args_i64.size(0),
         "program_rmsnorm_silu_backward slice out of bounds at group ",
         i,
         "."
      );
      row_offsets.push_back(row_offsets.back() + n);
      out_offsets.push_back(out_offsets.back() + len);
   }
   TORCH_CHECK(
      grad_rel.size(0) == out_offsets.back(),
      "program_rmsnorm_silu_backward expects grad_rel rows to match packed slot count."
   );

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
   if(grad_rel.is_cuda() && x.is_cuda() && relation_args_i64.is_cuda() && w10_stack.is_cuda()
      && b10_stack.is_cuda() && w20_stack.is_cuda() && b20_stack.is_cuda()
      && w11_stack.is_cuda() && b11_stack.is_cuda() && w21_stack.is_cuda() && b21_stack.is_cuda()
      && (rms_weight_stack.numel() == 0 || rms_weight_stack.is_cuda())
      && is_fastpath_dtype(dtype_of(x))) {
      Tensor slot_offsets_t = at::tensor(slot_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor row_offsets_t = at::tensor(row_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor out_offsets_t = at::tensor(out_offsets, relation_args_i64.options().dtype(at::kLong));
      return program_rmsnorm_silu_backward_cuda(
         grad_rel,
         x,
         relation_args_i64,
         slot_offsets_t,
         row_offsets_t,
         out_offsets_t,
         row_offsets.back(),
         arity,
         rms_weight_stack,
         rms_eps,
         w10_stack,
         b10_stack,
         w20_stack,
         b20_stack,
         w11_stack,
         b11_stack,
         w21_stack,
         b21_stack
      );
   }
#endif
   TORCH_CHECK(
      false,
      "program_rmsnorm_silu_backward currently supports only CUDA float32/float64 tensors."
   );
}

std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor >
block_pointwise_backward(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const std::vector< int64_t >& slot_offsets,
   const std::vector< int64_t >& row_sizes,
   int64_t arity,
   const Tensor& w1_stack,
   const Tensor& b1_stack,
   const Tensor& w2_stack,
   const Tensor& b2_stack,
   int64_t pointwise_code
)
{
   check_rank(grad_rel, 2, "grad_rel");
   check_rank(x, 2, "x");
   check_int32_or_int64_index(relation_args, "relation_args");
   TORCH_CHECK(
      x.device() == relation_args.device() && grad_rel.device() == x.device(),
      "block_pointwise_backward expects grad_rel, x, and relation_args on the same device."
   );
   TORCH_CHECK(
      slot_offsets.size() == row_sizes.size(),
      "block_pointwise_backward expects slot_offsets and row_sizes with equal lengths."
   );
   TORCH_CHECK(arity > 0, "block_pointwise_backward expects arity > 0.");
   check_rank(w1_stack, 3, "w1_stack");
   check_rank(w2_stack, 3, "w2_stack");

   const int64_t groups = static_cast< int64_t >(slot_offsets.size());
   const int64_t emb = x.size(1);
   const int64_t in_dim = emb * arity;
   TORCH_CHECK(
      grad_rel.size(1) == emb,
      "block_pointwise_backward expects grad_rel.shape[1] == emb."
   );
   TORCH_CHECK(
      w1_stack.size(0) == groups && w2_stack.size(0) == groups,
      "block_pointwise_backward weight stacks must match group count."
   );
   TORCH_CHECK(
      w1_stack.size(2) == in_dim && w2_stack.size(1) == in_dim,
      "block_pointwise_backward weight stack dims do not match arity * emb."
   );

   std::vector< int64_t > row_offsets;
   std::vector< int64_t > out_offsets;
   row_offsets.reserve(static_cast< size_t >(groups + 1));
   out_offsets.reserve(static_cast< size_t >(groups + 1));
   row_offsets.push_back(0);
   out_offsets.push_back(0);
   for(int64_t i = 0; i < groups; ++i) {
      const int64_t n = row_sizes[static_cast< size_t >(i)];
      TORCH_CHECK(
         n >= 0,
         "block_pointwise_backward row_sizes must be >= 0 at group ",
         i,
         "."
      );
      const int64_t start = slot_offsets[static_cast< size_t >(i)];
      const int64_t len = n * arity;
      TORCH_CHECK(
         start >= 0,
         "block_pointwise_backward expects non-negative slot offsets."
      );
      TORCH_CHECK(
         start + len <= relation_args.size(0),
         "block_pointwise_backward slice out of bounds at group ",
         i,
         "."
      );
      row_offsets.push_back(row_offsets.back() + n);
      out_offsets.push_back(out_offsets.back() + len);
   }
   TORCH_CHECK(
      grad_rel.size(0) == out_offsets.back(),
      "block_pointwise_backward expects grad_rel rows to match packed slot count."
   );

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
   if(grad_rel.is_cuda() && x.is_cuda() && relation_args.is_cuda() && w1_stack.is_cuda()
      && w2_stack.is_cuda() && (b1_stack.numel() == 0 || b1_stack.is_cuda())
      && (b2_stack.numel() == 0 || b2_stack.is_cuda()) && is_fastpath_dtype(dtype_of(grad_rel))) {
      Tensor relation_args_i64 =
         dtype_of(relation_args) == at::kLong ? relation_args : relation_args.to(at::kLong);
      Tensor slot_offsets_t = at::tensor(slot_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor row_offsets_t = at::tensor(row_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor out_offsets_t = at::tensor(out_offsets, relation_args_i64.options().dtype(at::kLong));
      return block_pointwise_backward_cuda(
         grad_rel,
         x,
         relation_args_i64,
         slot_offsets_t,
         row_offsets_t,
         out_offsets_t,
         row_offsets.back(),
         arity,
         w1_stack,
         b1_stack,
         w2_stack,
         b2_stack,
         pointwise_code
      );
   }
#endif

   TORCH_CHECK(
      false,
      "block_pointwise_backward currently supports only CUDA float32/float64 tensors."
   );
}

std::tuple< Tensor, Tensor > block_postnorm_ln(
   const Tensor& x,
   const Tensor& relation_args,
   const std::vector< int64_t >& slot_offsets,
   const std::vector< int64_t >& row_sizes,
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
   check_rank(x, 2, "x");
   check_int32_or_int64_index(relation_args, "relation_args");
   TORCH_CHECK(
      x.device() == relation_args.device(),
      "block_postnorm_ln expects x and relation_args on the same device."
   );
   TORCH_CHECK(
      slot_offsets.size() == row_sizes.size(),
      "block_postnorm_ln expects slot_offsets and row_sizes with equal lengths."
   );
   TORCH_CHECK(arity > 0, "block_postnorm_ln expects arity > 0.");
   check_rank(w1_stack, 3, "w1_stack");
   check_rank(w2_stack, 3, "w2_stack");

   const int64_t groups = static_cast< int64_t >(slot_offsets.size());
   const int64_t emb = x.size(1);
   const int64_t in_dim = emb * arity;
   TORCH_CHECK(
      w1_stack.size(0) == groups && w2_stack.size(0) == groups,
      "block_postnorm_ln weight stacks must match group count."
   );
   TORCH_CHECK(
      w1_stack.size(2) == in_dim && w2_stack.size(1) == in_dim,
      "block_postnorm_ln weight stack dims do not match arity * emb."
   );
   if(ln_weight_stack.numel() > 0) {
      check_rank(ln_weight_stack, 2, "ln_weight_stack");
      TORCH_CHECK(
         ln_weight_stack.size(0) == groups && ln_weight_stack.size(1) == in_dim,
         "block_postnorm_ln ln_weight_stack must have shape [groups, arity * emb]."
      );
   }
   if(ln_bias_stack.numel() > 0) {
      check_rank(ln_bias_stack, 2, "ln_bias_stack");
      TORCH_CHECK(
         ln_bias_stack.size(0) == groups && ln_bias_stack.size(1) == in_dim,
         "block_postnorm_ln ln_bias_stack must have shape [groups, arity * emb]."
      );
   }

   Tensor relation_args_i64 =
      dtype_of(relation_args) == at::kLong ? relation_args : relation_args.to(at::kLong);
   std::vector< int64_t > row_offsets;
   std::vector< int64_t > out_offsets;
   row_offsets.reserve(static_cast< size_t >(groups + 1));
   out_offsets.reserve(static_cast< size_t >(groups + 1));
   row_offsets.push_back(0);
   out_offsets.push_back(0);
   for(int64_t i = 0; i < groups; ++i) {
      const int64_t n = row_sizes[static_cast< size_t >(i)];
      TORCH_CHECK(
         n >= 0,
         "block_postnorm_ln row_sizes must be >= 0 at group ",
         i,
         "."
      );
      const int64_t start = slot_offsets[static_cast< size_t >(i)];
      const int64_t len = n * arity;
      TORCH_CHECK(start >= 0, "block_postnorm_ln expects non-negative slot offsets.");
      TORCH_CHECK(
         start + len <= relation_args_i64.size(0),
         "block_postnorm_ln slice out of bounds at group ",
         i,
         "."
      );
      row_offsets.push_back(row_offsets.back() + n);
      out_offsets.push_back(out_offsets.back() + len);
   }

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
   if(x.is_cuda() && relation_args_i64.is_cuda() && w1_stack.is_cuda() && w2_stack.is_cuda()
      && (b1_stack.numel() == 0 || b1_stack.is_cuda()) && (b2_stack.numel() == 0 || b2_stack.is_cuda())
      && (ln_weight_stack.numel() == 0 || ln_weight_stack.is_cuda())
      && (ln_bias_stack.numel() == 0 || ln_bias_stack.is_cuda()) && is_fastpath_dtype(dtype_of(x))) {
      Tensor slot_offsets_t = at::tensor(slot_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor row_offsets_t = at::tensor(row_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor out_offsets_t = at::tensor(out_offsets, relation_args_i64.options().dtype(at::kLong));
      return block_postnorm_ln_cuda(
         x,
         relation_args_i64,
         slot_offsets_t,
         row_offsets_t,
         out_offsets_t,
         row_offsets.back(),
         out_offsets.back(),
         arity,
         w1_stack,
         b1_stack,
         w2_stack,
         b2_stack,
         ln_weight_stack,
         ln_bias_stack,
         ln_eps,
         pointwise_code
      );
   }
#endif

   TORCH_CHECK(
      false,
      "block_postnorm_ln currently supports only CUDA float32/float64 tensors."
   );
}

std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor >
block_postnorm_ln_backward(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const std::vector< int64_t >& slot_offsets,
   const std::vector< int64_t >& row_sizes,
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
   check_rank(grad_rel, 2, "grad_rel");
   check_rank(x, 2, "x");
   check_int32_or_int64_index(relation_args, "relation_args");
   TORCH_CHECK(
      x.device() == relation_args.device() && grad_rel.device() == x.device(),
      "block_postnorm_ln_backward expects grad_rel, x, and relation_args on the same device."
   );
   TORCH_CHECK(
      slot_offsets.size() == row_sizes.size(),
      "block_postnorm_ln_backward expects slot_offsets and row_sizes with equal lengths."
   );
   TORCH_CHECK(arity > 0, "block_postnorm_ln_backward expects arity > 0.");
   check_rank(w1_stack, 3, "w1_stack");
   check_rank(w2_stack, 3, "w2_stack");

   const int64_t groups = static_cast< int64_t >(slot_offsets.size());
   const int64_t emb = x.size(1);
   const int64_t in_dim = emb * arity;
   TORCH_CHECK(
      grad_rel.size(1) == emb,
      "block_postnorm_ln_backward expects grad_rel.shape[1] == emb."
   );
   TORCH_CHECK(
      w1_stack.size(0) == groups && w2_stack.size(0) == groups,
      "block_postnorm_ln_backward weight stacks must match group count."
   );
   TORCH_CHECK(
      w1_stack.size(2) == in_dim && w2_stack.size(1) == in_dim,
      "block_postnorm_ln_backward weight stack dims do not match arity * emb."
   );
   if(ln_weight_stack.numel() > 0) {
      check_rank(ln_weight_stack, 2, "ln_weight_stack");
      TORCH_CHECK(
         ln_weight_stack.size(0) == groups && ln_weight_stack.size(1) == in_dim,
         "block_postnorm_ln_backward ln_weight_stack must have shape [groups, arity * emb]."
      );
   }
   if(ln_bias_stack.numel() > 0) {
      check_rank(ln_bias_stack, 2, "ln_bias_stack");
      TORCH_CHECK(
         ln_bias_stack.size(0) == groups && ln_bias_stack.size(1) == in_dim,
         "block_postnorm_ln_backward ln_bias_stack must have shape [groups, arity * emb]."
      );
   }

   std::vector< int64_t > row_offsets;
   std::vector< int64_t > out_offsets;
   row_offsets.reserve(static_cast< size_t >(groups + 1));
   out_offsets.reserve(static_cast< size_t >(groups + 1));
   row_offsets.push_back(0);
   out_offsets.push_back(0);
   for(int64_t i = 0; i < groups; ++i) {
      const int64_t n = row_sizes[static_cast< size_t >(i)];
      TORCH_CHECK(
         n >= 0,
         "block_postnorm_ln_backward row_sizes must be >= 0 at group ",
         i,
         "."
      );
      const int64_t start = slot_offsets[static_cast< size_t >(i)];
      const int64_t len = n * arity;
      TORCH_CHECK(start >= 0, "block_postnorm_ln_backward expects non-negative slot offsets.");
      TORCH_CHECK(
         start + len <= relation_args.size(0),
         "block_postnorm_ln_backward slice out of bounds at group ",
         i,
         "."
      );
      row_offsets.push_back(row_offsets.back() + n);
      out_offsets.push_back(out_offsets.back() + len);
   }
   TORCH_CHECK(
      grad_rel.size(0) == out_offsets.back(),
      "block_postnorm_ln_backward expects grad_rel rows to match packed slot count."
   );

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
   if(grad_rel.is_cuda() && x.is_cuda() && relation_args.is_cuda() && w1_stack.is_cuda()
      && w2_stack.is_cuda() && (b1_stack.numel() == 0 || b1_stack.is_cuda())
      && (b2_stack.numel() == 0 || b2_stack.is_cuda())
      && (ln_weight_stack.numel() == 0 || ln_weight_stack.is_cuda())
      && (ln_bias_stack.numel() == 0 || ln_bias_stack.is_cuda()) && is_fastpath_dtype(dtype_of(grad_rel))) {
      Tensor relation_args_i64 =
         dtype_of(relation_args) == at::kLong ? relation_args : relation_args.to(at::kLong);
      Tensor slot_offsets_t = at::tensor(slot_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor row_offsets_t = at::tensor(row_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor out_offsets_t = at::tensor(out_offsets, relation_args_i64.options().dtype(at::kLong));
      return block_postnorm_ln_backward_cuda(
         grad_rel,
         x,
         relation_args_i64,
         slot_offsets_t,
         row_offsets_t,
         out_offsets_t,
         row_offsets.back(),
         arity,
         w1_stack,
         b1_stack,
         w2_stack,
         b2_stack,
         ln_weight_stack,
         ln_bias_stack,
         ln_eps,
         pointwise_code
      );
   }
#endif

   TORCH_CHECK(
      false,
      "block_postnorm_ln_backward currently supports only CUDA float32/float64 tensors."
   );
}

std::tuple< Tensor, Tensor > block_prenorm_rms(
   const Tensor& x,
   const Tensor& relation_args,
   const std::vector< int64_t >& slot_offsets,
   const std::vector< int64_t >& row_sizes,
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
   check_rank(x, 2, "x");
   check_int32_or_int64_index(relation_args, "relation_args");
   TORCH_CHECK(
      x.device() == relation_args.device(),
      "block_prenorm_rms expects x and relation_args on the same device."
   );
   TORCH_CHECK(
      slot_offsets.size() == row_sizes.size(),
      "block_prenorm_rms expects slot_offsets and row_sizes with equal lengths."
   );
   TORCH_CHECK(arity > 0, "block_prenorm_rms expects arity > 0.");
   check_rank(w1_stack, 3, "w1_stack");
   check_rank(w2_stack, 3, "w2_stack");

   const int64_t groups = static_cast< int64_t >(slot_offsets.size());
   const int64_t emb = x.size(1);
   const int64_t in_dim = emb * arity;
   TORCH_CHECK(
      w1_stack.size(0) == groups && w2_stack.size(0) == groups,
      "block_prenorm_rms weight stacks must match group count."
   );
   TORCH_CHECK(
      w1_stack.size(2) == in_dim && w2_stack.size(1) == in_dim,
      "block_prenorm_rms weight stack dims do not match arity * emb."
   );
   if(rms_weight_stack.numel() > 0) {
      check_rank(rms_weight_stack, 2, "rms_weight_stack");
      TORCH_CHECK(
         rms_weight_stack.size(0) == groups && rms_weight_stack.size(1) == in_dim,
         "block_prenorm_rms rms_weight_stack must have shape [groups, arity * emb]."
      );
   }

   Tensor relation_args_i64 =
      dtype_of(relation_args) == at::kLong ? relation_args : relation_args.to(at::kLong);
   std::vector< int64_t > row_offsets;
   std::vector< int64_t > out_offsets;
   row_offsets.reserve(static_cast< size_t >(groups + 1));
   out_offsets.reserve(static_cast< size_t >(groups + 1));
   row_offsets.push_back(0);
   out_offsets.push_back(0);
   for(int64_t i = 0; i < groups; ++i) {
      const int64_t n = row_sizes[static_cast< size_t >(i)];
      TORCH_CHECK(
         n >= 0,
         "block_prenorm_rms row_sizes must be >= 0 at group ",
         i,
         "."
      );
      const int64_t start = slot_offsets[static_cast< size_t >(i)];
      const int64_t len = n * arity;
      TORCH_CHECK(start >= 0, "block_prenorm_rms expects non-negative slot offsets.");
      TORCH_CHECK(
         start + len <= relation_args_i64.size(0),
         "block_prenorm_rms slice out of bounds at group ",
         i,
         "."
      );
      row_offsets.push_back(row_offsets.back() + n);
      out_offsets.push_back(out_offsets.back() + len);
   }

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
   if(x.is_cuda() && relation_args_i64.is_cuda() && w1_stack.is_cuda() && w2_stack.is_cuda()
      && (b1_stack.numel() == 0 || b1_stack.is_cuda()) && (b2_stack.numel() == 0 || b2_stack.is_cuda())
      && (rms_weight_stack.numel() == 0 || rms_weight_stack.is_cuda()) && is_fastpath_dtype(dtype_of(x))) {
      Tensor slot_offsets_t = at::tensor(slot_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor row_offsets_t = at::tensor(row_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor out_offsets_t = at::tensor(out_offsets, relation_args_i64.options().dtype(at::kLong));
      return block_prenorm_rms_cuda(
         x,
         relation_args_i64,
         slot_offsets_t,
         row_offsets_t,
         out_offsets_t,
         row_offsets.back(),
         out_offsets.back(),
         arity,
         rms_weight_stack,
         rms_eps,
         w1_stack,
         b1_stack,
         w2_stack,
         b2_stack,
         pointwise_code
      );
   }
#endif

   TORCH_CHECK(
      false,
      "block_prenorm_rms currently supports only CUDA float32/float64 tensors."
   );
}

std::tuple< Tensor, Tensor, Tensor, Tensor, Tensor, Tensor >
block_prenorm_rms_backward(
   const Tensor& grad_rel,
   const Tensor& x,
   const Tensor& relation_args,
   const std::vector< int64_t >& slot_offsets,
   const std::vector< int64_t >& row_sizes,
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
   check_rank(grad_rel, 2, "grad_rel");
   check_rank(x, 2, "x");
   check_int32_or_int64_index(relation_args, "relation_args");
   TORCH_CHECK(
      x.device() == relation_args.device() && grad_rel.device() == x.device(),
      "block_prenorm_rms_backward expects grad_rel, x, and relation_args on the same device."
   );
   TORCH_CHECK(
      slot_offsets.size() == row_sizes.size(),
      "block_prenorm_rms_backward expects slot_offsets and row_sizes with equal lengths."
   );
   TORCH_CHECK(arity > 0, "block_prenorm_rms_backward expects arity > 0.");
   check_rank(w1_stack, 3, "w1_stack");
   check_rank(w2_stack, 3, "w2_stack");

   const int64_t groups = static_cast< int64_t >(slot_offsets.size());
   const int64_t emb = x.size(1);
   const int64_t in_dim = emb * arity;
   TORCH_CHECK(
      grad_rel.size(1) == emb,
      "block_prenorm_rms_backward expects grad_rel.shape[1] == emb."
   );
   TORCH_CHECK(
      w1_stack.size(0) == groups && w2_stack.size(0) == groups,
      "block_prenorm_rms_backward weight stacks must match group count."
   );
   TORCH_CHECK(
      w1_stack.size(2) == in_dim && w2_stack.size(1) == in_dim,
      "block_prenorm_rms_backward weight stack dims do not match arity * emb."
   );
   if(rms_weight_stack.numel() > 0) {
      check_rank(rms_weight_stack, 2, "rms_weight_stack");
      TORCH_CHECK(
         rms_weight_stack.size(0) == groups && rms_weight_stack.size(1) == in_dim,
         "block_prenorm_rms_backward rms_weight_stack must have shape [groups, arity * emb]."
      );
   }

   std::vector< int64_t > row_offsets;
   std::vector< int64_t > out_offsets;
   row_offsets.reserve(static_cast< size_t >(groups + 1));
   out_offsets.reserve(static_cast< size_t >(groups + 1));
   row_offsets.push_back(0);
   out_offsets.push_back(0);
   for(int64_t i = 0; i < groups; ++i) {
      const int64_t n = row_sizes[static_cast< size_t >(i)];
      TORCH_CHECK(
         n >= 0,
         "block_prenorm_rms_backward row_sizes must be >= 0 at group ",
         i,
         "."
      );
      const int64_t start = slot_offsets[static_cast< size_t >(i)];
      const int64_t len = n * arity;
      TORCH_CHECK(start >= 0, "block_prenorm_rms_backward expects non-negative slot offsets.");
      TORCH_CHECK(
         start + len <= relation_args.size(0),
         "block_prenorm_rms_backward slice out of bounds at group ",
         i,
         "."
      );
      row_offsets.push_back(row_offsets.back() + n);
      out_offsets.push_back(out_offsets.back() + len);
   }
   TORCH_CHECK(
      grad_rel.size(0) == out_offsets.back(),
      "block_prenorm_rms_backward expects grad_rel rows to match packed slot count."
   );

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
   if(grad_rel.is_cuda() && x.is_cuda() && relation_args.is_cuda() && w1_stack.is_cuda()
      && w2_stack.is_cuda() && (b1_stack.numel() == 0 || b1_stack.is_cuda())
      && (b2_stack.numel() == 0 || b2_stack.is_cuda())
      && (rms_weight_stack.numel() == 0 || rms_weight_stack.is_cuda()) && is_fastpath_dtype(dtype_of(grad_rel))) {
      Tensor relation_args_i64 =
         dtype_of(relation_args) == at::kLong ? relation_args : relation_args.to(at::kLong);
      Tensor slot_offsets_t = at::tensor(slot_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor row_offsets_t = at::tensor(row_offsets, relation_args_i64.options().dtype(at::kLong));
      Tensor out_offsets_t = at::tensor(out_offsets, relation_args_i64.options().dtype(at::kLong));
      return block_prenorm_rms_backward_cuda(
         grad_rel,
         x,
         relation_args_i64,
         slot_offsets_t,
         row_offsets_t,
         out_offsets_t,
         row_offsets.back(),
         arity,
         rms_weight_stack,
         rms_eps,
         w1_stack,
         b1_stack,
         w2_stack,
         b2_stack,
         pointwise_code
      );
   }
#endif

   TORCH_CHECK(
      false,
      "block_prenorm_rms_backward currently supports only CUDA float32/float64 tensors."
   );
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
      "block_pointwise(Tensor x, Tensor relation_args, int[] slot_offsets, int[] row_sizes, int arity, Tensor w1_stack, Tensor b1_stack, Tensor w2_stack, Tensor b2_stack, int pointwise_code) -> (Tensor, Tensor)"
   );
   m.def(
      "block_pointwise_backward(Tensor grad_rel, Tensor x, Tensor relation_args, int[] slot_offsets, int[] row_sizes, int arity, Tensor w1_stack, Tensor b1_stack, Tensor w2_stack, Tensor b2_stack, int pointwise_code) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
   );
   m.def(
      "block_postnorm_ln(Tensor x, Tensor relation_args, int[] slot_offsets, int[] row_sizes, int arity, Tensor w1_stack, Tensor b1_stack, Tensor w2_stack, Tensor b2_stack, Tensor ln_weight_stack, Tensor ln_bias_stack, float ln_eps, int pointwise_code) -> (Tensor, Tensor)"
   );
   m.def(
      "block_postnorm_ln_backward(Tensor grad_rel, Tensor x, Tensor relation_args, int[] slot_offsets, int[] row_sizes, int arity, Tensor w1_stack, Tensor b1_stack, Tensor w2_stack, Tensor b2_stack, Tensor ln_weight_stack, Tensor ln_bias_stack, float ln_eps, int pointwise_code) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
   );
   m.def(
      "block_prenorm_rms(Tensor x, Tensor relation_args, int[] slot_offsets, int[] row_sizes, int arity, Tensor rms_weight_stack, float rms_eps, Tensor w1_stack, Tensor b1_stack, Tensor w2_stack, Tensor b2_stack, int pointwise_code) -> (Tensor, Tensor)"
   );
   m.def(
      "block_prenorm_rms_backward(Tensor grad_rel, Tensor x, Tensor relation_args, int[] slot_offsets, int[] row_sizes, int arity, Tensor rms_weight_stack, float rms_eps, Tensor w1_stack, Tensor b1_stack, Tensor w2_stack, Tensor b2_stack, int pointwise_code) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
   );
   m.def(
      "program_silu_pair(Tensor x, Tensor relation_args, int[] slot_offsets, int[] row_sizes, int arity, Tensor w10_stack, Tensor b10_stack, Tensor w20_stack, Tensor b20_stack, Tensor w11_stack, Tensor b11_stack, Tensor w21_stack, Tensor b21_stack) -> (Tensor, Tensor)"
   );
   m.def(
      "program_silu_pair_backward(Tensor grad_rel, Tensor x, Tensor relation_args, int[] slot_offsets, int[] row_sizes, int arity, Tensor w10_stack, Tensor b10_stack, Tensor w20_stack, Tensor b20_stack, Tensor w11_stack, Tensor b11_stack, Tensor w21_stack, Tensor b21_stack) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
   );
   m.def(
      "program_silu_postnorm(Tensor x, Tensor relation_args, int[] slot_offsets, int[] row_sizes, int arity, Tensor w10_stack, Tensor b10_stack, Tensor w20_stack, Tensor b20_stack, Tensor w11_stack, Tensor b11_stack, Tensor w21_stack, Tensor b21_stack, Tensor ln_weight_stack, Tensor ln_bias_stack, float ln_eps) -> (Tensor, Tensor)"
   );
   m.def(
      "program_silu_postnorm_backward(Tensor grad_rel, Tensor x, Tensor relation_args, int[] slot_offsets, int[] row_sizes, int arity, Tensor w10_stack, Tensor b10_stack, Tensor w20_stack, Tensor b20_stack, Tensor w11_stack, Tensor b11_stack, Tensor w21_stack, Tensor b21_stack, Tensor ln_weight_stack, Tensor ln_bias_stack, float ln_eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
   );
   m.def(
      "program_rmsnorm_silu(Tensor x, Tensor relation_args, int[] slot_offsets, int[] row_sizes, int arity, Tensor rms_weight_stack, float rms_eps, Tensor w10_stack, Tensor b10_stack, Tensor w20_stack, Tensor b20_stack, Tensor w11_stack, Tensor b11_stack, Tensor w21_stack, Tensor b21_stack) -> (Tensor, Tensor)"
   );
   m.def(
      "program_rmsnorm_silu_backward(Tensor grad_rel, Tensor x, Tensor relation_args, int[] slot_offsets, int[] row_sizes, int arity, Tensor rms_weight_stack, float rms_eps, Tensor w10_stack, Tensor b10_stack, Tensor w20_stack, Tensor b20_stack, Tensor w11_stack, Tensor b11_stack, Tensor w21_stack, Tensor b21_stack) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
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
   m.impl(
      "block_pointwise", relm::mp::block_pointwise
   );
   m.impl(
      "block_pointwise_backward",
      relm::mp::block_pointwise_backward
   );
   m.impl(
      "block_postnorm_ln",
      relm::mp::block_postnorm_ln
   );
   m.impl(
      "block_postnorm_ln_backward",
      relm::mp::block_postnorm_ln_backward
   );
   m.impl(
      "block_prenorm_rms",
      relm::mp::block_prenorm_rms
   );
   m.impl(
      "block_prenorm_rms_backward",
      relm::mp::block_prenorm_rms_backward
   );
   m.impl(
      "program_silu_pair",
      relm::mp::program_silu_pair
   );
   m.impl(
      "program_silu_pair_backward",
      relm::mp::program_silu_pair_backward
   );
   m.impl(
      "program_silu_postnorm",
      relm::mp::program_silu_postnorm
   );
   m.impl(
      "program_silu_postnorm_backward",
      relm::mp::program_silu_postnorm_backward
   );
   m.impl(
      "program_rmsnorm_silu",
      relm::mp::program_rmsnorm_silu
   );
   m.impl(
      "program_rmsnorm_silu_backward",
      relm::mp::program_rmsnorm_silu_backward
   );
   m.impl("build_info", relm::mp::build_info);
}
