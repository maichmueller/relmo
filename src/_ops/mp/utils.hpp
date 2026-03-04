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

namespace relm::mp
{
    using at::Tensor;

    constexpr int64_t kModeSum = 0;
    constexpr int64_t kModeLogSumExp = 1;

    inline at::ScalarType dtype_of(const Tensor& t)
    {
        return t.scalar_type();
    }

    inline bool is_fastpath_dtype(at::ScalarType dtype)
    {
        return dtype == at::kFloat || dtype == at::kDouble;
    }

    inline void check_rank(const Tensor& t, int64_t expected, const char* name)
    {
        TORCH_CHECK(
           t.dim() == expected, name, " must be rank ", expected, ", got rank ", t.dim(), "."
        );
    }

    inline void check_int64_index(const Tensor& t, const char* name)
    {
        check_rank(t, 1, name);
        TORCH_CHECK(dtype_of(t) == at::kLong, name, " must have dtype torch.int64.");
    }

    inline void check_in_bounds(int64_t idx, int64_t size, const char* name)
    {
        TORCH_CHECK(
           idx >= 0 && idx < size, name, " index out of bounds: ", idx, " not in [0, ", size, ")."
        );
    }

    inline Tensor ensure_contiguous(const Tensor& t)
    {
        return t.is_contiguous() ? t : t.contiguous();
    }

    inline Tensor make_scatter_index(const Tensor& idx, int64_t emb)
    {
        return idx.unsqueeze(1).expand({idx.size(0), emb});
    }
}