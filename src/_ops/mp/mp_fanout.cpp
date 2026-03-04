#include "mp_fanout.hpp"

namespace relm::mp
{
    std::tuple<Tensor, Tensor, Tensor> fanout_pack_multi(
        const std::vector<Tensor>& x_parts,
        const std::vector<Tensor>& src_idx_parts,
        const std::vector<Tensor>& flat_dst_parts
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

        std::vector<Tensor> src_global_parts;
        std::vector<Tensor> x_cat_parts;
        std::vector<Tensor> flat_cat_parts;
        src_global_parts.reserve(x_parts.size());
        x_cat_parts.reserve(x_parts.size());
        flat_cat_parts.reserve(x_parts.size());
        int64_t row_offset = 0;

        for (size_t i = 0; i < x_parts.size(); ++i)
        {
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
#endif

    template <typename scalar_t>
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
        for (int64_t e = 0; e < num_edges; ++e)
        {
            const int64_t s = src_ptr[e];
            const int64_t d = dst_ptr[e];
            check_in_bounds(s, x_rows, "src_global_idx");
            check_in_bounds(d, out_rows, "flat_dst");
            std::memcpy(
                out_ptr + d * emb, x_ptr + s * emb, static_cast<size_t>(emb) * sizeof(scalar_t)
            );
        }
    }

    template <typename scalar_t>
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
        for (int64_t e = 0; e < num_edges; ++e)
        {
            const int64_t s = src_ptr[e];
            const int64_t d = dst_ptr[e];
            check_in_bounds(s, x_rows, "src_global_idx");
            check_in_bounds(d, grad_out_rows, "flat_dst");
            const scalar_t* in_row = grad_out_ptr + d * emb;
            scalar_t* out_row = grad_x_ptr + s * emb;
            for (int64_t k = 0; k < emb; ++k)
            {
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
        if (src_global_idx.numel() == 0 || out_rows == 0)
        {
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
        if (flat_dst.numel() == 0 || x_rows == 0)
        {
            return grad_x;
        }
        Tensor gathered = grad_out.index_select(0, flat_dst);
        grad_x.index_add_(0, src_global_idx, gathered);
        return grad_x;
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
        if (src_global_idx.numel() == 0 || out_rows == 0)
        {
            return out;
        }

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
        if (x_cat.is_cuda() && src_global_idx.is_cuda() && flat_dst.is_cuda()
            && is_fastpath_dtype(dtype_of(x_cat)))
        {
            return fanout_scatter_cuda(x_cat, src_global_idx, flat_dst, out_rows);
        }
#endif

        if (x_cat.is_cpu() && src_global_idx.is_cpu() && flat_dst.is_cpu()
            && is_fastpath_dtype(dtype_of(x_cat)))
        {
            Tensor x_work = ensure_contiguous(x_cat);
            Tensor src_work = ensure_contiguous(src_global_idx);
            Tensor dst_work = ensure_contiguous(flat_dst);
            const int64_t num_edges = src_work.size(0);

            if (dtype_of(x_work) == at::kFloat)
            {
                fanout_scatter_cpu_kernel<float>(
                    x_work.data_ptr<float>(),
                    src_work.data_ptr<int64_t>(),
                    dst_work.data_ptr<int64_t>(),
                    num_edges,
                    x_work.size(0),
                    out_rows,
                    emb,
                    out.data_ptr<float>()
                );
                return out;
            }
            fanout_scatter_cpu_kernel<double>(
                x_work.data_ptr<double>(),
                src_work.data_ptr<int64_t>(),
                dst_work.data_ptr<int64_t>(),
                num_edges,
                x_work.size(0),
                out_rows,
                emb,
                out.data_ptr<double>()
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
        if (src_global_idx.numel() == 0 || x_rows == 0)
        {
            return grad_x;
        }

#if defined(RELM_MP_HAS_CUDA) && RELM_MP_HAS_CUDA
        if (grad_out.is_cuda() && src_global_idx.is_cuda() && flat_dst.is_cuda()
            && is_fastpath_dtype(dtype_of(grad_out)))
        {
            return fanout_scatter_backward_cuda(grad_out, src_global_idx, flat_dst, x_rows);
        }
#endif

        if (grad_out.is_cpu() && src_global_idx.is_cpu() && flat_dst.is_cpu()
            && is_fastpath_dtype(dtype_of(grad_out)))
        {
            Tensor grad_out_work = ensure_contiguous(grad_out);
            Tensor src_work = ensure_contiguous(src_global_idx);
            Tensor dst_work = ensure_contiguous(flat_dst);
            const int64_t num_edges = src_work.size(0);

            if (dtype_of(grad_out_work) == at::kFloat)
            {
                fanout_scatter_backward_cpu_kernel<float>(
                    grad_out_work.data_ptr<float>(),
                    src_work.data_ptr<int64_t>(),
                    dst_work.data_ptr<int64_t>(),
                    num_edges,
                    x_rows,
                    grad_out_work.size(0),
                    emb,
                    grad_x.data_ptr<float>()
                );
                return grad_x;
            }
            fanout_scatter_backward_cpu_kernel<double>(
                grad_out_work.data_ptr<double>(),
                src_work.data_ptr<int64_t>(),
                dst_work.data_ptr<int64_t>(),
                num_edges,
                x_rows,
                grad_out_work.size(0),
                emb,
                grad_x.data_ptr<double>()
            );
            return grad_x;
        }

        return fanout_scatter_backward_fallback(grad_out, src_global_idx, flat_dst, x_rows);
    }
}
