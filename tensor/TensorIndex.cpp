#include "Tensor.hpp"

// 按下标取（gather）与它的反向 scatter-add（声明在 Tensor.hpp）
//
// 和 im2col 一样，这是个"按索引搬数据"的操作，用现有算子拼不出来
// （拼出来的中间张量是构造函数造的叶子，梯度会断），所以它自带反向。
//
// 两个函数共用同一套 outer/inner 分解：
//   index_select ：[..., n, ...]  ->  [..., m, ...]        （按 indices 读）
//   index_add_grad：[..., m, ...] ->  [..., n, ...] +=     （按 indices 累加写回）

namespace
{
    // 把 shape 在 dim 上拆成 (outer, n, inner) 三段，方便按"整块"搬运
    void outer_inner_of(const std::vector<int64_t>& shape, int64_t dim,
                        int64_t& outer, int64_t& n, int64_t& inner)
    {
        outer = 1;
        n = shape[static_cast<size_t>(dim)];
        inner = 1;
        for (int64_t i = 0; i < dim; ++i)
        {
            outer *= shape[static_cast<size_t>(i)];
        }
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i)
        {
            inner *= shape[static_cast<size_t>(i)];
        }
    }
}  // namespace

Tensor Tensor::index_select(int64_t dim, const std::vector<int64_t>& indices) const
{
    const int64_t ndim = static_cast<int64_t>(shape_.size());
    if (ndim == 0)
    {
        throw std::runtime_error("index_select: scalar has no dim");
    }
    const int64_t d = check_dim(dim, ndim, "index_select");

    int64_t outer = 0, n = 0, inner = 0;
    outer_inner_of(shape_, d, outer, n, inner);

    const int64_t m = static_cast<int64_t>(indices.size());
    std::vector<int64_t> out_shape = shape_;
    out_shape[static_cast<size_t>(d)] = m;

    for (int64_t idx : indices)
    {
        if (idx < 0 || idx >= n)
        {
            throw std::runtime_error("index_select: index out of range");
        }
    }

    // 每一段 inner 个 float 是连续的，整段拷贝
    std::vector<float> out(static_cast<size_t>(outer * m * inner));
    for (int64_t o = 0; o < outer; ++o)
    {
        for (int64_t k = 0; k < m; ++k)
        {
            const int64_t src = indices[static_cast<size_t>(k)];
            std::copy(data_.begin() + (o * n + src) * inner,
                      data_.begin() + (o * n + src + 1) * inner,
                      out.begin() + (o * m + k) * inner);
        }
    }

    Tensor res(std::move(out), out_shape);
    const std::vector<int64_t> in_shape = shape_;
    attach_grad(res, {*this}, [d, indices, in_shape](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, index_add_grad(c.g, indices, in_shape, d));
    });
    return res;
}

Tensor Tensor::index_add_grad(const Tensor& dout, const std::vector<int64_t>& indices,
                              const std::vector<int64_t>& in_shape, int64_t dim)
{
    int64_t outer = 0, n = 0, inner = 0;
    outer_inner_of(in_shape, dim, outer, n, inner);
    const int64_t m = static_cast<int64_t>(indices.size());

    std::vector<float> dx(static_cast<size_t>(numel(in_shape)), 0.0f);

    for (int64_t o = 0; o < outer; ++o)
    {
        for (int64_t k = 0; k < m; ++k)
        {
            // 关键：+= 而不是 =（同一行被选多次时梯度要累加）
            float* dst = dx.data() + (o * n + indices[static_cast<size_t>(k)]) * inner;
            const float* src = dout.data().data() + (o * m + k) * inner;
            for (int64_t t = 0; t < inner; ++t)
            {
                dst[t] += src[t];
            }
        }
    }
    return Tensor(std::move(dx), in_shape);
}
