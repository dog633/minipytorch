#include "Tensor.hpp"

// 内部公共工具（声明在 Tensor.hpp，实现按模块拆到这里）

int64_t Tensor::check_dim(int64_t dim, int64_t ndim, const char* op)
{
    if (dim < -ndim || dim >= ndim)
    {
        throw std::runtime_error(std::string(op) + ": dim out of range");
    }
    return dim < 0 ? dim + ndim : dim;
}

void Tensor::next_index(std::vector<int64_t>& idx, const std::vector<int64_t>& shape)
{
    for (size_t d = shape.size(); d-- > 0; )
    {
        if (++idx[d] < shape[d])
        {
            return;
        }
        idx[d] = 0;
    }
}

Tensor::OuterInner Tensor::outer_inner(int64_t dim) const
{
    const int64_t ndim = static_cast<int64_t>(shape_.size());
    OuterInner r{1, 1};
    for (int64_t i = 0; i < dim; ++i)
    {
        r.outer *= shape_[i];
    }
    for (int64_t i = dim + 1; i < ndim; ++i)
    {
        r.inner *= shape_[i];
    }
    return r;
}

Tensor Tensor::scalar(float v)
{
    return Tensor(std::vector<float>{v}, std::vector<int64_t>{});
}

Tensor Tensor::log_clamped(const Tensor& t)
{
    Tensor out = unary_elementwise(t, [](float x)
    {
        const float v = std::log(x);
        return v > -100.0f ? v : -100.0f;
    });
    // 反向：未触发截断的区域导数是 1/x，被截断的区域（log x ≤ -100）导数为 0。
    // 必须在一次逐元素运算里同时判断 + 除法，否则 x = 0 时会算出 inf * 0 = nan。
    attach_grad(out, {t}, [](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, binary_element(c.g, c.x(0), [](float gg, float xx)
        {
            return std::log(xx) > -100.0f ? gg / xx : 0.0f;
        }));
    });
    return out;
}

// 沿 dim 取 [start, start+count) 这一段（concat 的反向要沿拼接维切回去）
Tensor Tensor::slice_dim(const Tensor& src, int64_t dim, int64_t start, int64_t count)
{
    const Tensor::OuterInner seg = src.outer_inner(dim);

    std::vector<int64_t> out_shape = src.shape_;
    out_shape[dim] = count;
    std::vector<float> out(static_cast<size_t>(seg.outer * count * seg.inner));

    for (int64_t o = 0; o < seg.outer; ++o)
    {
        // 每段 count*inner 个 float 在内存里连续，整段拷贝
        std::copy(src.data_.begin() + (o * src.shape_[dim] + start) * seg.inner,
                  src.data_.begin() + (o * src.shape_[dim] + start + count) * seg.inner,
                  out.begin() + o * count * seg.inner);
    }
    return Tensor(std::move(out), out_shape);
}

int64_t Tensor::numel(const std::vector<int64_t>& shape)
{
    int64_t n = 1;
    for (auto d : shape)
    {
        if (d<0)
        {
            throw std::runtime_error("Tensor: negative dim");
        }
        n *= d;
    }
    return n;
}
