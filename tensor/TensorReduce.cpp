#include "Tensor.hpp"

// 一元 neg/abs 与各种归约（声明在 Tensor.hpp，实现按模块拆到这里）
//
// 反向分两类：
//   - 逐元素（neg / abs）：dx = g ⊙ f'(x)，用 c.grad(0, ...)
//   - 归约（sum / prod / max / min / mean）：梯度形状比输入小，
//     先用 c.grad_expand(0, ...) 把"被归约掉的那一维"补回来再广播

Tensor Tensor::neg() const
{
    Tensor out = unary_elementwise(*this, [](float x) { return -x; });
    // d(-x)/dx = -1
    attach_grad(out, {*this}, [](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, c.g.neg());
    });
    return out;
}

Tensor Tensor::abs() const
{
    Tensor out = unary_elementwise(*this, [](float x) { return std::fabs(x); });
    // d|x|/dx = sign(x)（x = 0 处取 0，与 PyTorch 一致）
    attach_grad(out, {*this}, [](const tensor_detail::BackwardCtx& c)
    {
        const Tensor sign = unary_elementwise(c.x(0), [](float v)
        {
            return v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f);
        });
        c.grad(0, c.g.mul(sign));
    });
    return out;
}

Tensor Tensor::sum() const
{
    Tensor out = reduce_all_identity([](float a, float b) { return a + b; }, 0.0f);
    // 反向：标量梯度原样广播回输入形状
    attach_grad(out, {*this}, [](const tensor_detail::BackwardCtx& c)
    {
        c.grad_expand(0, c.g);
    });
    return out;
}

Tensor Tensor::prod() const
{
    Tensor out = reduce_all_identity([](float a, float b) { return a * b; }, 1.0f);
    // 反向：dx_i = g · P / x_i（P 是整体连乘）。注意 x_i = 0 时会得到 inf/nan
    attach_grad(out, {*this}, [](const tensor_detail::BackwardCtx& c)
    {
        const Tensor x = c.x(0);
        const Tensor P = x.prod();              // 重算连乘（keepdim 不需要，后面靠广播）
        c.grad_expand(0, c.g.mul(P).div(x));
    });
    return out;
}

Tensor Tensor::max() const
{
    Tensor out = reduce_all_nonempty([](float a, float b) { return std::max(a, b); }, "max");
    // 反向：梯度只给到最大元素（并列时每个并列项都拿到完整梯度，PyTorch 只给其中一个）
    attach_grad(out, {*this}, [](const tensor_detail::BackwardCtx& c)
    {
        const Tensor x = c.x(0);
        const Tensor mask = binary_element(x, x.max(),
                                           [](float a, float b) { return a == b ? 1.0f : 0.0f; });
        c.grad_expand(0, c.g.mul(mask));
    });
    return out;
}

Tensor Tensor::min() const
{
    Tensor out = reduce_all_nonempty([](float a, float b) { return std::min(a, b); }, "min");
    attach_grad(out, {*this}, [](const tensor_detail::BackwardCtx& c)
    {
        const Tensor x = c.x(0);
        const Tensor mask = binary_element(x, x.min(),
                                           [](float a, float b) { return a == b ? 1.0f : 0.0f; });
        c.grad_expand(0, c.g.mul(mask));
    });
    return out;
}

Tensor Tensor::mean() const
{
    const float s = sum().data_[0];
    Tensor out = scalar(s / static_cast<float>(data_.size()));
    // 反向：dx = g / n，把 1/n 折进这个节点的反向里
    const float inv_n = 1.0f / static_cast<float>(data_.size());
    attach_grad(out, {*this}, [inv_n](const tensor_detail::BackwardCtx& c)
    {
        c.grad_expand(0, c.g.mul(scalar(inv_n)));
    });
    return out;
}

Tensor Tensor::sum(int64_t dim, bool keepdim) const
{
    Tensor out = reduce_dim(dim, [](float a, float b) { return a + b; }, keepdim);
    const int64_t d = check_dim(dim, static_cast<int64_t>(shape_.size()), "sum");
    // 反向：dx = g 沿该维广播回去（keepdim = false 时先补回长度为 1 的那一维）
    attach_grad(out, {*this}, [d, keepdim](const tensor_detail::BackwardCtx& c)
    {
        Tensor gg = c.g;
        if (!keepdim)
        {
            gg = gg.unsqueeze(d);
        }
        c.grad_expand(0, gg);
    });
    return out;
}

Tensor Tensor::prod(int64_t dim, bool keepdim) const
{
    Tensor out = reduce_dim(dim, [](float a, float b) { return a * b; }, keepdim);
    const int64_t d = check_dim(dim, static_cast<int64_t>(shape_.size()), "prod");
    // 反向：dx_i = g · P / x_i，P 是该维的连乘（keepdim 保留长度 1，便于广播）。
    // keepdim = false 时必须先把 g 补回长度 1 的那一维，否则 [2] 和 [2,1] 会广播成 [2,2]。
    // 注意 x_i = 0 时会得到 inf/nan
    attach_grad(out, {*this}, [d, keepdim](const tensor_detail::BackwardCtx& c)
    {
        const Tensor x = c.x(0);
        Tensor gg = c.g;
        if (!keepdim)
        {
            gg = gg.unsqueeze(d);
        }
        c.grad_expand(0, gg.mul(x.prod(d, true)).div(x));
    });
    return out;
}

Tensor Tensor::max(int64_t dim, bool keepdim) const
{
    Tensor out = reduce_dim(dim, [](float a, float b) { return std::max(a, b); }, keepdim);
    const int64_t d = check_dim(dim, static_cast<int64_t>(shape_.size()), "max");
    // 反向：该维最大值的位置分到梯度（并列时并列项都会拿到完整梯度，PyTorch 只给其中一个）
    attach_grad(out, {*this}, [d, keepdim](const tensor_detail::BackwardCtx& c)
    {
        const Tensor x = c.x(0);
        Tensor gg = c.g;
        if (!keepdim)
        {
            gg = gg.unsqueeze(d);
        }
        const Tensor mask = binary_element(x, x.max(d, true),
                                           [](float a, float b) { return a == b ? 1.0f : 0.0f; });
        c.grad_expand(0, gg.mul(mask));
    });
    return out;
}

Tensor Tensor::min(int64_t dim, bool keepdim) const
{
    Tensor out = reduce_dim(dim, [](float a, float b) { return std::min(a, b); }, keepdim);
    const int64_t d = check_dim(dim, static_cast<int64_t>(shape_.size()), "min");
    attach_grad(out, {*this}, [d, keepdim](const tensor_detail::BackwardCtx& c)
    {
        const Tensor x = c.x(0);
        Tensor gg = c.g;
        if (!keepdim)
        {
            gg = gg.unsqueeze(d);
        }
        const Tensor mask = binary_element(x, x.min(d, true),
                                           [](float a, float b) { return a == b ? 1.0f : 0.0f; });
        c.grad_expand(0, gg.mul(mask));
    });
    return out;
}

// ---------------- 指标类 ----------------

Tensor Tensor::arg_reduce_impl(const Tensor& t, int64_t dim, bool keepdim, bool want_max, const char* op)
{
    const int64_t ndim = static_cast<int64_t>(t.shape_.size());
    if (ndim == 0)
    {
        throw std::runtime_error(std::string(op) + ": scalar has no dim");
    }
    const int64_t d = check_dim(dim, ndim, op);
    const int64_t n = t.shape_[d];
    if (n == 0)
    {
        throw std::runtime_error(std::string(op) + ": empty dim");
    }
    const OuterInner seg = t.outer_inner(d);

    std::vector<int64_t> out_shape;
    out_shape.reserve(static_cast<size_t>(keepdim ? ndim : ndim - 1));
    for (int64_t i = 0; i < ndim; ++i)
    {
        if (i == d)
        {
            if (keepdim)
            {
                out_shape.push_back(1);
            }
        }
        else
        {
            out_shape.push_back(t.shape_[i]);
        }
    }

    // 和 reduce_dim 一样是"每段取一个"，只不过放的是**下标**（本库只有 float，就用 float 存整数）
    std::vector<float> out(static_cast<size_t>(seg.outer * seg.inner));
    for (int64_t o = 0; o < seg.outer; ++o)
    {
        for (int64_t in = 0; in < seg.inner; ++in)
        {
            const float* base = t.data_.data() + o * n * seg.inner + in;
            int64_t best = 0;
            for (int64_t k = 1; k < n; ++k)
            {
                const float v = base[k * seg.inner];
                // 严格比较 -> 并列时取第一个（行优先顺序）
                if (want_max ? (v > base[best * seg.inner]) : (v < base[best * seg.inner]))
                {
                    best = k;
                }
            }
            out[static_cast<size_t>(o * seg.inner + in)] = static_cast<float>(best);
        }
    }
    return Tensor(std::move(out), out_shape);
}

Tensor Tensor::argmax(int64_t dim, bool keepdim) const
{
    return arg_reduce_impl(*this, dim, keepdim, true, "argmax");
}

Tensor Tensor::argmin(int64_t dim, bool keepdim) const
{
    return arg_reduce_impl(*this, dim, keepdim, false, "argmin");
}

Tensor Tensor::argmax() const
{
    if (data_.empty())
    {
        throw std::runtime_error("argmax: empty tensor");
    }
    size_t best = 0;
    for (size_t i = 1; i < data_.size(); ++i)
    {
        if (data_[i] > data_[best])
        {
            best = i;
        }
    }
    return scalar(static_cast<float>(best));
}

Tensor Tensor::argmin() const
{
    if (data_.empty())
    {
        throw std::runtime_error("argmin: empty tensor");
    }
    size_t best = 0;
    for (size_t i = 1; i < data_.size(); ++i)
    {
        if (data_[i] < data_[best])
        {
            best = i;
        }
    }
    return scalar(static_cast<float>(best));
}

Tensor Tensor::var() const
{
    return var(VarCorrection::unbiased);
}

Tensor Tensor::std() const
{
    return var(VarCorrection::unbiased).sqrt();
}

Tensor Tensor::var(VarCorrection correction) const
{
    const Tensor mu = mean();
    const Tensor centered = sub(mu);
    Tensor v = centered.mul(centered).mean();     // 有偏方差 = mean((x-μ)²)

    if (correction == VarCorrection::unbiased)
    {
        if (data_.size() < 2)
        {
            throw std::runtime_error("var: unbiased variance needs at least 2 elements");
        }
        const float n = static_cast<float>(data_.size());
        v = v.mul(scalar(n / (n - 1.0f)));        // n/(n-1) 修正
    }
    return v;
}

Tensor Tensor::std(VarCorrection correction) const
{
    return var(correction).sqrt();
}

Tensor Tensor::var(int64_t dim, bool keepdim, VarCorrection correction) const
{
    const int64_t d = check_dim(dim, static_cast<int64_t>(shape_.size()), "var");

    const Tensor mu = mean(d, true);                  // keepdim 便于广播
    const Tensor centered = sub(mu);
    Tensor v = centered.mul(centered).mean(d, true);  // 有偏方差

    if (correction == VarCorrection::unbiased)
    {
        if (shape_[d] < 2)
        {
            throw std::runtime_error("var: unbiased variance needs at least 2 elements along dim");
        }
        const float n = static_cast<float>(shape_[d]);
        v = v.mul(scalar(n / (n - 1.0f)));
    }
    if (!keepdim)
    {
        v = v.squeeze(d);
    }
    return v;
}

Tensor Tensor::std(int64_t dim, bool keepdim, VarCorrection correction) const
{
    return var(dim, keepdim, correction).sqrt();
}

Tensor Tensor::mean(int64_t dim, bool keepdim) const
{
    const int64_t d = check_dim(dim, static_cast<int64_t>(shape_.size()), "mean");
    Tensor s = sum(d, keepdim);

    std::vector<float> out = s.data_;
    const float n = static_cast<float>(shape_[d]);
    for (float& v : out)
    {
        v /= n;
    }
    Tensor res(std::move(out), s.shape_);
    // 反向：dx = (g / n) 沿该维广播回输入形状（同样把 1/n 折进来）
    const float inv_n = 1.0f / n;
    attach_grad(res, {*this}, [d, keepdim, inv_n](const tensor_detail::BackwardCtx& c)
    {
        Tensor gg = c.g.mul(scalar(inv_n));
        if (!keepdim)
        {
            gg = gg.unsqueeze(d);
        }
        c.grad_expand(0, gg);
    });
    return res;
}
