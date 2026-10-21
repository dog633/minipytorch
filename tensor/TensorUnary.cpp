#include "Tensor.hpp"

#include <limits>

// 一元算子、激活与归一化（声明在 Tensor.hpp，实现按模块拆到这里）
//
// 一元逐元素算子的反向永远是 g ⊙ f'(x)：用 c.x(0)（detach 过的输入）算出局部导数，
// 乘上 c.g（detach 过的上游梯度），再 c.grad(0, ...) 写回去。
// 注意：这里都是"重算" f'(x) 而不是复用前向输出 —— 复用输出会让
// Tensor → Node → lambda → Tensor 形成 shared_ptr 环，永远不释放。

Tensor Tensor::sign() const
{
    // 阶跃函数，导数恒为 0 —— 不建图就是正确的（梯度到这儿本来就没有）
    return unary_elementwise(*this, [](float x)
    {
        return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
    });
}

Tensor Tensor::clamp(float lo, float hi) const
{
    if (lo > hi)
    {
        throw std::runtime_error("clamp: lo > hi");
    }
    Tensor out = unary_elementwise(*this, [lo, hi](float x)
    {
        return std::min(std::max(x, lo), hi);
    });
    // 反向：落在区间内的元素导数是 1，被夹住的是 0（边界算在区间内，与 PyTorch 一致）
    attach_grad(out, {*this}, [lo, hi](const tensor_detail::BackwardCtx& c)
    {
        const Tensor mask = unary_elementwise(c.x(0), [lo, hi](float x)
        {
            return (x >= lo && x <= hi) ? 1.0f : 0.0f;
        });
        c.grad(0, c.g.mul(mask));
    });
    return out;
}

Tensor Tensor::clamp_min(float lo) const
{
    return clamp(lo, std::numeric_limits<float>::infinity());
}

Tensor Tensor::clamp_max(float hi) const
{
    return clamp(-std::numeric_limits<float>::infinity(), hi);
}

Tensor Tensor::exp() const
{
    Tensor out = unary_elementwise(*this, [](float x) {return std::exp(x); });
    // d(exp)/dx = exp(x)
    attach_grad(out, {*this}, [](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, c.g.mul(c.x(0).exp()));
    });
    return out;
}

Tensor Tensor::log() const
{
    Tensor out = unary_elementwise(*this, [](float x) {return std::log(x); });
    // d(log)/dx = 1 / x
    attach_grad(out, {*this}, [](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, c.g.div(c.x(0)));
    });
    return out;
}

Tensor Tensor::sqrt() const
{
    Tensor out = unary_elementwise(*this, [](float x) {return std::sqrt(x); });
    // d(sqrt)/dx = 1 / (2·sqrt(x))
    attach_grad(out, {*this}, [](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, c.g.div(c.x(0).sqrt().mul(scalar(2.0f))));
    });
    return out;
}

Tensor Tensor::pow(float p) const
{
    Tensor out = unary_elementwise(*this, [p](float x) {return std::pow(x, p); });
    // d(x^p)/dx = p · x^(p-1)
    attach_grad(out, {*this}, [p](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, c.g.mul(c.x(0).pow(p - 1.0f)).mul(scalar(p)));
    });
    return out;
}

Tensor Tensor::sigmoid() const
{
    Tensor out = unary_elementwise(*this, [](float x) {return 1.0f/(1.0f + std::exp(-x)); });
    // dσ/dx = σ(x)·(1 − σ(x))
    attach_grad(out, {*this}, [](const tensor_detail::BackwardCtx& c)
    {
        const Tensor s = c.x(0).sigmoid();
        c.grad(0, c.g.mul(s).mul(scalar(1.0f).sub(s)));
    });
    return out;
}

Tensor Tensor::ReLU() const
{
    Tensor out = unary_elementwise(*this, [](float x) { return x > 0.0f ? x : 0.0f; });
    // d(relu)/dx = (x > 0) ? 1 : 0 —— 用掩码实现（x 恰为 0 时导数是 0，和 PyTorch 一致）
    attach_grad(out, {*this}, [](const tensor_detail::BackwardCtx& c)
    {
        const Tensor mask = unary_elementwise(c.x(0),
                                             [](float v) { return v > 0.0f ? 1.0f : 0.0f; });
        c.grad(0, c.g.mul(mask));
    });
    return out;
}

Tensor Tensor::tanh() const
{
    Tensor out = unary_elementwise(*this, [](float x) {return std::tanh(x); });
    // d(tanh)/dx = 1 − tanh²(x)
    attach_grad(out, {*this}, [](const tensor_detail::BackwardCtx& c)
    {
        const Tensor t = c.x(0).tanh();
        c.grad(0, c.g.mul(scalar(1.0f).sub(t.mul(t))));
    });
    return out;
}

Tensor Tensor::softmax(int64_t dim) const
{
    const int64_t ndim = static_cast<int64_t>(shape_.size());
    if (ndim == 0)
    {
        throw std::runtime_error("softmax: scalar has no dim");
    }
    dim = check_dim(dim, ndim, "softmax");

    const Tensor m = max(dim, true);
    const Tensor e = (*this - m).exp();
    const Tensor s = e.sum(dim, true);
    Tensor out = e / s;

    // 记 y = softmax(x)：dx = y ⊙ (g − Σ_dim(g ⊙ y))
    attach_grad(out, {*this}, [dim](const tensor_detail::BackwardCtx& c)
    {
        const Tensor y = c.x(0).softmax(dim);
        const Tensor inner = c.g.mul(y).sum(dim, true);   // keepdim 才能广播回去
        c.grad(0, y.mul(c.g.sub(inner)));
    });
    return out;
}

Tensor Tensor::log_softmax(int64_t dim) const
{
    const int64_t ndim = static_cast<int64_t>(shape_.size());
    if (ndim == 0)
    {
        throw std::runtime_error("log_softmax: scalar has no dim");
    }
    dim = check_dim(dim, ndim, "log_softmax");

    const Tensor d = *this - max(dim, true);
    Tensor out = d - d.exp().sum(dim, true).log();

    // 记 y = log_softmax(x)：dx = g − softmax(x) · Σ_dim(g)
    attach_grad(out, {*this}, [dim](const tensor_detail::BackwardCtx& c)
    {
        const Tensor s = c.x(0).softmax(dim);
        c.grad(0, c.g.sub(s.mul(c.g.sum(dim, true))));
    });
    return out;
}
