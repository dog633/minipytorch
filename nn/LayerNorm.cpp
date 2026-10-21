#include "LayerNorm.hpp"

#include <stdexcept>
#include <utility>

// LayerNorm（声明见 LayerNorm.hpp）

// 只标准化最后一维：LayerNorm(128)
LayerNorm::LayerNorm(int64_t normalized_size, float eps, bool elementwise_affine)
    : LayerNorm(std::vector<int64_t>{normalized_size}, eps, elementwise_affine)
{
}

// 列表写法：LayerNorm({3, 4})
LayerNorm::LayerNorm(std::initializer_list<int64_t> normalized_shape, float eps, bool elementwise_affine)
    : LayerNorm(std::vector<int64_t>(normalized_shape), eps, elementwise_affine)
{
}

// 主构造
LayerNorm::LayerNorm(const std::vector<int64_t>& normalized_shape, float eps, bool elementwise_affine)
    : gamma(Tensor::ones(normalized_shape)),
      beta(Tensor::zeros(normalized_shape)),
      normalized_shape_(normalized_shape),
      eps_(eps),
      affine_(elementwise_affine)
{
    if (normalized_shape_.empty())
    {
        throw std::runtime_error("LayerNorm: normalized_shape must not be empty");
    }
    for (int64_t d : normalized_shape_)
    {
        if (d <= 0)
        {
            throw std::runtime_error("LayerNorm: normalized_shape must be positive");
        }
    }
    if (eps_ <= 0.0f)
    {
        throw std::runtime_error("LayerNorm: eps must be positive");
    }
    gamma.requires_grad = true;
    beta.requires_grad = true;
}

Tensor LayerNorm::forward(const Tensor& x)
{
    const int64_t k = static_cast<int64_t>(normalized_shape_.size());
    const int64_t ndim = static_cast<int64_t>(x.shape().size());
    if (ndim < k)
    {
        throw std::runtime_error("LayerNorm: input has fewer dims than normalized_shape");
    }
    for (int64_t i = 0; i < k; ++i)
    {
        if (x.shape()[static_cast<size_t>(ndim - k + i)] != normalized_shape_[static_cast<size_t>(i)])
        {
            throw std::runtime_error("LayerNorm: normalized_shape mismatch");
        }
    }

    // 依次对最后 k 维求均值（keepdim=true）：先算子均值、再算方差
    Tensor mu = x;
    for (int64_t i = 0; i < k; ++i)
    {
        mu = mu.mean(-1, true);
    }

    const Tensor centered = x.sub(mu);

    Tensor var = centered.mul(centered);
    for (int64_t i = 0; i < k; ++i)
    {
        var = var.mean(-1, true);
    }

    const Tensor norm = centered.div(var.add(Tensor::scalar(eps_)).sqrt());
    if (!affine_)
    {
        return norm;
    }
    // γ/β 的形状就是 normalized_shape，靠广播作用到前面的批次维上
    return norm.mul(gamma).add(beta);
}

std::vector<Tensor*> LayerNorm::parameters()
{
    if (!affine_)
    {
        return {};
    }
    return {&gamma, &beta};
}
