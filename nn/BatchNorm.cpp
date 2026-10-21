#include "BatchNorm.hpp"

#include <stdexcept>

// 批归一化（声明见 BatchNorm.hpp）

namespace
{
    // 对"除通道维（第 1 维）以外的所有维"求均值，keepdim 让结果保持 [1, C, 1, ..., 1]。
    // 从最后一维往前消，keepdim 保证维号不变，所以不用重算下标。
    Tensor reduce_other_dims(const Tensor& t, int64_t ndim)
    {
        Tensor r = t;
        for (int64_t d = ndim - 1; d >= 0; --d)
        {
            if (d == 1)
            {
                continue;   // 通道维不参与统计
            }
            r = r.mean(d, true);
        }
        return r;
    }

    // [C] -> [1, C, 1, ..., 1]：只有这个形状才能对 [N, C, ...] 正确广播。
    // 直接拿 [C] 去广播会右对齐到最后一维上，那是错的。
    std::vector<int64_t> channel_broadcast_shape(int64_t ndim, int64_t channels)
    {
        std::vector<int64_t> s(static_cast<size_t>(ndim), 1);
        s[1] = channels;
        return s;
    }
}  // namespace

BatchNorm::BatchNorm(int64_t num_features, float eps, float momentum, bool affine)
    : weight(Tensor::ones({num_features})),
      bias(Tensor::zeros({num_features})),
      running_mean(Tensor::zeros({num_features})),
      running_var(Tensor::ones({num_features})),
      num_features_(num_features),
      eps_(eps),
      momentum_(momentum),
      affine_(affine)
{
    if (num_features_ <= 0)
    {
        throw std::runtime_error("BatchNorm: num_features must be positive");
    }
    if (eps_ <= 0.0f)
    {
        throw std::runtime_error("BatchNorm: eps must be positive");
    }
    if (!(momentum_ >= 0.0f && momentum_ <= 1.0f))
    {
        throw std::runtime_error("BatchNorm: momentum must be in [0, 1]");
    }
    // γ / β 是叶子参数；running_* 保持 requires_grad = false（它们是统计量）
    weight.requires_grad = true;
    bias.requires_grad = true;
}

Tensor BatchNorm::forward(const Tensor& x)
{
    const int64_t ndim = static_cast<int64_t>(x.shape().size());
    if (ndim < 2)
    {
        throw std::runtime_error("BatchNorm: input must be at least 2-D [N, C, ...]");
    }
    if (x.shape()[1] != num_features_)
    {
        throw std::runtime_error("BatchNorm: channel mismatch");
    }

    const int64_t channels = num_features_;
    const int64_t per_channel = static_cast<int64_t>(x.data().size()) / channels;   // N × 空间元素数

    // 拿到 μ / σ² 之后，归一化和仿射都是一样的（Tensor 没有默认构造函数，所以用 lambda 收尾）
    const std::vector<int64_t> bs = channel_broadcast_shape(ndim, channels);
    auto normalize = [&](const Tensor& mu, const Tensor& var) -> Tensor
    {
        Tensor y = x.sub(mu).div(var.add(Tensor::scalar(eps_)).sqrt());
        if (affine_)
        {
            y = y.mul(weight.reshape(bs)).add(bias.reshape(bs));
        }
        return y;
    };

    if (!is_training())
    {
        // 推理模式：用攒下来的滑动统计量
        return normalize(running_mean.reshape(bs), running_var.reshape(bs));
    }

    if (per_channel < 2)
    {
        throw std::runtime_error("BatchNorm: need more than 1 value per channel when training");
    }

    // 训练模式：用当前 batch 的统计量。这两个都**带计算图** —— 它们依赖 x，
    // 反向必须把 ∂μ/∂x、∂σ²/∂x 也算进去（这正是 BatchNorm 反向公式里那几项）
    const Tensor mu = reduce_other_dims(x, ndim);
    const Tensor centered = x.sub(mu);
    const Tensor var = reduce_other_dims(centered.mul(centered), ndim);   // 有偏方差（÷ n），用来归一化

    // 更新滑动统计量：detach 掉（不参与反向），只在训练模式做。
    // 注意 running_var 用的是**无偏**方差（÷ (n-1)），归一化用的是有偏 —— 和 PyTorch 一致
    const float n = static_cast<float>(per_channel);
    const Tensor mu_flat = mu.detach().reshape({channels});
    const Tensor var_unbiased = var.detach().reshape({channels}).mul(Tensor::scalar(n / (n - 1.0f)));
    running_mean = running_mean.mul(Tensor::scalar(1.0f - momentum_))
                               .add(mu_flat.mul(Tensor::scalar(momentum_)));
    running_var = running_var.mul(Tensor::scalar(1.0f - momentum_))
                             .add(var_unbiased.mul(Tensor::scalar(momentum_)));

    return normalize(mu, var);
}

std::vector<Tensor*> BatchNorm::parameters()
{
    if (!affine_)
    {
        return {};
    }
    return {&weight, &bias};
}
