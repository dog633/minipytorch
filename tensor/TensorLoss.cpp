#include "Tensor.hpp"

// 损失函数（声明在 Tensor.hpp，实现按模块拆到这里）

Tensor Tensor::mse(const Tensor& other) const
{
    if (shape_ != other.shape_)
    {
        throw std::runtime_error("MSE: shape mismatch");
    }
    Tensor diff = this->sub(other);
    return diff.mul(diff).mean();
}

Tensor Tensor::cross_entropy(const Tensor& targets) const
{
    if (shape_.size() != 2)
    {
        throw std::runtime_error("cross_entropy: logits must be 2-D [B, C]");
    }
    if (targets.shape_ != shape_)
    {
        throw std::runtime_error("cross_entropy: target shape mismatch");
    }
    return log_softmax(1).mul(targets).sum(1).neg().mean();
}

Tensor Tensor::cross_entropy(const std::vector<int64_t>& targets) const
{
    if (shape_.size() != 2)
    {
        throw std::runtime_error("cross_entropy: logits must be 2-D [B, C]");
    }
    if (static_cast<int64_t>(targets.size()) != shape_[0])
    {
        throw std::runtime_error("cross_entropy: target count != batch size");
    }
    return cross_entropy(one_hot(targets, shape_[1]));
}

Tensor Tensor::bce_with_logits(const Tensor& targets) const
{
    if (targets.shape_ != shape_)
    {
        throw std::runtime_error("bce_with_logits: shape mismatch");
    }
    // exp(-|x|) ∈ (0, 1]  ->  1 + exp(-|x|) ∈ (1, 2]，不存在灾难性抵消，不需要 log1p
    const Tensor softplus = ReLU() + (scalar(1.0f) + abs().neg().exp()).log();
    return (softplus - mul(targets)).mean();
}

Tensor Tensor::bce(const Tensor& targets) const
{
    if (targets.shape_ != shape_)
    {
        throw std::runtime_error("bce: shape mismatch");
    }
    const Tensor one = scalar(1.0f);
    const Tensor lp = log_clamped(*this);        // log(p)
    const Tensor l1p = log_clamped(one - *this); // log(1-p)
    return (targets * lp + (one - targets) * l1p).neg().mean();
}
