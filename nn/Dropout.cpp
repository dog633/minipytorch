#include "Dropout.hpp"

#include <stdexcept>
#include <vector>

// Dropout（声明见 Dropout.hpp）

Dropout::Dropout(float p, uint32_t seed) : p_(p), seed_(seed)
{
    if (!(p_ >= 0.0f && p_ < 1.0f))
    {
        throw std::runtime_error("Dropout: p must be in [0, 1)");
    }
}

Tensor Dropout::forward(const Tensor& x)
{
    if (!is_training() || p_ == 0.0f)
    {
        return x;   // 推理模式（或 p=0）：恒等
    }

    const float keep_scale = 1.0f / (1.0f - p_);   // inverted dropout 的补偿系数
    std::vector<float> mask(x.data().size());
    for (float& m : mask)
    {
        // 简单的线性同余发生器：自带的随机状态，不依赖 <random>
        seed_ = seed_ * 1664525u + 1013904223u;
        const float u = static_cast<float>(seed_ >> 8) * (1.0f / 16777216.0f);   // [0, 1)
        m = (u < p_) ? 0.0f : keep_scale;
    }
    // 掩码是常量张量，反向就是 dx = g · mask（被扔掉的位置梯度也是 0）
    return x.mul(Tensor(std::move(mask), x.shape()));
}
