#include "GELU.hpp"

#include <cmath>

// GELU 激活（tanh 近似，声明见 GELU.hpp）

Tensor Gelu(const Tensor& x)
{
    const float k = std::sqrt(2.0f / 3.14159265358979323846f);   // √(2/π)
    const float c = 0.044715f;

    // 0.5·x·(1 + tanh(k·(x + c·x³)))
    const Tensor inner = x.add(x.pow(3.0f).mul(Tensor::scalar(c))).mul(Tensor::scalar(k));
    return x.mul(inner.tanh().add(Tensor::scalar(1.0f))).mul(Tensor::scalar(0.5f));
}

Tensor GELU::forward(const Tensor& x)
{
    return Gelu(x);
}
