#pragma once
// SiLU（a.k.a. Swish）激活层：silu(x) = x · sigmoid(x)
//
// 和 ReLU 层一样：真正干活的是自由函数 `Silu`，`SiLU` 只是把它包成一层。
// 因为只用现有算子（mul + sigmoid），反向是自动的，没写一行反向代码。
#include "Module.hpp"

// 激活函数：silu(x) = x · σ(x)
Tensor Silu(const Tensor& x);

class SiLU : public Module
{
public:
    Tensor forward(const Tensor& x) override;
};
