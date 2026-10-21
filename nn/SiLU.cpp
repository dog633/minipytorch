#include "SiLU.hpp"

// SiLU / Swish 激活（声明见 SiLU.hpp）

Tensor Silu(const Tensor& x)
{
    // x · sigmoid(x)：两个现成算子，链式法则自动搞定反向
    return x.mul(x.sigmoid());
}

Tensor SiLU::forward(const Tensor& x)
{
    return Silu(x);
}
