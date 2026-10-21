#include "ReLU.hpp"

// ReLU 层（声明见 ReLU.hpp）

Tensor ReLUctant(const Tensor& x)
{
    // 直接复用张量库里的算子（反向也是现成的：dx = g · (x > 0)）
    return x.ReLU();
}

Tensor ReLU::forward(const Tensor& x)
{
    return ReLUctant(x);
}
