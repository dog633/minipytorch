#pragma once
// 激活层 ReLU：逐元素 max(x, 0)
//
// 层本身没有参数，真正干活的是自由函数 `ReLUctant`（名字是 ReLU + "ctant" 的花名，
// 免得和 Tensor 上的算子 `Tensor::ReLU()` 混在一起）。
#include "Module.hpp"

// 激活函数：relu(x) = max(x, 0)
Tensor ReLUctant(const Tensor& x);

// 把 ReLUctant 包成一层，好塞进 Sequential
class ReLU : public Module
{
public:
    Tensor forward(const Tensor& x) override;
};
