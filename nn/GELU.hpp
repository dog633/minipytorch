#pragma once
// GELU 激活层。
//
// 这里用的是 **tanh 近似式**（也就是 PyTorch `nn.GELU(approximate="tanh")`）：
//     gelu(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
//
// 为什么不用精确式：精确式是 0.5·x·(1 + erf(x/√2))，而张量库现在**没有 erf 算子**。
// 想在 nn 这一层不新增算子的前提下拿到可导的 GELU，就只能用 tanh 近似。
// 两者最大差约 1e-3（测试里量过），够用；要精确版就得给 Tensor 加一个 erf/gelu 算子。
#include "Module.hpp"

// 激活函数：GELU（tanh 近似）
Tensor Gelu(const Tensor& x);

class GELU : public Module
{
public:
    Tensor forward(const Tensor& x) override;
};
