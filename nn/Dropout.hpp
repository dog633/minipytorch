#pragma once
// Dropout（inverted dropout）：
//   训练时：每个元素以概率 p 被置 0，活下来的元素**放大 1/(1-p)**
//   推理时：原样返回（什么都不做）
//
// 那个 1/(1-p) 是关键：它让输出的期望等于输入，所以推理时不需要再乘任何系数
// （这就是 "inverted" 的含义，PyTorch 用的也是这套）。
//
// 实现上只需要现有算子：造一个 0 / 1/(1-p) 的掩码张量，然后逐元素乘。
// 掩码是常量，梯度自然就是 dx = g · mask —— 又是零反向代码。
#include <cstdint>

#include "Module.hpp"

class Dropout : public Module
{
public:
    // p 必须落在 [0, 1)：p=0 等于没扔，p=1 会把所有元素扔光（除以 0），所以不允许
    explicit Dropout(float p = 0.5f, uint32_t seed = 20240601u);

    Tensor forward(const Tensor& x) override;

    float p() const { return p_; }

private:
    float p_;
    uint32_t seed_;   // 每个实例自己的随机状态（同一实例连续调用会得到不同掩码）
};
