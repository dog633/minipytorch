#pragma once
// 均方误差损失层：把 Tensor::mse 包成 Module，好塞进 Sequential。
//
// 目标固定在层里（理由同 CrossEntropyLoss：Module::forward 只接一个输入）。
// 注意 Tensor 没有默认构造函数，所以没有默认构造的 MSELoss —— 目标必须在构造时给。
#include <cstdint>
#include <vector>

#include "Module.hpp"

class MSELoss : public Module
{
public:
    // 直接给目标张量
    explicit MSELoss(Tensor target);

    // 或者给数据 + 形状
    MSELoss(std::vector<float> data, std::vector<int64_t> shape);

    void set_target(Tensor target);

    // pred: 和目标同形状 -> 0 维标量
    Tensor forward(const Tensor& pred) override;

    const Tensor& target() const { return target_; }

private:
    Tensor target_;
};
