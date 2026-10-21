#pragma once
// 交叉熵损失层：把张量库的 cross_entropy 包成 Module，这样它能直接坐在 Sequential 末尾。
//
//     Sequential(Linear(2, 3), CrossEntropyLoss({0, 1}))   // 前向直接得到标量损失
//
// 和 PyTorch 的差别：PyTorch 的 criterion(output, target) 是"前向时把 target 也传进去"，
// 而 Module::forward 只接一个输入，所以这里让**目标固定在层里**（构造时给，或 set_target 改）。
// 只是想要一个损失值的话，直接调 Tensor::cross_entropy(targets) 更顺手 —— 这个层是为了能塞进 Sequential。
#include <cstdint>
#include <vector>

#include "Module.hpp"

class CrossEntropyLoss : public Module
{
public:
    CrossEntropyLoss() = default;

    // targets 是整数类别下标，长度 = batch
    explicit CrossEntropyLoss(std::vector<int64_t> targets);

    // 换一批目标（例如每个 batch 换标签）
    void set_target(std::vector<int64_t> targets);

    // logits: [B, C] -> 0 维标量
    Tensor forward(const Tensor& logits) override;

private:
    std::vector<int64_t> targets_;
};
