#pragma once
// 随机梯度下降（支持动量 / 权重衰减 / Nesterov）。
//
// 优化器持有的是**参数指针**，所以参数的生命周期必须长于优化器
// （别把临时 Tensor 的地址传进来）。
#include <vector>

#include "../tensor/Tensor.hpp"

// 随机梯度下降（带动量 / 权重衰减 / Nesterov）
//   g  = grad + weight_decay * param
//   v  = momentum * v + g                       （第一次 v = g）
//   g' = nesterov ? g + momentum * v : v
//   param -= lr * g'
class SGD
{
public:
    SGD(std::vector<Tensor*> params, float lr,
        float momentum = 0.0f, float weight_decay = 0.0f, bool nesterov = false);

    void step();
    void zero_grad();

private:
    std::vector<Tensor*> params_;
    std::vector<Tensor> velocity_;   // 动量缓冲，仅在 momentum != 0 时使用
    float lr_;
    float momentum_;
    float weight_decay_;
    bool nesterov_;
};
