#pragma once
// Adam（Kingma & Ba, 2015）。
//
// 优化器持有的是**参数指针**，所以参数的生命周期必须长于优化器
// （别把临时 Tensor 的地址传进来）。
#include <vector>

#include "../tensor/Tensor.hpp"

// Adam（Kingma & Ba, 2015）
//   g  = grad + weight_decay * param
//   m  = b1 * m + (1 - b1) * g
//   v  = b2 * v + (1 - b2) * g²
//   m̂ = m / (1 - b1^t)，v̂ = v / (1 - b2^t)        （t 从 1 开始）
//   param -= lr * m̂ / (sqrt(v̂) + eps)
class Adam
{
public:
    Adam(std::vector<Tensor*> params, float lr = 1e-3f,
         float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
         float weight_decay = 0.0f);

    void step();
    void zero_grad();

private:
    std::vector<Tensor*> params_;
    std::vector<Tensor> m_;   // 一阶矩
    std::vector<Tensor> v_;   // 二阶矩
    int64_t t_ = 0;           // 步数（从 1 开始，用于偏差修正）
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
};
