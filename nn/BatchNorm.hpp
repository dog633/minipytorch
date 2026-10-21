#pragma once
// 批归一化（Ioffe & Szegedy, 2015）。
//
// 一份实现覆盖 PyTorch 的 BatchNorm1d / 2d / 3d：输入是 [N, C, ...]，
// 就按**通道 C 以外的所有维**（batch 维 + 空间维）统计，逐通道归一化。
//     BatchNorm(3).forward([8, 3])        相当于 BatchNorm1d 处理 [N, C]
//     BatchNorm(3).forward([8, 3, 6, 6])  相当于 BatchNorm2d 处理 [N, C, H, W]
//
//     y = (x - μ) / √(σ² + eps) · γ + β
//
// 训练模式：μ / σ² 用**当前 batch** 的统计量，同时把 running_mean / running_var 按滑动平均更新；
// 推理模式：μ / σ² 直接用 running_mean / running_var（固定不动）。
//
// 两个容易踩的细节（和 PyTorch 保持一致）：
//   1) 归一化用的是**有偏**方差（÷ n），而 running_var 的滑动平均用的是**无偏**方差（÷ (n-1)）；
//      这里 n = N × 空间元素数，是"每个通道的元素个数"。
//   2) running_* 只在训练模式更新，并且必须**脱离计算图**（它们是统计量，不参与反向）。
//
// 它是本项目里第一个"有状态"的层：前向会改自己的 running_*，
// 所以 Module::forward 当初故意不加 const 是对的。
#include <cstdint>
#include <vector>

#include "Module.hpp"

class BatchNorm : public Module
{
public:
    // num_features = 通道数 C；eps 和 momentum 的含义同 PyTorch（momentum 是滑动平均系数）
    explicit BatchNorm(int64_t num_features, float eps = 1e-5f, float momentum = 0.1f,
                       bool affine = true);

    // x: [N, C, ...] -> 同形状
    Tensor forward(const Tensor& x) override;

    // 只有 γ / β 是参数（affine = false 时返回空）；running_* 是统计量，不给优化器
    std::vector<Tensor*> parameters() override;

    int64_t num_features() const { return num_features_; }
    bool is_affine() const { return affine_; }

    Tensor weight;         // γ，初值全 1
    Tensor bias;           // β，初值全 0
    Tensor running_mean;   // 初值全 0
    Tensor running_var;    // 初值全 1

private:
    int64_t num_features_;
    float eps_;
    float momentum_;
    bool affine_;
};
