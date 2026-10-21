#pragma once
// LayerNorm（Ba et al., 2016）：对**最后 len(normalized_shape) 维**做标准化，再仿射变换。
//
//     μ   = mean(x, 最后 k 维)
//     σ²  = mean((x-μ)², 最后 k 维)            ← 有偏方差，和 PyTorch 一致
//     y   = (x-μ)/√(σ²+eps) · γ + β
//
// γ 初值全 1、β 初值全 0（和 PyTorch 一样，所以刚建好时只做标准化）。
// 和 BatchNorm 不同，它不依赖 batch 统计量，所以训练/推理行为完全一样。
#include <initializer_list>
#include <vector>

#include "Module.hpp"

class LayerNorm : public Module
{
public:
    // normalized_shape 支持三种写法（和 PyTorch 一样灵活）：
    //     LayerNorm(128)          // 只标准化最后一维
    //     LayerNorm({3, 4})       // 标准化最后两维
    //     LayerNorm(shape_vec)    // 形状由变量给
    // 三个重载不会打架：花括号列表会优先匹配 initializer_list（否则 {3,4} 会被
    // 静默解释成 "normalized_size=3, eps=4" 这种灾难）。
    explicit LayerNorm(int64_t normalized_size, float eps = 1e-5f,
                       bool elementwise_affine = true);
    explicit LayerNorm(std::initializer_list<int64_t> normalized_shape, float eps = 1e-5f,
                       bool elementwise_affine = true);
    explicit LayerNorm(const std::vector<int64_t>& normalized_shape, float eps = 1e-5f,
                       bool elementwise_affine = true);

    Tensor forward(const Tensor& x) override;
    std::vector<Tensor*> parameters() override;   // 关掉 affine 时返回空

    Tensor gamma;   // [normalized_shape]，初值全 1
    Tensor beta;    // [normalized_shape]，初值全 0

private:
    std::vector<int64_t> normalized_shape_;
    float eps_;
    bool affine_;
};
