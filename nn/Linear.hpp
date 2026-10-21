#pragma once
// 全连接层：y = x @ W + b
//
// 注意权重布局：这里存的是 **[in_features, out_features]**，和 PyTorch 的
// [out_features, in_features] 相反。原因是本库没有转置视图（transpose 会真搬数据），
// 存成 [in, out] 就能让前向只做一次 matmul、一次拷贝都不用。
#include <cstdint>
#include <vector>

#include "Module.hpp"

class Linear : public Module
{
public:
    // 参数按 U(-1/√fan_in, 1/√fan_in) 初始化 —— 这正是 PyTorch 默认的
    // kaiming_uniform_(a=√5) 在均匀分布上的等价形式（fan_in = in_features）。
    // seed 固定，所以每次构造出来的参数完全可复现（便于调试与对拍）。
    Linear(int64_t in_features, int64_t out_features, uint32_t seed = 12345u);

    // x: [..., in_features] -> [..., out_features]
    // 末尾维参与矩阵乘，前面的任意批次维都支持（也支持 x 是 1 维）
    Tensor forward(const Tensor& x) override;

    // 交给优化器：{&weight, &bias}（注意生命周期：层的存活期要长于优化器）
    std::vector<Tensor*> parameters() override;

    Tensor weight;   // [in, out]
    Tensor bias;     // [out]
};
