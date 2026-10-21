#pragma once
// 摊平层：把 [start_dim, end_dim] 这几维合并成一维。
//
// 默认 start_dim = 1、end_dim = -1，也就是"把 batch 维之后的所有维压平"——
// 和 PyTorch 的 nn.Flatten 默认行为一致：
//     [N, C, H, W] -> [N, C*H*W]      （卷积接全连接时最常用）
//     [N, C]       -> [N, C]          （start == end，原样返回）
//
// 前向就是一次 reshape，反向是 reshape 的现成反向，所以层里没有反向代码、也没有参数。
#include <cstdint>

#include "Module.hpp"

class Flatten : public Module
{
public:
    explicit Flatten(int64_t start_dim = 1, int64_t end_dim = -1);

    Tensor forward(const Tensor& x) override;

private:
    int64_t start_dim_;
    int64_t end_dim_;
};
