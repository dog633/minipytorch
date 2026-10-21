#pragma once
// 2D 卷积层，用 im2col 实现：
//
//     col = x.im2col(k, k, stride, padding)     // [N, C*k*k, L]，L = OH*OW
//     w2  = weight.reshape({O, C*k*k})          // 4 维权重摊平（reshape 有反向，梯度照样回到 4 维）
//     y   = w2.matmul(col)                      // [O,K] @ [N,K,L] -> [N, O, L]
//     y  += bias.reshape({O, 1})                // 广播到 L 上
//     y   = y.reshape({N, O, OH, OW})
//
// 前向就是"im2col + 矩阵乘 + 广播加 + reshape"，每一步的反向都是现成的
// （im2col 的反向是 col2im），所以**这个层里一行反向代码都没有**。
//
// weight 保持 PyTorch 的 4 维布局 [out_channels, in_channels, k, k]，
// 可以直接和 PyTorch 的卷积核逐元素对照。
#include <cstdint>
#include <vector>

#include "Module.hpp"

class Conv2d : public Module
{
public:
    // stride / padding 两个方向相同（per-axis 的版本以后要的话再加）。
    // 参数按 U(-1/√fan_in, 1/√fan_in) 初始化，fan_in = in_channels*k*k
    // （等价于 PyTorch 默认的 kaiming_uniform_(a=√5)）；seed 固定所以可复现。
    Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
           int64_t stride = 1, int64_t padding = 0, uint32_t seed = 12345u);

    // x: [N, C, H, W] -> [N, out_channels, OH, OW]
    Tensor forward(const Tensor& x) override;

    // 交给优化器：{&weight, &bias}
    std::vector<Tensor*> parameters() override;

    // 输出特征图的高宽（算下一层形状时用）
    int64_t output_height(int64_t h) const;
    int64_t output_width(int64_t w) const;

    Tensor weight;   // [out_channels, in_channels, k, k]
    Tensor bias;     // [out_channels]

private:
    int64_t in_channels_;
    int64_t out_channels_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
};
