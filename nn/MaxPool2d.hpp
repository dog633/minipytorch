#pragma once
// 2D 最大池化：每个 k×k 窗口取最大值。
//
// 实现方式是把 im2col 和 max 接起来（所以这一层同样没有反向代码）：
//     col = x.im2col(k, k, stride, padding, -inf)   // [N, C*k*k, L]
//     col = col.reshape({N, C, k*k, L})             // 把"同一通道的 k*k 个位置"聚到一起
//     y   = col.max(2)                              // 沿窗口维取最大 -> [N, C, L]
//     y   = y.reshape({N, C, OH, OW})
//
// **填充值必须是 -inf 而不是 0**：否则窗口里全是负数时，0 会当上最大值，输出就错了。
//
// 两个和 PyTorch 的差异（都源于"复用 max 算子"）：
//   1) 反向时梯度会均分给窗口内**所有并列**的最大值（PyTorch 只给其中一个）；
//   2) im2col 会把输入展开成 k*k 倍大小（PyTorch 的 max_pool 是直接算的，没有这块开销）。
#include <cstdint>

#include "Module.hpp"

class MaxPool2d : public Module
{
public:
    // stride = 0 表示"和 kernel_size 一样"（PyTorch 里 stride=None 的行为）。
    // kernel / stride / padding 两个方向相同。
    explicit MaxPool2d(int64_t kernel_size, int64_t stride = 0, int64_t padding = 0);

    // x: [N, C, H, W] -> [N, C, OH, OW]
    Tensor forward(const Tensor& x) override;

    int64_t output_height(int64_t h) const;
    int64_t output_width(int64_t w) const;

    int64_t kernel_size() const { return kernel_size_; }
    int64_t stride() const { return stride_; }

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
};
