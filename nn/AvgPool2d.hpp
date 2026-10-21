#pragma once
// 2D 平均池化：每个 k×k 窗口取平均。
//
// 实现和 MaxPool2d 同构，只是把 max 换成 mean：
//     col = x.im2col(k, k, stride, padding, 0)      // 0 填充（平均值里也算 0）
//     col = col.reshape({N, C, k*k, L})
//     y   = col.mean(2)                             // 沿窗口维取平均 -> [N, C, L]
//     y   = y.reshape({N, C, OH, OW})
// 同样**零反向代码**（mean 的反向把梯度均分给窗口内每个位置）。
//
// count_include_pad = false 时（PyTorch 的参数）分母只算"落在输入里的位置个数"：
// 这时改成先 sum 再除以一个用 ones 张量算出来的计数（多一次 im2col 的开销）。
#include <cstdint>

#include "Module.hpp"

class AvgPool2d : public Module
{
public:
    // stride = 0 表示"和 kernel_size 一样"（PyTorch 的 stride=None）。
    // count_include_pad = true 是 PyTorch 的默认：填充的 0 也算进分母。
    explicit AvgPool2d(int64_t kernel_size, int64_t stride = 0, int64_t padding = 0,
                       bool count_include_pad = true);

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
    bool count_include_pad_;
};
