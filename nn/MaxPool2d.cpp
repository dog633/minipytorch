#include "MaxPool2d.hpp"

#include <limits>
#include <stdexcept>

// 2D 最大池化（声明见 MaxPool2d.hpp）

MaxPool2d::MaxPool2d(int64_t kernel_size, int64_t stride, int64_t padding)
    : kernel_size_(kernel_size),
      stride_(stride == 0 ? kernel_size : stride),   // 0 = "跟 kernel 一样"
      padding_(padding)
{
    if (kernel_size_ <= 0)
    {
        throw std::runtime_error("MaxPool2d: kernel_size must be positive");
    }
    if (stride_ <= 0)
    {
        throw std::runtime_error("MaxPool2d: stride must be positive");
    }
    if (padding_ < 0)
    {
        throw std::runtime_error("MaxPool2d: padding must be non-negative");
    }
}

int64_t MaxPool2d::output_height(int64_t h) const
{
    return tensor_detail::conv_output_size(h, kernel_size_, stride_, padding_);
}

int64_t MaxPool2d::output_width(int64_t w) const
{
    return tensor_detail::conv_output_size(w, kernel_size_, stride_, padding_);
}

Tensor MaxPool2d::forward(const Tensor& x)
{
    if (x.shape().size() != 4)
    {
        throw std::runtime_error("MaxPool2d: input must be 4-D [N, C, H, W]");
    }
    const int64_t n = x.shape()[0];
    const int64_t channels = x.shape()[1];
    const int64_t out_h = output_height(x.shape()[2]);
    const int64_t out_w = output_width(x.shape()[3]);
    if (out_h <= 0 || out_w <= 0)
    {
        throw std::runtime_error("MaxPool2d: input is smaller than the kernel");
    }

    const int64_t window = kernel_size_ * kernel_size_;
    const int64_t cols = out_h * out_w;                                  // L

    // 1) 展开成窗口矩阵，越界填 -inf（关键：不能用 0）
    const Tensor col = x.im2col(kernel_size_, kernel_size_, stride_, padding_,
                                -std::numeric_limits<float>::infinity());   // [N, C*k*k, L]

    // 2) K 维的顺序是 (c, i, j)，所以每个通道的 k*k 行是连续的，reshape 正好把它们聚成一组
    const Tensor grouped = col.reshape({n, channels, window, cols});

    // 3) 沿窗口那一维取最大：max 的反向会把梯度送回窗口里的最大值位置
    const Tensor pooled = grouped.max(2);                                // [N, C, L]

    // 4) [N, C, L] -> [N, C, OH, OW]（L = OH*OW，行优先，顺序天然对得上）
    return pooled.reshape({n, channels, out_h, out_w});
}
