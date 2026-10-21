#include "AvgPool2d.hpp"

#include <stdexcept>

// 2D 平均池化（声明见 AvgPool2d.hpp）

AvgPool2d::AvgPool2d(int64_t kernel_size, int64_t stride, int64_t padding, bool count_include_pad)
    : kernel_size_(kernel_size),
      stride_(stride == 0 ? kernel_size : stride),   // 0 = "跟 kernel 一样"
      padding_(padding),
      count_include_pad_(count_include_pad)
{
    if (kernel_size_ <= 0)
    {
        throw std::runtime_error("AvgPool2d: kernel_size must be positive");
    }
    if (stride_ <= 0)
    {
        throw std::runtime_error("AvgPool2d: stride must be positive");
    }
    if (padding_ < 0)
    {
        throw std::runtime_error("AvgPool2d: padding must be non-negative");
    }
}

int64_t AvgPool2d::output_height(int64_t h) const
{
    return tensor_detail::conv_output_size(h, kernel_size_, stride_, padding_);
}

int64_t AvgPool2d::output_width(int64_t w) const
{
    return tensor_detail::conv_output_size(w, kernel_size_, stride_, padding_);
}

Tensor AvgPool2d::forward(const Tensor& x)
{
    if (x.shape().size() != 4)
    {
        throw std::runtime_error("AvgPool2d: input must be 4-D [N, C, H, W]");
    }
    const int64_t n = x.shape()[0];
    const int64_t channels = x.shape()[1];
    const int64_t out_h = output_height(x.shape()[2]);
    const int64_t out_w = output_width(x.shape()[3]);
    if (out_h <= 0 || out_w <= 0)
    {
        throw std::runtime_error("AvgPool2d: input is smaller than the kernel");
    }

    const int64_t window = kernel_size_ * kernel_size_;
    const int64_t cols = out_h * out_w;                                  // L

    // 平均池化的填充值就是 0（0 也是"参与平均"的合法值）
    const Tensor col = x.im2col(kernel_size_, kernel_size_, stride_, padding_, 0.0f);   // [N, C*k*k, L]
    const Tensor grouped = col.reshape({n, channels, window, cols});

    if (count_include_pad_ || padding_ == 0)
    {
        // 分母固定是 k*k（填充的 0 也算）
        return grouped.mean(2).reshape({n, channels, out_h, out_w});
    }

    // 分母只算落在输入里的位置：拿一个全 1 的张量做同样的 im2col，
    // 越界位置是 0，沿窗口维求和就是这个窗口的有效元素个数
    const Tensor ones = Tensor::ones(x.shape());
    const Tensor count = ones.im2col(kernel_size_, kernel_size_, stride_, padding_, 0.0f)
                             .reshape({n, channels, window, cols})
                             .sum(2);                                        // [N, C, L]
    return grouped.sum(2).div(count).reshape({n, channels, out_h, out_w});
}
