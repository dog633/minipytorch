#include "Conv2d.hpp"

#include <cmath>
#include <stdexcept>

// 2D 卷积层（im2col 实现；声明见 Conv2d.hpp）

namespace
{
    // 简单的线性同余发生器：可复现、不依赖 <random>（和 Linear 里那套一样）
    float uniform_signed(uint32_t& seed, float bound)
    {
        seed = seed * 1664525u + 1013904223u;
        const float u = static_cast<float>(seed >> 8) * (1.0f / 16777216.0f);   // [0, 1)
        return (2.0f * u - 1.0f) * bound;
    }

    std::vector<float> init_values(int64_t count, uint32_t& seed, float bound)
    {
        std::vector<float> v(static_cast<size_t>(count));
        for (int64_t i = 0; i < count; ++i)
        {
            v[static_cast<size_t>(i)] = uniform_signed(seed, bound);
        }
        return v;
    }

    // 参数校验 + 初始化上界 1/√fan_in，fan_in = in_channels * k * k。
    // 在初始化列表里调用：尺寸非法会先抛异常，不会真的去分配/给负长度建向量。
    float conv_init_bound(int64_t in_channels, int64_t out_channels, int64_t kernel_size)
    {
        if (in_channels <= 0 || out_channels <= 0 || kernel_size <= 0)
        {
            throw std::runtime_error("Conv2d: in_channels / out_channels / kernel_size must be positive");
        }
        const float fan_in = static_cast<float>(in_channels * kernel_size * kernel_size);
        return 1.0f / std::sqrt(fan_in);
    }
}  // namespace

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
               int64_t stride, int64_t padding, uint32_t seed)
    : weight(init_values(out_channels * in_channels * kernel_size * kernel_size, seed,
                         conv_init_bound(in_channels, out_channels, kernel_size)),
             {out_channels, in_channels, kernel_size, kernel_size}),
      bias(init_values(out_channels, seed,
                       conv_init_bound(in_channels, out_channels, kernel_size)),
           {out_channels}),
      in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding)
{
    if (stride_ <= 0)
    {
        throw std::runtime_error("Conv2d: stride must be positive");
    }
    if (padding_ < 0)
    {
        throw std::runtime_error("Conv2d: padding must be non-negative");
    }

    // 叶子参数：之后所有用到它们的算子都会建图
    weight.requires_grad = true;
    bias.requires_grad = true;
}

int64_t Conv2d::output_height(int64_t h) const
{
    return tensor_detail::conv_output_size(h, kernel_size_, stride_, padding_);
}

int64_t Conv2d::output_width(int64_t w) const
{
    return tensor_detail::conv_output_size(w, kernel_size_, stride_, padding_);
}

Tensor Conv2d::forward(const Tensor& x)
{
    if (x.shape().size() != 4)
    {
        throw std::runtime_error("Conv2d: input must be 4-D [N, C, H, W]");
    }
    if (x.shape()[1] != in_channels_)
    {
        throw std::runtime_error("Conv2d: input channels mismatch");
    }

    const int64_t n = x.shape()[0];
    const int64_t out_h = output_height(x.shape()[2]);
    const int64_t out_w = output_width(x.shape()[3]);
    if (out_h <= 0 || out_w <= 0)
    {
        throw std::runtime_error("Conv2d: input is smaller than the kernel");
    }

    // 1) im2col：一次索引重排，反向往回散布（col2im），梯度能到 x
    const Tensor col = x.im2col(kernel_size_, kernel_size_, stride_, padding_);   // [N, K, L]

    // 2) 4 维权重摊平成 [O, K] 再乘。reshape 有反向，所以梯度会回到 4 维的 weight 上
    const int64_t k = in_channels_ * kernel_size_ * kernel_size_;
    const Tensor w2 = weight.reshape({out_channels_, k});

    // 3) [O,K] @ [N,K,L] 靠批次广播 -> [N,O,L]
    Tensor y = w2.matmul(col);

    // 4) bias [O] -> [O,1]：右对齐后正好落在 O 那一维上，广播到每个输出位置
    y = y.add(bias.reshape({out_channels_, 1}));

    // 5) [N,O,L] -> [N,O,OH,OW]（L = OH*OW，行优先，所以顺序天然对得上）
    return y.reshape({n, out_channels_, out_h, out_w});
}

std::vector<Tensor*> Conv2d::parameters()
{
    return {&weight, &bias};
}
