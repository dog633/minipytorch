#include "Tensor.hpp"

// 卷积用的 im2col / col2im（声明在 Tensor.hpp）
//
// 为什么把 im2col 放在张量层而不是 nn 层：它是一次"按索引重排数据"的操作
// （本质是 gather），用现有的 reshape/transpose/concat 拼不出来 —— 拼出来的中间
// 张量是构造函数造出来的叶子，梯度会断在输入上。所以它必须是一个自带反向的算子。
//
// 两个函数是同一套循环的正反两遍：
//   im2col：x[n][c][ih][iw] -> col[n][k][l]        （读）
//   col2im：col[n][k][l]    -> dx[n][c][ih][iw] += （累加，因为 patch 会重叠）

namespace
{
    // 输出特征图的高宽（floor 除法，见 tensor_detail::conv_output_size）
    int64_t out_size(int64_t in, int64_t k, int64_t stride, int64_t padding)
    {
        return tensor_detail::conv_output_size(in, k, stride, padding);
    }

    void check_im2col_args(int64_t kh, int64_t kw, int64_t stride, int64_t padding)
    {
        if (kh <= 0 || kw <= 0)
        {
            throw std::runtime_error("im2col: kernel size must be positive");
        }
        if (stride <= 0)
        {
            throw std::runtime_error("im2col: stride must be positive");
        }
        if (padding < 0)
        {
            throw std::runtime_error("im2col: padding must be non-negative");
        }
    }
}  // namespace

Tensor Tensor::im2col(int64_t kh, int64_t kw, int64_t stride, int64_t padding, float pad_value) const
{
    if (shape_.size() != 4)
    {
        throw std::runtime_error("im2col: input must be 4-D [N, C, H, W]");
    }
    check_im2col_args(kh, kw, stride, padding);

    const int64_t n_batch = shape_[0], channels = shape_[1], height = shape_[2], width = shape_[3];
    const int64_t out_h = out_size(height, kh, stride, padding);
    const int64_t out_w = out_size(width, kw, stride, padding);
    if (out_h <= 0 || out_w <= 0)
    {
        throw std::runtime_error("im2col: kernel larger than padded input");
    }

    const int64_t rows = channels * kh * kw;   // K
    const int64_t cols = out_h * out_w;        // L
    // 越界位置保持 pad_value：卷积填 0（零填充），max 池化填 -inf
    std::vector<float> col(static_cast<size_t>(n_batch * rows * cols), pad_value);

    // 循环顺序把 oh/ow 放最内层：读 x 时是顺序访问，写 col 也是顺序访问
    for (int64_t n = 0; n < n_batch; ++n)
    {
        for (int64_t c = 0; c < channels; ++c)
        {
            for (int64_t i = 0; i < kh; ++i)
            {
                for (int64_t j = 0; j < kw; ++j)
                {
                    const int64_t k = (c * kh + i) * kw + j;
                    for (int64_t oh = 0; oh < out_h; ++oh)
                    {
                        const int64_t ih = oh * stride + i - padding;
                        if (ih < 0 || ih >= height)
                        {
                            continue;   // 越界保持 0（零填充）
                        }
                        for (int64_t ow = 0; ow < out_w; ++ow)
                        {
                            const int64_t iw = ow * stride + j - padding;
                            if (iw < 0 || iw >= width)
                            {
                                continue;
                            }
                            col[static_cast<size_t>((n * rows + k) * cols + oh * out_w + ow)] =
                                data_[static_cast<size_t>(((n * channels + c) * height + ih) * width + iw)];
                        }
                    }
                }
            }
        }
    }

    Tensor out(std::move(col), {n_batch, rows, cols});
    // 反向只需要输入形状，所以把 shape_ 按值捕获，不用为了拿形状去拷贝整个输入
    const std::vector<int64_t> x_shape = shape_;
    attach_grad(out, {*this}, [x_shape, kh, kw, stride, padding](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, col2im(c.g, x_shape, kh, kw, stride, padding));
    });
    return out;
}

Tensor Tensor::col2im(const Tensor& dcol, const std::vector<int64_t>& x_shape,
                      int64_t kh, int64_t kw, int64_t stride, int64_t padding)
{
    const int64_t n_batch = x_shape[0], channels = x_shape[1];
    const int64_t height = x_shape[2], width = x_shape[3];
    const int64_t out_h = out_size(height, kh, stride, padding);
    const int64_t out_w = out_size(width, kw, stride, padding);
    const int64_t rows = channels * kh * kw;
    const int64_t cols = out_h * out_w;

    if (dcol.shape() != std::vector<int64_t>({n_batch, rows, cols}))
    {
        throw std::runtime_error("col2im: grad shape mismatch");
    }

    std::vector<float> dx(static_cast<size_t>(n_batch * channels * height * width), 0.0f);

    for (int64_t n = 0; n < n_batch; ++n)
    {
        for (int64_t c = 0; c < channels; ++c)
        {
            for (int64_t i = 0; i < kh; ++i)
            {
                for (int64_t j = 0; j < kw; ++j)
                {
                    const int64_t k = (c * kh + i) * kw + j;
                    for (int64_t oh = 0; oh < out_h; ++oh)
                    {
                        const int64_t ih = oh * stride + i - padding;
                        if (ih < 0 || ih >= height)
                        {
                            continue;   // 来自填充位置的梯度没有归宿，丢掉
                        }
                        for (int64_t ow = 0; ow < out_w; ++ow)
                        {
                            const int64_t iw = ow * stride + j - padding;
                            if (iw < 0 || iw >= width)
                            {
                                continue;
                            }
                            // 关键：+= 而不是 =（stride < kernel 时同一个输入像素会被多个 patch 覆盖）
                            dx[static_cast<size_t>(((n * channels + c) * height + ih) * width + iw)] +=
                                dcol.data()[static_cast<size_t>((n * rows + k) * cols + oh * out_w + ow)];
                        }
                    }
                }
            }
        }
    }
    return Tensor(std::move(dx), x_shape);
}
