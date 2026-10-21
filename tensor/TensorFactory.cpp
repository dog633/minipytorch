#include "Tensor.hpp"

// 生成函数（工厂）（声明在 Tensor.hpp，实现按模块拆到这里）

Tensor Tensor::zeros(std::vector<int64_t> shape)
{
    const int64_t n = numel(shape);
    return Tensor(std::vector<float>(static_cast<size_t>(n), 0.0f), std::move(shape));
}

Tensor Tensor::ones(std::vector<int64_t> shape)
{
    const int64_t n = numel(shape);
    return Tensor(std::vector<float>(static_cast<size_t>(n), 1.0f), std::move(shape));
}

Tensor Tensor::eye(int64_t n)
{
    return eye(n, n);
}

Tensor Tensor::eye(int64_t n, int64_t m)
{
    if (n < 0 || m < 0)
    {
        throw std::runtime_error("eye: negative size");
    }
    std::vector<float> data(static_cast<size_t>(n * m), 0.0f);
    const int64_t d = std::min(n, m);
    for (int64_t i = 0; i < d; ++i)
    {
        data[static_cast<size_t>(i * m + i)] = 1.0f;
    }
    return Tensor(std::move(data), {n, m});
}

Tensor Tensor::arange(float end)
{
    return arange(0.0f, end, 1.0f);
}

Tensor Tensor::arange(float start, float end, float step)
{
    if (step == 0.0f)
    {
        throw std::runtime_error("arange: step must not be zero");
    }

    const double span = static_cast<double>(end) - static_cast<double>(start);
    const double count = span / static_cast<double>(step);
    const int64_t n = (count > 0.0) ? static_cast<int64_t>(std::ceil(count)) : 0;

    std::vector<float> data(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
    {
        data[static_cast<size_t>(i)] = start + static_cast<float>(i) * step;
    }
    return Tensor(std::move(data), {n});
}

Tensor Tensor::one_hot(const std::vector<int64_t>& targets, int64_t num_classes)
{
    if (num_classes <= 0)
    {
        throw std::runtime_error("one_hot: num_classes must be positive");
    }
    const int64_t n = static_cast<int64_t>(targets.size());
    std::vector<float> data(static_cast<size_t>(n * num_classes), 0.0f);
    for (int64_t i = 0; i < n; ++i)
    {
        const int64_t t = targets[static_cast<size_t>(i)];
        if (t < 0 || t >= num_classes)
        {
            throw std::runtime_error("one_hot: target out of range");
        }
        data[static_cast<size_t>(i * num_classes + t)] = 1.0f;
    }
    return Tensor(std::move(data), {n, num_classes});
}
