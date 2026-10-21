#include "Flatten.hpp"

#include <stdexcept>
#include <vector>

// 摊平层（声明见 Flatten.hpp）

namespace
{
    // 负数维按 Python 习惯换算，并检查范围。
    // Tensor::check_dim 是张量库的私有工具，nn 层用不了，所以这里自己来
    int64_t normalize_dim(int64_t dim, int64_t ndim, const char* what)
    {
        if (dim < -ndim || dim >= ndim)
        {
            throw std::runtime_error(std::string("Flatten: ") + what + " out of range");
        }
        return dim < 0 ? dim + ndim : dim;
    }
}  // namespace

Flatten::Flatten(int64_t start_dim, int64_t end_dim)
    : start_dim_(start_dim), end_dim_(end_dim)
{
}

Tensor Flatten::forward(const Tensor& x)
{
    const int64_t ndim = static_cast<int64_t>(x.shape().size());
    if (ndim == 0)
    {
        throw std::runtime_error("Flatten: scalar has no dim");
    }

    const int64_t start = normalize_dim(start_dim_, ndim, "start_dim");
    const int64_t end = normalize_dim(end_dim_, ndim, "end_dim");
    if (start > end)
    {
        throw std::runtime_error("Flatten: start_dim must not come after end_dim");
    }
    if (start == end)
    {
        return x;   // 只有一维要合并 -> 什么都不用做
    }

    // [d0, ..., start-1] + [start..end 的乘积] + [end+1, ...]
    std::vector<int64_t> new_shape;
    new_shape.reserve(static_cast<size_t>(ndim - (end - start)));
    for (int64_t i = 0; i < start; ++i)
    {
        new_shape.push_back(x.shape()[static_cast<size_t>(i)]);
    }
    int64_t flat = 1;
    for (int64_t i = start; i <= end; ++i)
    {
        flat *= x.shape()[static_cast<size_t>(i)];
    }
    new_shape.push_back(flat);
    for (int64_t i = end + 1; i < ndim; ++i)
    {
        new_shape.push_back(x.shape()[static_cast<size_t>(i)]);
    }

    // reshape 自带反向（梯度 reshape 回原形状），所以这一层没有反向代码
    return x.reshape(std::move(new_shape));
}
