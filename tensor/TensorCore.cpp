#include "Tensor.hpp"

// 构造函数与基本状态访问（声明在 Tensor.hpp，实现按模块拆到这里）

Tensor Tensor::detach() const
{
    return Tensor(data_, shape_);
}

float Tensor::item() const
{
    if (data_.size() != 1)
    {
        throw std::runtime_error("item: tensor must have exactly one element");
    }
    return data_[0];
}

void Tensor::sub_inplace(const Tensor& other, float scale)
{
    if (shape_ != other.shape_)
    {
        throw std::runtime_error("sub_inplace: shape mismatch");
    }
    // 直接改 data_：不建图、不分配，供优化器更新参数
    for (size_t i = 0; i < data_.size(); ++i)
    {
        data_[i] -= scale * other.data_[i];
    }
}

Tensor::Tensor(std::vector<float> data, std::vector<int64_t> shape)
    : data_(std::move(data)), shape_(std::move(shape)),
      meta_(std::make_shared<tensor_detail::AutogradMeta>())
{
    if (static_cast<int64_t>(data_.size()) != numel(shape_))
    {
        throw std::runtime_error("Tensor: data size != shape product");
    }
}
