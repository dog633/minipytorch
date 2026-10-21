#include "Tensor.hpp"

// 形状变换（声明在 Tensor.hpp，实现按模块拆到这里）

Tensor Tensor::reshape(std::vector<int64_t> new_shape) const
{
    int64_t infer = -1;
    int64_t known = 1;
    for (size_t i = 0; i < new_shape.size(); ++i)
    {
        if (new_shape[i] == -1)
        {
            if (infer != -1)
            {
                throw std::runtime_error("reshape: only one -1 allowed");
            }
            infer = static_cast<int64_t>(i);
        }
        else
        {
            if (new_shape[i] < 0)
            {
                throw std::runtime_error("reshape: negative dim");
            }
            known *= new_shape[i];
        }
    }

    const int64_t total = static_cast<int64_t>(data_.size());

    if (infer >= 0)
    {
        if (known == 0 || total % known != 0)
        {
            throw std::runtime_error("reshape: cannot infer -1");
        }
        new_shape[infer] = total / known;
    }
    else
    {
        if (known != total)
        {
            throw std::runtime_error("reshape: size mismatch");
        }
    }
    Tensor res(data_, new_shape);
    // 反向：形状算子数据没动、只是换个看法，所以梯度 reshape 回输入形状即可
    const std::vector<int64_t> orig = shape_;
    attach_grad(res, {*this}, [orig](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, c.g.reshape(orig));
    });
    return res;
}

Tensor Tensor::flatten() const
{
    return reshape({static_cast<int64_t>(data_.size())});
}

Tensor Tensor::unsqueeze(int64_t dim) const
{
    const int64_t ndim = static_cast<int64_t>(shape_.size());

    if (dim < -ndim - 1 || dim > ndim)
    {
        throw std::runtime_error("unsqueeze: dim out of range");
    }
    if (dim < 0)
    {
        dim += ndim + 1;
    }
    std::vector<int64_t> new_shape;
    new_shape.reserve(ndim + 1);
    for (int64_t i = 0; i < ndim; ++i)
    {
        if (dim == i)
        {
            new_shape.push_back(1);
        }
        new_shape.push_back(shape_[i]);
    }
    if (dim == ndim)
    {
        new_shape.push_back(1);
    }
    Tensor res(data_, new_shape);
    // 反向：reshape 回输入形状（unsqueeze 就是把某一维变成 1）
    const std::vector<int64_t> orig = shape_;
    attach_grad(res, {*this}, [orig](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, c.g.reshape(orig));
    });
    return res;
}

Tensor Tensor::squeeze() const
{
    std::vector<int64_t> new_shape;
    for (auto d : shape_)
    {
        if (d != 1)
        {
            new_shape.push_back(d);
        }
    }
    Tensor res(data_, new_shape);
    // 反向：reshape 回输入形状
    const std::vector<int64_t> orig = shape_;
    attach_grad(res, {*this}, [orig](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, c.g.reshape(orig));
    });
    return res;
}

Tensor Tensor::squeeze(int64_t dim) const
{
    const int64_t ndim = static_cast<int64_t>(shape_.size());
    dim = check_dim(dim, ndim, "squeeze");

    if (shape_[dim] != 1)
    {
        throw std::runtime_error("squeeze: dim size is not 1");
    }
    std::vector<int64_t> new_shape;
    new_shape.reserve(static_cast<size_t>(ndim - 1));
    for (int64_t i = 0; i < ndim; ++i)
    {
        if (i != dim)
        {
            new_shape.push_back(shape_[i]);
        }
    }
    Tensor res(data_, new_shape);
    // 反向：reshape 回输入形状
    const std::vector<int64_t> orig = shape_;
    attach_grad(res, {*this}, [orig](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, c.g.reshape(orig));
    });
    return res;
}

Tensor Tensor::concat(const Tensor& other, int64_t dim) const
{
    const int64_t ndim = static_cast<int64_t>(shape_.size());
    if (shape_.size() != other.shape_.size())
    {
        throw std::runtime_error("concat: ndim mismatch");
    }
    dim = check_dim(dim, ndim, "concat");

    for (int64_t i = 0; i < ndim; ++i)
    {
        if (i != dim && shape_[i] != other.shape_[i])
        {
            throw std::runtime_error("concat: shape mismatch");
        }
    }

    const int64_t a_dim = shape_[dim];
    const int64_t b_dim = other.shape_[dim];
    const OuterInner seg = outer_inner(dim);

    std::vector<int64_t> out_shape = shape_;
    out_shape[dim] = a_dim + b_dim;
    std::vector<float> out(static_cast<size_t>(seg.outer * out_shape[dim] * seg.inner));

    for (int64_t o = 0; o < seg.outer; ++o)
    {
        std::copy(data_.begin() + o * a_dim * seg.inner,
                  data_.begin() + (o + 1) * a_dim * seg.inner,
                  out.begin() + o * out_shape[dim] * seg.inner);
        std::copy(other.data_.begin() + o * b_dim * seg.inner,
                  other.data_.begin() + (o + 1) * b_dim * seg.inner,
                  out.begin() + (o * out_shape[dim] + a_dim) * seg.inner);
    }
    Tensor res(std::move(out), out_shape);

    // 反向：沿拼接维把 g 切回两块（前 a_dim 个给 a，接着 b_dim 个给 b）
    attach_grad(res, {*this, other}, [dim, a_dim, b_dim](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, slice_dim(c.g, dim, 0, a_dim));
        c.grad(1, slice_dim(c.g, dim, a_dim, b_dim));
    });
    return res;
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const
{
    const int64_t ndim = static_cast<int64_t>(shape_.size());
    dim0 = check_dim(dim0, ndim, "transpose");
    dim1 = check_dim(dim1, ndim, "transpose");
    if (dim0 == dim1)
    {
        return *this;
    }

    std::vector<int64_t> out_shape = shape_;
    std::swap(out_shape[dim0], out_shape[dim1]);

    const int64_t total = static_cast<int64_t>(data_.size());
    std::vector<float> out(static_cast<size_t>(total));

    std::vector<int64_t> in_stride(ndim, 1);
    for (int64_t i = ndim - 2; i >= 0; --i)
    {
        in_stride[i] = in_stride[i + 1] * shape_[i + 1];
    }

    std::vector<int64_t> idx(ndim, 0);

    for (int64_t t = 0; t < total; ++t)
    {
        int64_t in_pos = 0;
        for (int64_t d = 0; d < ndim; ++d)
        {
            const int64_t src_d = (d == dim0) ? dim1 : (d == dim1) ? dim0 : d;
            in_pos += idx[d] * in_stride[src_d];
        }
        out[t] = data_[in_pos];

        next_index(idx, out_shape);
    }
    Tensor res(std::move(out), std::move(out_shape));
    // 反向：transpose 是自己的逆运算，交换同样两维即可
    attach_grad(res, {*this}, [dim0, dim1](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, c.g.transpose(dim0, dim1));
    });
    return res;
}
