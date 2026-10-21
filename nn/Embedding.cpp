#include "Embedding.hpp"

#include <cmath>
#include <stdexcept>

// 查表层（声明见 Embedding.hpp）

namespace
{
    // 和 Linear / Conv2d 同一套可复现的 LCG
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

    // 校验放在初始化列表里用（尺寸非法时先抛，不会去给负长度建向量）
    int64_t check_positive(int64_t v, const char* name)
    {
        if (v <= 0)
        {
            throw std::runtime_error(std::string("Embedding: ") + name + " must be positive");
        }
        return v;
    }
}  // namespace

Embedding::Embedding(int64_t num_embeddings, int64_t embedding_dim, uint32_t seed)
    : weight(init_values(check_positive(num_embeddings, "num_embeddings")
                             * check_positive(embedding_dim, "embedding_dim"),
                         seed, 1.0f),
             {num_embeddings, embedding_dim}),
      num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim)
{
    weight.requires_grad = true;
}

Tensor Embedding::lookup(const std::vector<int64_t>& ids) const
{
    return weight.index_select(0, ids);   // [ids.size(), D]
}

Tensor Embedding::forward(const Tensor& ids)
{
    if (ids.shape().empty())
    {
        throw std::runtime_error("Embedding: ids must not be a scalar");
    }

    // 下标用 float 存整数：这里校验一下，免得 3.7 这种值被静默截断
    std::vector<int64_t> flat;
    flat.reserve(ids.data().size());
    for (float v : ids.data())
    {
        const float rounded = std::round(v);
        if (std::fabs(v - rounded) > 1e-4f)
        {
            throw std::runtime_error("Embedding: indices must be integers");
        }
        flat.push_back(static_cast<int64_t>(rounded));
    }

    // [m, D] -> ids 的形状 + [D]
    std::vector<int64_t> out_shape = ids.shape();
    out_shape.push_back(embedding_dim_);
    return lookup(flat).reshape(out_shape);
}

std::vector<Tensor*> Embedding::parameters()
{
    return {&weight};
}
