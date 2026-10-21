#include "Linear.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

// 全连接层（声明见 Linear.hpp）

namespace
{
    // 简单的线性同余发生器：只为初始化服务，不依赖 <random>，且结果可复现
    float uniform_signed(uint32_t& seed, float bound)
    {
        seed = seed * 1664525u + 1013904223u;
        const float u = static_cast<float>(seed >> 8) * (1.0f / 16777216.0f);   // [0, 1)
        return (2.0f * u - 1.0f) * bound;                                       // [-bound, bound)
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

    // 参数校验 + 算出初始化上界 1/√fan_in。
    // 在初始化列表里调用：非法尺寸会先抛异常，不会真的去分配/计算。
    // @param fan_in  权重按 PyTorch 的 kaiming_uniform_(a=√5) 展开后，上界只取决于 fan_in
    float init_bound(int64_t in_features, int64_t out_features)
    {
        if (in_features <= 0 || out_features <= 0)
        {
            throw std::runtime_error("Linear: in_features / out_features must be positive");
        }
        return 1.0f / std::sqrt(static_cast<float>(in_features));
    }
}

Linear::Linear(int64_t in_features, int64_t out_features, uint32_t seed)
    : weight(init_values(in_features * out_features, seed,
                         init_bound(in_features, out_features)),
             {in_features, out_features}),
      bias(init_values(out_features, seed,
                       init_bound(in_features, out_features)),
           {out_features})
{
    // 标记成叶子参数：之后所有用到它们的算子都会建图
    weight.requires_grad = true;
    bias.requires_grad = true;
}

Tensor Linear::forward(const Tensor& x)
{
    // [..., in] @ [in, out] -> [..., out]，bias 广播到最后一维
    return x.matmul(weight).add(bias);
}

std::vector<Tensor*> Linear::parameters()
{
    return {&weight, &bias};
}
