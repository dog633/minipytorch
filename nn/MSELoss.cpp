#include "MSELoss.hpp"

#include <utility>

// 均方误差损失层（声明见 MSELoss.hpp）

MSELoss::MSELoss(Tensor target) : target_(std::move(target))
{
}

MSELoss::MSELoss(std::vector<float> data, std::vector<int64_t> shape)
    : target_(std::move(data), std::move(shape))
{
}

void MSELoss::set_target(Tensor target)
{
    target_ = std::move(target);
}

Tensor MSELoss::forward(const Tensor& pred)
{
    // 本体就是张量库那个函数（反向也是现成的）
    return pred.mse(target_);
}
