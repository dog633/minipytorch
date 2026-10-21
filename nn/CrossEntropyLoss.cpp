#include "CrossEntropyLoss.hpp"

#include <stdexcept>
#include <utility>

// 交叉熵损失层（声明见 CrossEntropyLoss.hpp）

CrossEntropyLoss::CrossEntropyLoss(std::vector<int64_t> targets)
    : targets_(std::move(targets))
{
}

void CrossEntropyLoss::set_target(std::vector<int64_t> targets)
{
    targets_ = std::move(targets);
}

Tensor CrossEntropyLoss::forward(const Tensor& logits)
{
    if (targets_.empty())
    {
        throw std::runtime_error("CrossEntropyLoss: target is not set");
    }
    // 本体就是张量库那个函数（反向也是现成的）
    return logits.cross_entropy(targets_);
}
