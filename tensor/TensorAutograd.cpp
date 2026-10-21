#include "Tensor.hpp"

// 自动微分的实现（反向传播与内部工具）（声明在 Tensor.hpp，实现按模块拆到这里）

void Tensor::accumulate_grad(const Tensor& t, const Tensor& g)
{
if (!t.requires_grad)
{
    return;   // 不需要梯度的输入直接跳过
}
if (t.shape_ != g.shape_)
{
    throw std::runtime_error("backward: grad shape mismatch");
}

if (t.meta_->grad == nullptr)
{
    t.meta_->grad = std::make_shared<Tensor>(
        std::vector<float>(g.data_.size(), 0.0f), t.shape_);
}
Tensor& slot = *t.meta_->grad;
for (size_t i = 0; i < slot.data_.size(); ++i)
{
    slot.data_[i] += g.data_[i];
}
}

Tensor Tensor::reduce_to(const Tensor& g, const std::vector<int64_t>& target)
{
Tensor r = g.detach();
while (r.shape_.size() > target.size())
{
    r = r.sum(0, false);
}
for (size_t i = 0; i < target.size(); ++i)
{
    if (target[i] == 1 && r.shape_[i] != 1)
    {
        r = r.sum(static_cast<int64_t>(i), true);
    }
}
return r;
}

Tensor Tensor::expand_like(const Tensor& g, const std::vector<int64_t>& target)
{
return ones(target).mul(g.detach());
}

void Tensor::collect_topo(const std::shared_ptr<tensor_detail::Node>& n,
                             std::vector<std::shared_ptr<tensor_detail::Node>>& order)
{
if (n == nullptr || n->visited)
{
    return;
}
n->visited = true;
for (const Tensor& in : n->inputs)
{
    collect_topo(in.meta_->grad_fn, order);
}
order.push_back(n);
}

void Tensor::backward()
{
if (!is_grad_enabled())
{
    // no_grad 作用域里根本没建图，这里再反传只会静默什么都不做 —— 直接报出来
    throw std::runtime_error("backward: called inside a no_grad scope (grad recording is off)");
}

accumulate_grad(*this, ones(shape_));

if (meta_->grad_fn == nullptr)
{
    return;   // 叶子自己就是终点
}

std::vector<std::shared_ptr<tensor_detail::Node>> order;
collect_topo(meta_->grad_fn, order);

for (auto it = order.rbegin(); it != order.rend(); ++it)
{
    const std::shared_ptr<tensor_detail::Node>& n = *it;
    if (n->out_meta->grad == nullptr)
    {
        continue;   // 这个输出没收到上游梯度（比如分支里没被用到的那些）
    }
    // 组装反向下文：输入（活的，梯度写回它们的 meta_）+ 上游梯度（detach 的副本）
    const tensor_detail::BackwardCtx ctx{&n->inputs, n->out_meta->grad->detach()};
    n->backward(ctx);
}

for (const auto& n : order)   // 清掉访问标记，允许再次 backward
{
    n->visited = false;
}
}
