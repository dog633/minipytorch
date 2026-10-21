#include "SGD.hpp"

#include <cmath>
#include <utility>

// SGD（声明见 SGD.hpp）
SGD::SGD(std::vector<Tensor*> params, float lr,
         float momentum, float weight_decay, bool nesterov)
    : params_(std::move(params)), lr_(lr),
      momentum_(momentum), weight_decay_(weight_decay), nesterov_(nesterov)
{
    if (momentum_ != 0.0f)
    {
        velocity_.reserve(params_.size());
        for (const Tensor* p : params_)
        {
            velocity_.push_back(Tensor::zeros(p->shape()));
        }
    }
}

void SGD::step()
{
    for (size_t i = 0; i < params_.size(); ++i)
    {
        Tensor* p = params_[i];
        const Tensor* g = p->grad();
        if (g == nullptr)
        {
            continue;   // 这一步没收到梯度（例如没参与前向）
        }

        // g = grad + weight_decay * param
        Tensor grad = *g;
        if (weight_decay_ != 0.0f)
        {
            grad = grad.add(p->mul(Tensor::scalar(weight_decay_)));
        }

        if (momentum_ != 0.0f)
        {
            // v = momentum * v + g
            Tensor& v = velocity_[i];
            v = v.mul(Tensor::scalar(momentum_)).add(grad);
            grad = nesterov_ ? grad.add(v.mul(Tensor::scalar(momentum_))) : v;
        }

        p->sub_inplace(grad, lr_);
    }
}

void SGD::zero_grad()
{
    for (Tensor* p : params_)
    {
        p->zero_grad();
    }
}
