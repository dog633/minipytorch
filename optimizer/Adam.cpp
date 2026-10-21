#include "Adam.hpp"

#include <cmath>
#include <utility>

// Adam（声明见 Adam.hpp）
Adam::Adam(std::vector<Tensor*> params, float lr,
           float beta1, float beta2, float eps, float weight_decay)
    : params_(std::move(params)), lr_(lr),
      beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay)
{
    m_.reserve(params_.size());
    v_.reserve(params_.size());
    for (const Tensor* p : params_)
    {
        m_.push_back(Tensor::zeros(p->shape()));
        v_.push_back(Tensor::zeros(p->shape()));
    }
}

void Adam::step()
{
    ++t_;
    const float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));   // 1 - b1^t
    const float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));   // 1 - b2^t

    for (size_t i = 0; i < params_.size(); ++i)
    {
        Tensor* p = params_[i];
        const Tensor* g = p->grad();
        if (g == nullptr)
        {
            continue;
        }

        Tensor grad = *g;
        if (weight_decay_ != 0.0f)
        {
            grad = grad.add(p->mul(Tensor::scalar(weight_decay_)));
        }

        // m = b1*m + (1-b1)*g，v = b2*v + (1-b2)*g²
        m_[i] = m_[i].mul(Tensor::scalar(beta1_)).add(grad.mul(Tensor::scalar(1.0f - beta1_)));
        v_[i] = v_[i].mul(Tensor::scalar(beta2_)).add(grad.mul(grad).mul(Tensor::scalar(1.0f - beta2_)));

        // 偏差修正 + 更新：p -= lr * m̂ / (sqrt(v̂) + eps)
        const Tensor m_hat = m_[i].div(Tensor::scalar(bc1));
        const Tensor v_hat = v_[i].div(Tensor::scalar(bc2));
        const Tensor delta = m_hat.div(v_hat.sqrt().add(Tensor::scalar(eps_)));
        p->sub_inplace(delta, lr_);
    }
}

void Adam::zero_grad()
{
    for (Tensor* p : params_)
    {
        p->zero_grad();
    }
}
