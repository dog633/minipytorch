#include "Sequential.hpp"

// 顺序容器（声明见 Sequential.hpp）

Tensor Sequential::forward(const Tensor& x)
{
    Tensor y = x;
    for (const std::shared_ptr<Module>& layer : layers_)
    {
        y = layer->forward(y);   // 前一层的输出就是后一层的输入；每次都是新张量
    }
    return y;
}

std::vector<Tensor*> Sequential::parameters()
{
    std::vector<Tensor*> all;
    for (const std::shared_ptr<Module>& layer : layers_)
    {
        const std::vector<Tensor*> one = layer->parameters();
        all.insert(all.end(), one.begin(), one.end());
    }
    return all;
}

void Sequential::set_training(bool mode)
{
    // 模式必须传到底：net.eval() 之后里面的 Dropout 才会真的关掉
    for (const std::shared_ptr<Module>& layer : layers_)
    {
        layer->train(mode);
    }
}
