#include "Module.hpp"

// 层的基类（声明见 Module.hpp）

void Module::zero_grad()
{
    // 默认实现：问本层要参数，然后逐个清。
    // 有参层（Linear）直接用这份；无参层（ReLU）拿到空列表，等于什么都不做。
    for (Tensor* p : parameters())
    {
        p->zero_grad();
    }
}
