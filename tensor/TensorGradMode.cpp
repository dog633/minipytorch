#include "TensorGradMode.hpp"

// 计算图开关（声明见 TensorGradMode.hpp）

namespace
{
    // 函数内 static + thread_local：每个线程一份、按需初始化，不会有多重定义问题。
    // 关掉后出错了要容易发现，所以默认值是 true（记录）。
    bool& grad_enabled_flag()
    {
        static thread_local bool enabled = true;
        return enabled;
    }
}  // namespace

bool is_grad_enabled()
{
    return grad_enabled_flag();
}

void set_grad_enabled(bool enabled)
{
    grad_enabled_flag() = enabled;
}
