#pragma once
// 计算图的开关（PyTorch 的 torch.no_grad() / set_grad_enabled）。
//
// 默认是"记录"状态。关掉之后，所有算子都不再挂反向节点：
//   - 算子输出自然变成 requires_grad = false 的叶子；
//   - 因此**不会**为每个算子把输入深拷贝进 Node，也不会分配中间梯度 —— 推理时又快又省内存。
//
// 三种用法：
//     { NoGradGuard guard;  Tensor y = net.forward(x); }      // RAII，离开作用域自动恢复（可嵌套）
//     set_grad_enabled(false);  ...  set_grad_enabled(true);  // 手动
//     { GradModeGuard g(true);  ... }                         // 在 no_grad 里临时重新打开（微调冻结层等）
//
// 注意：这和 Module::eval() 是**两件事**。eval() 只切层的行为（比如关掉 Dropout），
// 不关 autograd —— PyTorch 也是这样，验证集上想算梯度就得让 autograd 开着。
#include <stdexcept>

// 当前是否记录计算图（每个线程一份）
bool is_grad_enabled();

// 手动设置开关
void set_grad_enabled(bool enabled);

// 通用守卫：构造时把开关设成 enabled，析构时恢复成进入前的值（可嵌套）
class GradModeGuard
{
public:
    explicit GradModeGuard(bool enabled) : prev_(is_grad_enabled())
    {
        set_grad_enabled(enabled);
    }

    ~GradModeGuard() { set_grad_enabled(prev_); }

    GradModeGuard(const GradModeGuard&) = delete;             // 不许拷贝：否则会恢复两次
    GradModeGuard& operator=(const GradModeGuard&) = delete;

private:
    bool prev_;
};

// 最常用的那个：进入作用域后不再记录计算图
class NoGradGuard : public GradModeGuard
{
public:
    NoGradGuard() : GradModeGuard(false) {}
};
