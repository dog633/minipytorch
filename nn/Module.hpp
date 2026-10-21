#pragma once
// 所有层的基类 —— Sequential 靠它才能把不同类型的层装进同一个容器。
//
// 一个层需要提供两件事：
//   1) forward(x)：前向计算
//   2) parameters()：本层的可训练参数（无参层返回空）
// zero_grad() 有默认实现（按 parameters() 逐个清），一般不用改写。
//
// 训练 / 推理模式：默认就是训练模式，用 train() / eval() 切换。
// 需要区分的层（Dropout）在前向里读 is_training()；有子层的层（Sequential）
// 重写 set_training() 把模式递归转发下去。
#include <vector>

#include "../tensor/Tensor.hpp"

class Module
{
public:
    virtual ~Module() = default;

    // 前向。故意**不是 const**：Dropout / BatchNorm 这类层在前向时就要改自身状态
    virtual Tensor forward(const Tensor& x) = 0;

    // 本层的可训练参数指针（交给优化器）。无参层用基类的默认实现返回空
    virtual std::vector<Tensor*> parameters() { return {}; }

    // 清掉本层所有参数的梯度槽
    virtual void zero_grad();

    // ---------------- 训练 / 推理模式 ----------------
    // 切模式，并递归通知子层（有子层的层重写 set_training）
    void train(bool mode = true)
    {
        training_ = mode;
        set_training(mode);
    }
    void eval() { train(false); }
    bool is_training() const { return training_; }

    // 让层可以像函数一样用：net(x) 等价于 net.forward(x)
    Tensor operator()(const Tensor& x) { return forward(x); }

protected:
    // 有子层的层重写它来转发模式；无子层的层不用管
    virtual void set_training(bool mode) { (void)mode; }

private:
    bool training_ = true;   // 和 PyTorch 一样，默认是训练模式
};
