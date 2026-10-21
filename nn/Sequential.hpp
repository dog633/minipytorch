#pragma once
// 顺序容器：把若干层首尾相接，forward 时依次调用（等价于 PyTorch 的 nn.Sequential）。
//
// 用法：
//     Sequential net(Linear(3, 8), ReLU(), Linear(8, 2));
//     Tensor y = net.forward(x);        // 或者 net(x)
//     SGD opt(net.parameters(), 0.1f);  // 容器会把所有层的参数汇总起来
//
// 注意 C++ 的 most vexing parse：只传**一个**临时对象时，
//     Sequential net(ReLU());      // ← 会被当成函数声明！
// 要么用花括号，要么多给一层，要么先起名字：
//     Sequential net{ReLU()};
//     Sequential net(ReLU(), Linear(2, 1));
//     ReLU act; Sequential net(act);
//
// 层以 shared_ptr 存放（左值拷贝、右值移动，都先用 make_shared 放到堆上）：
//   - 层的地址因此是稳定的，优化器从 parameters() 拿到的 Tensor* 不会失效；
//   - 想留着句柄查看训练后的参数，可以 add(std::make_shared<Linear>(...))。
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "Module.hpp"

class Sequential : public Module
{
public:
    // 像 PyTorch 一样一次塞进去：Sequential(Linear(3,8), ReLU(), Linear(8,2))
    template <typename... Layers>
    explicit Sequential(Layers&&... layers)
    {
        (add(std::forward<Layers>(layers)), ...);   // C++17 折叠表达式
    }

    // 加一层：只接受派生自 Module 的类型（否则会去实例化 make_shared<非层> 而报一堆错）
    template <typename Layer,
              typename = std::enable_if_t<std::is_base_of<Module, std::decay_t<Layer>>::value>>
    void add(Layer&& layer)
    {
        auto ptr = std::make_shared<std::decay_t<Layer>>(std::forward<Layer>(layer));
        ptr->train(is_training());   // 新层跟随容器当前的训练/推理模式
        layers_.push_back(std::move(ptr));
    }

    // 直接给一个已经建好的层：容器与调用方共享所有权（适合留着句柄的场景）。
    // 注意重载：传 shared_ptr<Linear> 时走这个（派生类指针能隐式转成基类指针），
    // 不会跑去实例化上面那个模板 —— 上面的 enable_if 把这种情况挡掉了。
    void add(std::shared_ptr<Module> layer)
    {
        layer->train(is_training());
        layers_.push_back(std::move(layer));
    }

    // 依次前向；空容器就直接把输入原样返回
    Tensor forward(const Tensor& x) override;

    // 汇总所有层的参数（顺序 = 层顺序，层内顺序 = 该层 parameters() 的顺序）
    std::vector<Tensor*> parameters() override;

    std::size_t size() const { return layers_.size(); }
    Module& operator[](std::size_t i) { return *layers_[i]; }
    const Module& operator[](std::size_t i) const { return *layers_[i]; }

protected:
    // 训练/推理模式要往下传（否则嵌在里面的 Dropout 不知道该怎么办）
    void set_training(bool mode) override;

private:
    std::vector<std::shared_ptr<Module>> layers_;
};
