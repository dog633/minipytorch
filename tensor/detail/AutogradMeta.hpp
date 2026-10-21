#pragma once
// Tensor 的自动微分元信息。
// 和 detail/TensorDetail.hpp 一样，实现细节统一放在 detail/ 目录下，Tensor.hpp 里
// 不再出现 namespace tensor_detail 的定义。
//
// 本文件可以被 Tensor.hpp 在定义 class Tensor **之前** 包含：它只用到
// shared_ptr<Tensor>（允许不完整类型），不需要完整的 Tensor。
#include <memory>

class Tensor;   // 前置声明

namespace tensor_detail
{
    struct Node;          // 反向节点，定义在 detail/AutogradNode.hpp（需要完整的 Tensor）
    struct BackwardCtx;   // 反向下文，同上（算子写反向时通过它取输入值、写回梯度）

    // 每个 Tensor 在构造时创建一份。Tensor 拷贝时这个 shared_ptr 被一起复制，
    // 于是"同一个变量"的所有副本共用一份梯度槽 —— 值语义下这是梯度能被看见的关键。
    struct AutogradMeta
    {
        std::shared_ptr<Node> grad_fn;    // 产生这个张量的反向节点；叶子为 null
        std::shared_ptr<Tensor> grad;     // 梯度（反向时懒创建）
    };
}  // namespace tensor_detail
