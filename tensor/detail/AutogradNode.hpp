#pragma once
// 反向节点 + 反向下文。
//
// **只能在 class Tensor 定义之后包含**：两者都用到完整的 Tensor
// （Node 里按值存 std::vector<Tensor>；BackwardCtx 里按值存上游梯度）。
// 而 Tensor 里只存 shared_ptr<Node>（允许不完整类型），所以两边不会死锁。
#include <functional>
#include <memory>
#include <vector>

#include "AutogradMeta.hpp"

class Tensor;

namespace tensor_detail
{
    // 反向下文：把每个算子反向里原本都要写一遍的三件事收进来 ——
    //   1) 取输入的值（并 detach，避免反向过程里又建图）
    //   2) 取上游梯度（已 detach）
    //   3) 把算好的梯度累加回某个输入（自动做广播归约）
    // 于是算子的反向只需要写"局部导数是什么"。
    struct BackwardCtx
    {
        const std::vector<Tensor>* inputs;   // 前向输入（活的：梯度要写进它们共享的 meta_）
        Tensor g;                            // 上游梯度，已 detach

        // 第 i 个输入的值（detach 过的副本，可直接参与反向计算而不会再次建图）
        Tensor x(size_t i) const { return (*inputs)[i].detach(); }

        // 把 v 当作第 i 个输入的梯度累加进去；v 的形状会自动归约回该输入的形状
        // （元素级算子的输入可能被广播过，比如 {2,3} + {3}）
        void grad(size_t i, const Tensor& v) const
        {
            Tensor::accumulate_grad((*inputs)[i], Tensor::reduce_to(v, (*inputs)[i].shape_));
        }

        // 归约类算子用：v 的形状比输入小，先把 v 广播到输入形状再累加
        void grad_expand(size_t i, const Tensor& v) const
        {
            Tensor::accumulate_grad((*inputs)[i], Tensor::expand_like(v, (*inputs)[i].shape_));
        }
    };

    // 一个反向节点 = "前向的输入是什么" + "怎么算它们的梯度" + "本节点输出的元信息"
    struct Node
    {
        std::vector<Tensor> inputs;   // 前向输入；它们的 meta_ 与原张量共享，所以梯度写进去能被看见
        std::shared_ptr<AutogradMeta> out_meta;   // 本节点产生的那个张量的元信息（取上游梯度用）
        std::function<void(const BackwardCtx&)> backward;
        bool visited = false;         // 逆拓扑遍历用
    };
}  // namespace tensor_detail
