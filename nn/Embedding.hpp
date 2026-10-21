#pragma once
// 查表层（Embedding）：把整数下标映射成向量，等价于 PyTorch 的 nn.Embedding。
//
//     weight[V, D]  +  ids（任意形状的整数下标）  ->  ids 的形状 + [D]
//
// 本体是一次 index_select(0, ids)（张量库里的 gather，自带 scatter-add 反向），
// 所以**这个层也没有反向代码**：只有被查到的行会拿到梯度，同一行被查多次就累加。
//
// 关于下标的类型：本库只有 float32 张量，所以下标用 float 存整数值
// （和 argmax 的返回值是同一种约定）。因此 forward 接的是"值必须是整数的张量"，
// 非整数会抛异常；另外也提供了直接吃 vector<int64_t> 的 lookup()。
#include <cstdint>
#include <vector>

#include "Module.hpp"

class Embedding : public Module
{
public:
    // 权重按 U(-1, 1) 初始化（PyTorch 用的是 N(0,1)；这里没有正态 RNG，改用均匀分布）
    Embedding(int64_t num_embeddings, int64_t embedding_dim, uint32_t seed = 12345u);

    // ids 的形状随意：[L] -> [L, D]，[N, L] -> [N, L, D]
    Tensor forward(const Tensor& ids) override;

    // 直接用整数下标（不经过张量）
    Tensor lookup(const std::vector<int64_t>& ids) const;

    std::vector<Tensor*> parameters() override;

    Tensor weight;   // [num_embeddings, embedding_dim]

    int64_t num_embeddings() const { return num_embeddings_; }
    int64_t embedding_dim() const { return embedding_dim_; }

private:
    int64_t num_embeddings_;
    int64_t embedding_dim_;
};
