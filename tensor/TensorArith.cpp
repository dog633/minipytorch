#include "Tensor.hpp"

// 逐元素算术与矩阵乘法（声明在 Tensor.hpp，实现按模块拆到这里）
//
// 反向的写法：给 attach_grad 传一个 lambda，参数是 tensor_detail::BackwardCtx，
// 里面已经准备好了输入值（c.x(i)，detach 过）和上游梯度（c.g，detach 过），
// 算完用 c.grad(i, v) 写回去 —— 它会自动把 v 的形状归约回第 i 个输入的形状
// （所以被广播过的输入不用自己处理）。归约类算子用 c.grad_expand。

Tensor Tensor::add(const Tensor& other) const
{
    Tensor out = binary_element(*this, other, [](float x, float y) { return x + y; });
    // 反向：da = g，db = g
    attach_grad(out, {*this, other}, [](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, c.g);
        c.grad(1, c.g);
    });
    return out;
}

Tensor Tensor::sub(const Tensor& other) const
{
    Tensor out = binary_element(*this, other, [](float x, float y) { return x - y; });
    // 反向：da = g，db = -g
    attach_grad(out, {*this, other}, [](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, c.g);
        c.grad(1, c.g.neg());
    });
    return out;
}

Tensor Tensor::mul(const Tensor& other) const
{
    Tensor out = binary_element(*this, other, [](float x, float y) { return x * y; });
    // 反向：da = g * b，db = g * a（要用前向的输入值，从 c.x 取）
    attach_grad(out, {*this, other}, [](const tensor_detail::BackwardCtx& c)
    {
        c.grad(0, c.g.mul(c.x(1)));
        c.grad(1, c.g.mul(c.x(0)));
    });
    return out;
}

Tensor Tensor::div(const Tensor& other) const
{
    Tensor out = binary_element(*this, other, [](float x, float y) { return x / y; });
    // 反向：da = g / b，db = -g * a / b²
    attach_grad(out, {*this, other}, [](const tensor_detail::BackwardCtx& c)
    {
        const Tensor a = c.x(0);
        const Tensor b = c.x(1);
        c.grad(0, c.g.div(b));
        c.grad(1, c.g.mul(a).div(b.mul(b)).neg());
    });
    return out;
}

// ---------------- 掩码 / 条件 ----------------
// 比较算子返回 1.0 / 0.0，支持广播。它们本身是阶跃函数、不可导，所以不建图；
// 但用它们做出来的掩码可以喂给 where / mul，梯度照样流得动。

Tensor Tensor::gt(const Tensor& other) const
{
    return binary_element(*this, other, [](float a, float b) { return a > b ? 1.0f : 0.0f; });
}

Tensor Tensor::lt(const Tensor& other) const
{
    return binary_element(*this, other, [](float a, float b) { return a < b ? 1.0f : 0.0f; });
}

Tensor Tensor::ge(const Tensor& other) const
{
    return binary_element(*this, other, [](float a, float b) { return a >= b ? 1.0f : 0.0f; });
}

Tensor Tensor::le(const Tensor& other) const
{
    return binary_element(*this, other, [](float a, float b) { return a <= b ? 1.0f : 0.0f; });
}

Tensor Tensor::eq(const Tensor& other) const
{
    return binary_element(*this, other, [](float a, float b) { return a == b ? 1.0f : 0.0f; });
}

Tensor Tensor::ne(const Tensor& other) const
{
    return binary_element(*this, other, [](float a, float b) { return a != b ? 1.0f : 0.0f; });
}

Tensor Tensor::where(const Tensor& cond, const Tensor& a, const Tensor& b)
{
    // 先把 cond 归一成严格 0/1 的掩码：cond = 0.5 时也应该是"选 a"，
    // 而不是插值出 0.5a + 0.5b（那样就与 PyTorch 的 where 语义不一致了）。
    // detach 让梯度不会流回条件张量（PyTorch 同样只对 a / b 求导）。
    const Tensor mask = cond.detach().ne(scalar(0.0f));
    const Tensor inv = mask.neg().add(scalar(1.0f));   // 1 - mask

    // a·mask + b·(1-mask)：反向自动得到 da = g·mask、db = g·(1-mask)
    return a.mul(mask).add(b.mul(inv));
}

Tensor Tensor::matmul(const Tensor& other) const
{
    using namespace tensor_detail;

    MatmulPlan p = plan_matmul(shape_, other.shape_);
    const int64_t batches = numel(p.batch);
    const int64_t out_mat = p.m * p.n;
    std::vector<float> out(static_cast<size_t>(batches * out_mat));

    std::vector<int64_t> idx(p.batch.size(), 0);

    for (int64_t t = 0; t < batches; ++t)
    {
        // 由批次下标算出这一批用 a / b 的第几个矩阵（广播维步长为 0，自动复用）
        int64_t ba = 0;
        int64_t bb = 0;
        for (size_t d = 0; d < p.batch.size(); ++d)
        {
            ba += idx[d] * p.a_stride[d];
            bb += idx[d] * p.b_stride[d];
        }

        const float* A = data_.data() + ba * p.m * p.k;
        const float* B = other.data_.data() + bb * p.k * p.n;
        float* O = out.data() + t * out_mat;

        for (int64_t i = 0; i < p.m; ++i)
        {
            for (int64_t kk = 0; kk < p.k; ++kk)
            {
                const float aik = A[i * p.k + kk];
                for (int64_t j = 0; j < p.n; ++j)
                {
                    O[i * p.n + j] += aik * B[kk * p.n + j];
                }
            }
        }

        next_index(idx, p.batch);
    }
    Tensor res(std::move(out), p.out_shape);

    // ---------------- 反向 ----------------
    // 在"提升后的矩阵空间"里算：A:[...,m,k]，B:[...,k,n]，C:[...,m,n]
    //   dA = g @ Bᵀ，dB = Aᵀ @ g
    // 两件收尾：批次维被广播过的要 sum 回去；1 维操作数提升出来的那一维要压掉。
    const std::vector<int64_t> as0 = shape_;
    const std::vector<int64_t> bs0 = other.shape_;
    attach_grad(res, {*this, other},
                [p, as0, bs0](const tensor_detail::BackwardCtx& c)
    {
        const bool a_1d = (as0.size() == 1);
        const bool b_1d = (bs0.size() == 1);

        Tensor Ap = c.x(0);
        Tensor Bp = c.x(1);
        if (a_1d)
        {
            Ap = Ap.unsqueeze(0);      // [k] -> [1, k]
        }
        if (b_1d)
        {
            Bp = Bp.unsqueeze(-1);     // [k] -> [k, 1]
        }

        // g 的形状可能缺了 m 或 n 那一维（1 维操作数在结果里被压掉了），先 reshape 回 [..., m, n]
        std::vector<int64_t> g_shape = p.batch;
        g_shape.push_back(p.m);
        g_shape.push_back(p.n);
        const Tensor gP = c.g.reshape(g_shape);

        const Tensor dAp = gP.matmul(Bp.transpose(-2, -1));   // [..., m, k]
        const Tensor dBp = Ap.transpose(-2, -1).matmul(gP);   // [..., k, n]

        // 批次广播出来的那几维要被 sum 掉，再用 reshape 把提升出来的 m/n 压回去
        c.grad(0, reduce_to(dAp, Ap.shape_).reshape(as0));
        c.grad(1, reduce_to(dBp, Bp.shape_).reshape(bs0));
    });
    return res;
}
