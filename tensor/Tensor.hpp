#pragma once
// mini-pytorch 的 Tensor。
// 类内只保留：声明 + 数据成员 + 必须留在头文件里的模板实现。
// 各模块的实现分别在 TensorCore / TensorFactory / TensorArith / TensorReduce /
// TensorShape / TensorUnary / TensorLoss / TensorAutograd / TensorUtil.cpp 里。
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <cmath>
#include <memory>
#include <functional>

#include "TensorGradMode.hpp"         // is_grad_enabled()：no_grad 作用域里不建图
#include "detail/TensorDetail.hpp"    // broadcast_shape / strides_for / plan_matmul
#include "detail/AutogradMeta.hpp"    // tensor_detail::AutogradMeta（构造时必须完整）

// 类体只留声明；实现分布在各模块 .cpp 里。
// 必须留在头文件里的只有：数据成员、内联访问器、以及模板（模板定义必须对使用点可见）。

// 方差 / 标准差的修正方式（对应 PyTorch 的 unbiased 参数）
enum class VarCorrection
{
    unbiased,   // 除以 (n-1)：PyTorch 的默认
    biased      // 除以 n：LayerNorm 内部用的就是这种
};
class Tensor
{
public:
    bool requires_grad = false;           // 叶子标记；要在参与运算之前设置

    // ---------------- 自动微分接口 ----------------
    bool is_leaf() const { return meta_->grad_fn == nullptr; }
    const Tensor* grad() const { return meta_->grad.get(); }   // 还没反向时为 nullptr
    void zero_grad() { meta_->grad.reset(); }
    void backward();

    // 脱离计算图的副本：数据相同，但 requires_grad = false、没有反向节点
    Tensor detach() const;

    // ---------------- 就地更新（只给优化器用） ----------------
    // this -= scale * other，直接改 data_、不建图。
    // 这是全库唯一破坏"值语义 / 无 in-place"约定的地方：优化器必须能就地改参数。
    // 因为图里保存的是输入的**副本**，就地改参数不会污染已经建好的计算图。
    void sub_inplace(const Tensor& other, float scale = 1.0f);

    Tensor(std::vector<float> data, std::vector<int64_t> shape);

    static Tensor zeros(std::vector<int64_t> shape);

    static Tensor ones(std::vector<int64_t> shape);

    static Tensor eye(int64_t n);

    static Tensor eye(int64_t n, int64_t m);

    static Tensor arange(float end);

    static Tensor arange(float start, float end, float step = 1.0f);

    // one-hot 编码：返回 [N, C]，其中 N = targets.size()
    // 第 i 行只有第 targets[i] 个元素是 1，其余是 0
    static Tensor one_hot(const std::vector<int64_t>& targets, int64_t num_classes);

    // 0 维单元素张量：用来和标量做广播运算（优化器里到处都要，比如 lr、beta）
    static Tensor scalar(float v);

    Tensor add(const Tensor& other) const;

    Tensor sub(const Tensor& other) const;

    Tensor mul(const Tensor& other) const;

    Tensor div(const Tensor& other) const;

    friend Tensor operator+(const Tensor& a, const Tensor& b) { return a.add(b); }
    friend Tensor operator-(const Tensor& a, const Tensor& b) { return a.sub(b); }
    friend Tensor operator*(const Tensor& a, const Tensor& b) { return a.mul(b); }
    friend Tensor operator/(const Tensor& a, const Tensor& b) { return a.div(b); }

    // ---------------- 掩码 / 条件（都不建图：它们本身不可导） ----------------
    // 逐元素比较，返回 1.0 / 0.0（支持广播）。要做掩码就用它们再配 where / mul。
    // 想和常数比：x.gt(Tensor::scalar(0.0f))
    Tensor gt(const Tensor& other) const;   // x >  other
    Tensor lt(const Tensor& other) const;   // x <  other
    Tensor ge(const Tensor& other) const;   // x >= other
    Tensor le(const Tensor& other) const;   // x <= other
    Tensor eq(const Tensor& other) const;   // x == other
    Tensor ne(const Tensor& other) const;   // x != other

    // 逐元素三选一：cond != 0 取 a，否则取 b（三个张量都按广播对齐）。
    // 可导：dL/da = g·mask，dL/db = g·(1-mask)；对 cond 不可导（和 PyTorch 一致）
    static Tensor where(const Tensor& cond, const Tensor& a, const Tensor& b);

    // 逐元素取符号：-1 / 0 / +1。不可导（导数恒为 0），所以不建图
    Tensor sign() const;

    // 逐元素夹到 [lo, hi] / [lo, +inf) / (-inf, hi]。可导：区间内 1，被夹住的部分 0
    Tensor clamp(float lo, float hi) const;
    Tensor clamp_min(float lo) const;
    Tensor clamp_max(float hi) const;

    Tensor matmul(const Tensor& other) const;

    Tensor neg() const;

    Tensor abs() const;

    // ---------------- 整体归约：把整个张量算成一个 0 维标量 ----------------
    // sum / prod 有空输入的单位元（0 / 1），max / min 没有，空张量只能报错（和 NumPy 一致）
    Tensor sum() const;

    Tensor prod() const;

    Tensor max() const;

    Tensor min() const;

    Tensor mean() const;

    Tensor sum(int64_t dim, bool keepdim = false) const;

    Tensor prod(int64_t dim, bool keepdim = false) const;

    Tensor max(int64_t dim, bool keepdim = false) const;

    Tensor min(int64_t dim, bool keepdim = false) const;

    Tensor mean(int64_t dim, bool keepdim = false) const;

    // ---------------- 指标类 ----------------
    // 下标张量：形状同归约结果，**值就是下标**（本库只有 float32，所以整数用 float 存）。
    // 不可导，也不建图（和 PyTorch 一样，argmax 本来就没有梯度）。
    // 并列时取第一个（按行优先顺序）。
    Tensor argmax() const;
    Tensor argmin() const;
    Tensor argmax(int64_t dim, bool keepdim = false) const;
    Tensor argmin(int64_t dim, bool keepdim = false) const;

    // 方差 / 标准差。用 mean/sub/mul 组合出来，所以反向是自动的。
    // 修正方式单独用强类型表示（不用 bool）：否则 x.var(1) 会在
    // "dim=1" 和 "unbiased=1" 两个重载之间歧义，而 x.var(false) 又会被
    // 静默当成 dim=0 —— 两种坑都躲开。
    Tensor var() const;                                            // 整体，无偏（PyTorch 默认）
    Tensor std() const;
    Tensor var(VarCorrection correction) const;                     // 整体，指定修正
    Tensor std(VarCorrection correction) const;
    Tensor var(int64_t dim, bool keepdim = false, VarCorrection correction = VarCorrection::unbiased) const;
    Tensor std(int64_t dim, bool keepdim = false, VarCorrection correction = VarCorrection::unbiased) const;

    // 单元素张量取值（标量或 [1]）；元素数不为 1 时抛异常。
    // 等价于 PyTorch 的 .item()，省得每处都写 data()[0]
    float item() const;

    Tensor reshape(std::vector<int64_t> new_shape) const;

    Tensor flatten() const;

    Tensor unsqueeze(int64_t dim) const;

    Tensor squeeze() const;

    Tensor squeeze(int64_t dim) const;

    Tensor concat(const Tensor& other, int64_t dim) const;

    Tensor transpose(int64_t dim0, int64_t dim1) const;

    // ---------------- 卷积用的 im2col ----------------
    // 把 [N, C, H, W] 按 kh×kw 的滑窗展开成 [N, K, L]：
    //     K = C*kh*kw（每个通道、每个卷积核位置占一行）
    //     L = OH*OW   （每个输出位置占一列，l = oh*OW + ow）
    // col[n][(c*kh+i)*kw+j][oh*OW+ow] = x[n][c][oh*stride+i-padding][ow*stride+j-padding]，
    // 越界位置取 pad_value（默认 0）—— 这就是填充，不必真把输入 pad 出来（省一块内存）。
    // 卷积用默认的 0（零填充）；**max 池化必须传 -inf**，否则全负的窗口会被 0 抢走最大值。
    // 布局和 PyTorch 的 F.unfold 一致（[N, C*kh*kw, L]），于是：
    //     weight[O, K].matmul(col[N, K, L]) -> [N, O, L] -> reshape -> [N, O, OH, OW]
    // 全程没有转置、没有多余拷贝。**有反向**：反向是 col2im，重叠的 patch 要相加。
    Tensor im2col(int64_t kh, int64_t kw, int64_t stride = 1, int64_t padding = 0,
                  float pad_value = 0.0f) const;

    // ---------------- 按下标取（gather）----------------
    // 沿 dim 按 indices 取：结果形状 = 把 shape_[dim] 换成 indices.size()，其余维不变。
    // 典型用法（Embedding 查表）：weight[V, D].index_select(0, ids) -> [ids.size(), D]
    // **有反向**：梯度按同样的下标累加回原张量 —— 同一行被选多次就累加多次，
    // 没被选中的行梯度为 0（和 PyTorch 的 embedding 反向一致）。
    Tensor index_select(int64_t dim, const std::vector<int64_t>& indices) const;

    Tensor exp() const;

    Tensor log() const;

    Tensor sqrt() const;

    Tensor pow(float p) const;

    Tensor sigmoid() const;

    Tensor ReLU() const;

    Tensor tanh() const;

    Tensor softmax(int64_t dim) const;

    Tensor log_softmax(int64_t dim) const;

    Tensor mse(const Tensor& other) const;

    // ---------------- 损失函数 ----------------

    // 交叉熵。target 是 one-hot 或软标签，和 logits 同形状 [B, C]
    // loss = mean_i( -sum_c target[i,c] * log_softmax(logits)[i,c] )
    // 顺序很重要：先 sum 掉类别维得到每个样本的 [B]，再对 batch 取 mean
    Tensor cross_entropy(const Tensor& targets) const;

    // 交叉熵：整数类别下标版本。targets 长度 B，logits 形状 [B, C]
    // 做法就是把下标转成 one-hot，再走上面的加权求和 —— 结果等价于取 -log_softmax[i, targets[i]]
    // 代价是中间的 one-hot 有 B*C 个 float（C 很大时费内存，这也是以后要上 gather 的原因）
    Tensor cross_entropy(const std::vector<int64_t>& targets) const;

    // 二元交叉熵（在 logits 上算，数值最稳）
    // loss = mean( max(x,0) - x*y + log(1 + exp(-|x|)) )
    // 由 -( y*log σ(x) + (1-y)*log(1-σ(x)) ) 化简而来（softplus(x) - y*x）
    Tensor bce_with_logits(const Tensor& targets) const;

    // 二元交叉熵（输入已经是概率 p）
    // loss = mean( -( y*log(p) + (1-y)*log(1-p) ) )
    Tensor bce(const Tensor& targets) const;

    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<float>& data() const { return data_; }

private:
    // 反向下文是"算子的反向怎么写"的接口，它要调用下面几个私有工具
    friend struct tensor_detail::BackwardCtx;

    std::vector<float> data_;
    std::vector<int64_t> shape_;
    std::shared_ptr<tensor_detail::AutogradMeta> meta_;   // 自动微分元信息（拷贝时共享）

    // ---------------- 自动微分的内部工具 ----------------
    // 反向公式由算子自己传进来（每个算子的反向都不同，不能在这里 if/else 判算子）
    template <typename Backward>
    static void attach_grad(Tensor& out, std::vector<Tensor> inputs, Backward backward);
    static void accumulate_grad(const Tensor& t, const Tensor& g);
    static Tensor reduce_to(const Tensor& g, const std::vector<int64_t>& target);
    static Tensor expand_like(const Tensor& g, const std::vector<int64_t>& target);
    static void collect_topo(const std::shared_ptr<tensor_detail::Node>& n,
                             std::vector<std::shared_ptr<tensor_detail::Node>>& order);

    static int64_t check_dim(int64_t dim, int64_t ndim, const char* op);

    // 沿 dim 取极值下标（argmax / argmin 共用；want_max 决定取大还是取小）
    static Tensor arg_reduce_impl(const Tensor& t, int64_t dim, bool keepdim, bool want_max, const char* op);

    // im2col 的反向：把 dcol[N, K, L] 累加回 [N, C, H, W]。
    // 重叠的 patch 相加；越界的贡献直接丢掉（等价于零填充处梯度为 0）
    static Tensor col2im(const Tensor& dcol, const std::vector<int64_t>& x_shape,
                         int64_t kh, int64_t kw, int64_t stride, int64_t padding);

    // index_select 的反向：把 dout 按下标累加回 in_shape（同一位置被选多次就累加多次）
    static Tensor index_add_grad(const Tensor& dout, const std::vector<int64_t>& indices,
                                 const std::vector<int64_t>& in_shape, int64_t dim);

    // 沿 dim 取 [start, start+count) 这一段（concat 的反向要沿拼接维切回去）
    static Tensor slice_dim(const Tensor& src, int64_t dim, int64_t start, int64_t count);

    // 行优先多维下标 +1（末维进位）；批次遍历和 transpose 共用
    static void next_index(std::vector<int64_t>& idx, const std::vector<int64_t>& shape);

    struct OuterInner
    {
        int64_t outer;
        int64_t inner;
    };

    OuterInner outer_inner(int64_t dim) const;

    template<typename BinaryOp>
    static Tensor binary_element(const Tensor& a, const Tensor& b, BinaryOp op)
    {
        using namespace tensor_detail;

        if (a.shape_ == b.shape_)
        {
            const int64_t total = numel(a.shape_);
            std::vector<float> out(static_cast<size_t>(total));
            for (int64_t i = 0; i < total; ++i)
            {
                out[static_cast<size_t>(i)] = op(a.data_[static_cast<size_t>(i)],
                                                b.data_[static_cast<size_t>(i)]);
            }
            return Tensor(std::move(out), a.shape_);
        }

        std::vector<int64_t> out_shape = broadcast_shape(a.shape_, b.shape_);
        const std::vector<int64_t> a_stride = strides_for(a.shape_, out_shape);
        const std::vector<int64_t> b_stride = strides_for(b.shape_, out_shape);

        const int64_t total = numel(out_shape);
        std::vector<float> out(static_cast<size_t>(total));

        std::vector<int64_t> idx(out_shape.size(), 0);
        int64_t a_off = 0, b_off = 0;

        for (int64_t t = 0; t < total; ++t)
        {
            out[t] = op(a.data_[a_off], b.data_[b_off]);

            for (size_t d = out_shape.size(); d-- > 0; )
            {
                ++idx[d];
                a_off += a_stride[d];
                b_off += b_stride[d];
                if (idx[d] < out_shape[d])
                {
                    break;
                }
                idx[d] = 0;
                a_off -= a_stride[d] * out_shape[d];
                b_off -= b_stride[d] * out_shape[d];
            }
        }
        return Tensor(std::move(out), out_shape);
    }

    // 逐元素 log 并截到 >= -100：PyTorch 的 BCELoss 也这么做，
    // 避免 p 恰好是 0 或 1 时 log 出 -inf，再乘 0 变成 nan
    static Tensor log_clamped(const Tensor& t);

    // 整体归约的公共实现（一）：有空输入时用的单位元（sum 用 0，prod 用 1）
    template <typename Accumulate>
    Tensor reduce_all_identity(Accumulate acc, float identity) const
    {
        float cur = identity;
        for (size_t i = 0; i < data_.size(); ++i)
        {
            cur = acc(cur, data_[i]);
        }
        return scalar(cur);
    }

    // 整体归约的公共实现（二）：没有单位元，空张量直接报错
    template <typename Accumulate>
    Tensor reduce_all_nonempty(Accumulate acc, const char* op) const
    {
        if (data_.empty())
        {
            throw std::runtime_error(std::string(op) + ": empty tensor has no identity");
        }
        float cur = data_[0];
        for (size_t i = 1; i < data_.size(); ++i)
        {
            cur = acc(cur, data_[i]);
        }
        return scalar(cur);
    }

    template <typename Accumulate>
    Tensor reduce_dim(int64_t dim, Accumulate acc, bool keepdim) const
    {
        const int64_t ndim = static_cast<int64_t>(shape_.size());
        if (ndim == 0)
        {
            throw std::runtime_error("reduce_dim: scalar has no dim");
        }
        dim = check_dim(dim, ndim, "reduce_dim");

        std::vector<int64_t> out_shape;
        out_shape.reserve(static_cast<size_t>(ndim) - 1 + (keepdim ? 1 : 0));
        for (int64_t i = 0; i < ndim; ++i)
        {
            if (i == dim)
            {
                if (keepdim)
                {
                    out_shape.push_back(1);
                }
            }
            else
            {
                out_shape.push_back(shape_[i]);
            }
        }

        const int64_t dim_size = shape_[dim];
        if (dim_size == 0)
        {
            throw std::runtime_error("reduce_dim: empty dim");
        }

        const OuterInner seg = outer_inner(dim);
        std::vector<float> out(static_cast<size_t>(seg.outer * seg.inner));

        for (int64_t o = 0; o < seg.outer; ++o)
        {
            for (int64_t i = 0; i < seg.inner; ++i)
            {
                float cur = data_[o * dim_size * seg.inner + i];

                for (int64_t k = 1; k < dim_size; ++k)
                {
                    cur = acc(cur, data_[o * dim_size * seg.inner + k * seg.inner + i]);
                }
                out[o * seg.inner + i] = cur;
            }
        }
        return Tensor(std::move(out), out_shape);
    }

    static int64_t numel(const std::vector<int64_t>& shape);

    template <typename ActivateOp>
    static Tensor unary_elementwise(const Tensor& a, ActivateOp op)
    {
        const int64_t total = numel(a.shape_);
        std::vector<float> out(static_cast<size_t>(total));
        for (int64_t i = 0; i < total; ++i)
        {
            out[i] = op(a.data_[i]);
        }
        return Tensor(std::move(out), a.shape_);
    }
};

// ==================== 自动微分的实现 ====================
// Node 里按值存 Tensor，所以 detail/AutogradNode.hpp 只能在 class 之后包含；
// 而 Tensor 里只存 shared_ptr<Node>（允许不完整类型），两边不会死锁。
#include "detail/AutogradNode.hpp"

// attach_grad 是模板（要接住算子传来的 lambda），定义必须留在头文件里。
// 其余自动微分的实现都在 TensorAutograd.cpp。
// 需要梯度时给输出挂上反向节点；反向公式由调用者（算子）传进来
template <typename Backward>
inline void Tensor::attach_grad(Tensor& out, std::vector<Tensor> inputs, Backward backward)
{
    if (!is_grad_enabled())
    {
        return;   // no_grad 作用域：不建图（输出因此自然保持 requires_grad = false）
    }

    bool need_grad = false;
    for (const Tensor& in : inputs)
    {
        if (in.requires_grad)
        {
            need_grad = true;
            break;
        }
    }
    if (!need_grad)
    {
        return;   // 输入都不需要梯度 -> 不建图（和 PyTorch 一致）
    }

    out.requires_grad = true;
    out.meta_->grad_fn = std::make_shared<tensor_detail::Node>();
    out.meta_->grad_fn->inputs = std::move(inputs);
    out.meta_->grad_fn->out_meta = out.meta_;
    out.meta_->grad_fn->backward = std::move(backward);
}

// 梯度累加（不是覆盖）：第一次访问时初始化成 0
// 注意 t 是 const 引用也没关系：改的是它 meta_ 指向的那份共享元信息
// 把梯度 g 归约回 target 形状：广播算子的反向要用
// 多出来的前导维直接 sum 掉；target 里是 1 而 g 里 > 1 的维，sum 掉但保留长度 1
// 把梯度 g "广播"到 target 形状：归约算子的反向要用（ones(target) * g）
// 后序 DFS：输入在前、自己最后；反转后就是反向传播要的顺序
// 反向传播：播种 d(loss)/d(loss) = 1，然后按逆拓扑序逐节点回传
