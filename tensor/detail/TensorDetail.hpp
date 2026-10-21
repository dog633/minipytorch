#pragma once
// Tensor 的内部工具：广播形状推导、步长计算、matmul 的形状规划。
// 这些函数会跨多个 .cpp 使用，所以必须是 inline（头文件里的普通函数不加 inline
// 会被每个包含它的翻译单元各定义一份，导致 multiple definition 链接错误）。
#include <vector>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

namespace tensor_detail
{
    // floor 除法（Python // 的语义）。C++ 的整数除法是向零截断的，
    // 分子为负时两者结果不同：int64_t(-2)/5 == 0，而 -2 // 5 == -1。
    inline int64_t floor_div(int64_t a, int64_t b)
    {
        const int64_t q = a / b;
        return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
    }

    // 卷积 / 池化的输出边长：floor((in + 2*pad - k) / stride) + 1。
    // **必须用 floor 除法**：否则 "核比输入还大" 时会算出 1 而不是 <= 0
    // （例如 in=3, k=5, stride=5：(3-5)/5 在 C++ 里是 0 -> 输出 1，
    //   于是本该报错的非法配置会静默算出一块越界的窗口）。
    inline int64_t conv_output_size(int64_t in, int64_t k, int64_t stride, int64_t padding)
    {
        return floor_div(in + 2 * padding - k, stride) + 1;
    }

    // matmul 的"执行计划"：判定与执行分离，循环里不再做形状判断
    struct MatmulPlan
    {
        int64_t m, k, n;
        std::vector<int64_t> batch;               // 广播后的批次形状
        std::vector<int64_t> a_stride, b_stride;  // 批次步长，单位 = 矩阵块，广播维为 0
        std::vector<int64_t> out_shape;           // 结果形状
    };

    // 把 own 的连续布局步长右对齐到 out；自身为 1 或缺失的维 -> 0（即广播）
    inline std::vector<int64_t> strides_for(const std::vector<int64_t>& own,
                                            const std::vector<int64_t>& out)
    {
        std::vector<int64_t> s(out.size(), 0);
        int64_t acc = 1;
        for (size_t i = own.size(); i-- > 0; )
        {
            const size_t pos = out.size() - own.size() + i;
            s[pos] = (own[i] == 1) ? 0 : acc;
            acc *= own[i];
        }
        return s;
    }

    // NumPy 风格广播：从最后一维往前对齐，每对必须相等或其中一个为 1
    inline std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
                                                const std::vector<int64_t>& b)
    {
        const size_t n = std::max(a.size(), b.size());
        std::vector<int64_t> s(n, 1);
        for (size_t i = 0; i < n; ++i)
        {
            const int64_t ai = (i < a.size()) ? a[a.size() - 1 - i] : 1;
            const int64_t bi = (i < b.size()) ? b[b.size() - 1 - i] : 1;
            if (ai != bi && ai != 1 && bi != 1)
            {
                throw std::runtime_error("broadcast: shape mismatch");
            }
            s[n - 1 - i] = std::max(ai, bi);
        }
        return s;
    }

    // 判定两个形状能否做矩阵乘，并给出 m / k / n、批次形状、批次步长和结果形状
    inline MatmulPlan plan_matmul(const std::vector<int64_t>& a, const std::vector<int64_t>& b)
    {
        if (a.empty() || b.empty())
        {
            throw std::runtime_error("matmul: dim not exist");
        }
        std::vector<int64_t> as = a, bs = b;
        const bool a_1d = (as.size() == 1);
        const bool b_1d = (bs.size() == 1);
        if (a_1d)
        {
            as.insert(as.begin(), 1);
        }
        if (b_1d)
        {
            bs.push_back(1);
        }

        MatmulPlan p{};
        p.m = as[as.size() - 2];
        p.k = as[as.size() - 1];
        p.n = bs[bs.size() - 1];
        if (p.k != bs[bs.size() - 2])
        {
            throw std::runtime_error("matmul: inner dim mismatch");
        }
        const std::vector<int64_t> ab(as.begin(), as.end() - 2);
        const std::vector<int64_t> bb(bs.begin(), bs.end() - 2);
        p.batch = broadcast_shape(ab, bb);
        p.a_stride = strides_for(ab, p.batch);
        p.b_stride = strides_for(bb, p.batch);
        p.out_shape = p.batch;
        if (!a_1d)
        {
            p.out_shape.push_back(p.m);
        }
        if (!b_1d)
        {
            p.out_shape.push_back(p.n);
        }
        return p;
    }
}  // namespace tensor_detail
