// 入口文件：演示整个库 —— 张量算子、autograd、损失、nn 层、优化器
#include "nn/BatchNorm.hpp"
#include "nn/AvgPool2d.hpp"
#include "nn/Conv2d.hpp"
#include "nn/CrossEntropyLoss.hpp"
#include "nn/Embedding.hpp"
#include "nn/Dropout.hpp"
#include "nn/Flatten.hpp"
#include "nn/GELU.hpp"
#include "nn/LayerNorm.hpp"
#include "nn/Linear.hpp"
#include "nn/MaxPool2d.hpp"
#include "nn/ReLU.hpp"
#include "nn/Sequential.hpp"
#include "nn/SiLU.hpp"
#include "optimizer/Adam.hpp"
#include "optimizer/SGD.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/TensorGradMode.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void show(const char* tag, const Tensor& t)
{
    std::printf("%-22s shape=[", tag);
    for (size_t i = 0; i < t.shape().size(); ++i)
    {
        std::printf(i ? ",%lld" : "%lld", static_cast<long long>(t.shape()[i]));
    }
    std::printf("]  data=[");
    for (size_t i = 0; i < t.data().size(); ++i)
    {
        std::printf(i ? ", %g" : "%g", static_cast<double>(t.data()[i]));
    }
    std::printf("]\n");
}

int main()
{
    const Tensor a({1, 2, 3, 4, 5, 6}, {2, 3});   // [2,3]
    const Tensor b({10, 20, 30}, {3});            // [3]，用于广播

    std::printf("=== elementwise (broadcast) ===\n");
    show("a", a);
    show("b", b);
    show("a + b", a + b);
    show("b + a", b + a);        // 左操作数更小时同样成立
    show("a * b", a * b);
    show("a / b", a / b);
    show("-a", a.neg());
    show("a - a", a - a);

    std::printf("\n=== matmul ===\n");
    const Tensor x({1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor w({1, 0, 0, 1, 1, 1}, {3, 2});
    show("x @ w", x.matmul(w));                  // [2,3] @ [3,2] -> [2,2]
    show("x @ v   (v is [3])", x.matmul(Tensor({1, 1, 1}, {3})));  // [2,3] @ [3] -> [2]

    std::printf("\n=== unary / shape ops ===\n");
    show("a - 4", a - Tensor({4, 4, 4}, {3}));
    show("ReLU(a - 4)", (a - Tensor({4, 4, 4}, {3})).ReLU());
    show("t  = [1,3,1]", Tensor({1, 2, 3}, {1, 3, 1}));
    show("t.squeeze(0)", Tensor({1, 2, 3}, {1, 3, 1}).squeeze(0));
    show("t.squeeze(-1)", Tensor({1, 2, 3}, {1, 3, 1}).squeeze(-1));
    show("t.squeeze()", Tensor({1, 2, 3}, {1, 3, 1}).squeeze());
    show("v.unsqueeze(0)", Tensor({1, 2, 3}, {3}).unsqueeze(0));

    std::printf("\n=== whole-tensor reduce (0-dim scalar) ===\n");
    show("a.sum()", a.sum());
    show("a.prod()", a.prod());
    show("a.max()", a.max());
    show("a.min()", a.min());
    show("a.mean()", a.mean());
    show("a - a.mean()", a - a.mean());          // 0 维结果参与广播
    show("a / a.sum()", a / a.sum());

    std::printf("\n=== softmax ===\n");
    const Tensor logits({1, 2, 3, 1, 1, 1}, {2, 3});
    show("logits", logits);
    show("softmax(logits, 1)", logits.softmax(1));   // 每行和为 1
    show("softmax(logits, 0)", logits.softmax(0));   // 每列和为 1
    show("softmax([1000,1001,1002], 0)",
         Tensor({1000.0f, 1001.0f, 1002.0f}, {3}).softmax(0));  // 减去 max，不会 exp 上溢

    std::printf("\n=== factory ===\n");
    show("zeros({2,3})", Tensor::zeros({2, 3}));
    show("ones({3})", Tensor::ones({3}));
    show("eye(3)", Tensor::eye(3));
    show("eye(2,4)", Tensor::eye(2, 4));
    show("arange(5)", Tensor::arange(5.0f));
    show("arange(1,7,2)", Tensor::arange(1.0f, 7.0f, 2.0f));
    show("arange(5,0,-1)", Tensor::arange(5.0f, 0.0f, -1.0f));
    show("ones({2,2}) + eye(2)", Tensor::ones({2, 2}) + Tensor::eye(2));  // 工厂 + 广播
    show("one_hot({0,2}, 3)", Tensor::one_hot({0, 2}, 3));

    std::printf("\n=== losses ===\n");
    const Tensor lg({0.5f, -1.5f, 2.0f, 3.0f, 0.0f, -0.5f}, {2, 3});
    const Tensor onehot({1, 0, 0, 0, 0, 1}, {2, 3});          // 手工 one-hot：类别 0 和类别 2
    show("lg", lg);
    show("lg.log_softmax(1)", lg.log_softmax(1));             // exp 后每行和为 1
    show("lg.cross_entropy(onehot)", lg.cross_entropy(onehot));
    show("lg.cross_entropy({0,2})", lg.cross_entropy(std::vector<int64_t>{0, 2}));  // 整数下标版，应与上面相同
    show("lg.cross_entropy({1,1})", lg.cross_entropy(std::vector<int64_t>{1, 1}));

    const Tensor xb({-8, -3, 3, 8}, {2, 2});                  // logits
    const Tensor yb({1, 0, 0, 1}, {2, 2});                    // 0/1 标签
    show("xb.bce_with_logits(yb)", xb.bce_with_logits(yb));   // 推荐：在 logits 上算
    show("xb.sigmoid()", xb.sigmoid());
    show("xb.sigmoid().bce(yb)", xb.sigmoid().bce(yb));       // 两者应当很接近

    std::printf("\n=== training: 学一个线性回归 y = X @ w + b ===\n");
    {
        const Tensor X({0.1f, -0.8f, 0.5f,
                        0.6f, 0.2f, -0.7f,
                        -0.4f, 0.9f, 0.3f,
                        0.2f, -0.1f, 0.8f}, {4, 3});
        const Tensor w_true({0.5f, -1.0f, 2.0f}, {3});
        const Tensor b_true({0.3f}, {});
        const Tensor target = X.matmul(w_true).add(b_true);   // 训练目标

        // 从全 0 开始学
        Tensor w = Tensor::zeros({3});
        Tensor b = Tensor::zeros({});
        w.requires_grad = true;
        b.requires_grad = true;

        SGD sgd({&w, &b}, 0.05f, 0.9f);     // lr = 0.05，动量 0.9
        Adam adam({&w, &b}, 0.05f);         // 默认 betas=(0.9, 0.999), eps=1e-8

        for (int i = 0; i < 300; ++i)
        {
            sgd.zero_grad();
            Tensor loss = X.matmul(w).add(b).mse(target);
            loss.backward();
            sgd.step();
            if (i == 0 || i == 299)
            {
                std::printf("  SGD  step %-3d loss = %.8f\n", i, (double)loss.data()[0]);
            }
        }
        std::printf("  SGD  学到的 w = [%.5f, %.5f, %.5f]  b = %.5f   (真值 [0.5, -1, 2], 0.3)\n",
                    (double)w.data()[0], (double)w.data()[1], (double)w.data()[2], (double)b.data()[0]);

        w = Tensor::zeros({3});
        b = Tensor::zeros({});
        w.requires_grad = true;
        b.requires_grad = true;
        for (int i = 0; i < 800; ++i)
        {
            adam.zero_grad();
            Tensor loss = X.matmul(w).add(b).mse(target);
            loss.backward();
            adam.step();
            if (i == 0 || i == 799)
            {
                std::printf("  Adam step %-3d loss = %.8f\n", i, (double)loss.data()[0]);
            }
        }
        std::printf("  Adam 学到的 w = [%.5f, %.5f, %.5f]  b = %.5f\n",
                    (double)w.data()[0], (double)w.data()[1], (double)w.data()[2], (double)b.data()[0]);
    }

    std::printf("\n=== nn.Linear: y = x @ W + b ===\n");
    {
        const Tensor X({0.1f, -0.8f, 0.5f,
                        0.6f, 0.2f, -0.7f,
                        -0.4f, 0.9f, 0.3f,
                        0.2f, -0.1f, 0.8f}, {4, 3});
        const Tensor W_true({0.5f, -1.0f, 2.0f, 0.3f, -0.7f, 1.5f}, {3, 2});   // [in, out]
        const Tensor b_true({0.2f, -0.4f}, {2});
        const Tensor target = X.matmul(W_true).add(b_true);                    // [4, 2]

        Linear lin(3, 2);                       // 默认初始化（固定 seed，可复现）
        show("lin.weight", lin.weight);         // [in, out] = [3, 2]
        show("lin.bias", lin.bias);             // [out] = [2]
        show("lin.forward(X)", lin.forward(X)); // [4, 3] @ [3, 2] -> [4, 2]

        SGD opt(lin.parameters(), 0.1f, 0.9f);  // 直接要吃层里的参数指针
        for (int i = 0; i < 600; ++i)
        {
            opt.zero_grad();
            Tensor loss = lin.forward(X).mse(target);
            loss.backward();
            opt.step();
            if (i == 0 || i == 599)
            {
                std::printf("  step %-3d loss = %.8f\n", i, (double)loss.data()[0]);
            }
        }
        std::printf("  weight = [[%.5f, %.5f], [%.5f, %.5f], [%.5f, %.5f]]\n",
                    (double)lin.weight.data()[0], (double)lin.weight.data()[1],
                    (double)lin.weight.data()[2], (double)lin.weight.data()[3],
                    (double)lin.weight.data()[4], (double)lin.weight.data()[5]);
        std::printf("  真值   = [[0.50000, -1.00000], [2.00000, 0.30000], [-0.70000, 1.50000]]\n");
        std::printf("  bias   = [%.5f, %.5f]   真值 = [0.20000, -0.40000]\n",
                    (double)lin.bias.data()[0], (double)lin.bias.data()[1]);
    }

    std::printf("\n=== nn.Sequential + nn.ReLU: 解 XOR ===\n");
    {
        // XOR：4 个点，(0,0)/(1,1) -> 0，(0,1)/(1,0) -> 1（线性不可分）
        const Tensor X({0, 0, 0, 1, 1, 0, 1, 1}, {4, 2});
        const Tensor Y({0, 1, 1, 0}, {4, 1});

        Sequential net(Linear(2, 8), ReLU(), Linear(8, 1));   // 像搭积木一样串层
        std::printf("  层数 = %u，可训练参数张量 = %u 个\n",
                    (unsigned)net.size(), (unsigned)net.parameters().size());

        Adam opt(net.parameters(), 0.05f);                    // 容器汇总所有层的参数
        for (int i = 0; i < 2000; ++i)
        {
            opt.zero_grad();
            Tensor loss = net.forward(X).bce_with_logits(Y);
            loss.backward();
            opt.step();
            if (i == 0 || i == 1999)
            {
                std::printf("  step %-4d loss = %.6f\n", i, (double)loss.data()[0]);
            }
        }
        const Tensor out = net.forward(X);
        std::printf("  预测 logits = [%.3f, %.3f, %.3f, %.3f]  ->  %s\n",
                    (double)out.data()[0], (double)out.data()[1],
                    (double)out.data()[2], (double)out.data()[3],
                    (out.data()[0] < 0 && out.data()[1] > 0 && out.data()[2] > 0 && out.data()[3] < 0)
                        ? "4/4 全对" : "有错");

        // 反证：不带 ReLU 的纯线性模型学不了 XOR，loss 会停在 ln2 ≈ 0.693
        Sequential flat(Linear(2, 1));
        Adam opt2(flat.parameters(), 0.05f);
        float flat_loss = 0.0f;
        for (int i = 0; i < 2000; ++i)
        {
            opt2.zero_grad();
            Tensor loss = flat.forward(X).bce_with_logits(Y);
            loss.backward();
            opt2.step();
            flat_loss = loss.data()[0];
        }
        std::printf("  对照（只有 Linear）：loss = %.6f ≈ ln2 = %.6f，学不动\n",
                    (double)flat_loss, (double)std::log(2.0));
    }

    std::printf("\n=== 激活函数对比（x = -3 .. 3）===\n");
    {
        const Tensor t({-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f}, {7});
        show("ReLUctant(x)", ReLUctant(t));
        show("Silu(x)", Silu(t));
        show("Gelu(x) tanh 近似", Gelu(t));
    }

    std::printf("\n=== 训练/推理模式 + LayerNorm / Dropout ===\n");
    {
        Sequential net(Linear(2, 16), LayerNorm(16), GELU(), Dropout(0.1f), Linear(16, 1));
        std::printf("  默认 is_training = %s\n", net.is_training() ? "true" : "false");

        const Tensor probe = Tensor::ones({20, 50});   // 1000 个元素
        Dropout& dp = dynamic_cast<Dropout&>(net[3]);
        auto count_zeros = [](const Tensor& t)
        {
            int n = 0;
            for (float v : t.data())
            {
                n += (v == 0.0f) ? 1 : 0;
            }
            return n;
        };
        std::printf("  train 模式下 Dropout(0.1)：1000 个元素里 %d 个被置零，存活值放大到 %.4f\n",
                    count_zeros(dp.forward(probe)), (double)(1.0f / (1.0f - 0.1f)));
        net.eval();
        std::printf("  net.eval() 之后：置零 %d 个（模式也传到了子层：is_training = %s）\n",
                    count_zeros(dp.forward(probe)), dp.is_training() ? "true" : "false");
        net.train();

        // LayerNorm 干的事：沿最后一维标准化成均值 0、方差 1，再仿射
        LayerNorm demo_ln(4);                 // 只标准化最后一维
        const Tensor raw({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4});
        show("LayerNorm 输入", raw);
        show("LayerNorm 输出", demo_ln.forward(raw));

        // 用整套新层再解一次 XOR
        const Tensor X({0, 0, 0, 1, 1, 0, 1, 1}, {4, 2});
        const Tensor Y({0, 1, 1, 0}, {4, 1});
        Adam opt(net.parameters(), 0.03f);
        float last = 0.0f;
        for (int i = 0; i < 3000; ++i)
        {
            opt.zero_grad();
            Tensor loss = net.forward(X).bce_with_logits(Y);
            loss.backward();
            opt.step();
            last = loss.data()[0];
        }
        net.eval();                      // 推理模式：Dropout 关掉，结果确定
        const Tensor out = net.forward(X);
        std::printf("  Linear+LayerNorm+GELU+Dropout 训 3000 步：loss = %.8f\n", (double)last);
        std::printf("  eval 预测 logits = [%.2f, %.2f, %.2f, %.2f]  ->  %s\n",
                    (double)out.data()[0], (double)out.data()[1],
                    (double)out.data()[2], (double)out.data()[3],
                    (out.data()[0] < 0 && out.data()[1] > 0 && out.data()[2] > 0 && out.data()[3] < 0)
                        ? "4/4 全对" : "有错");
    }

    std::printf("\n=== no_grad：推理时关掉计算图记录 ===\n");
    {
        Sequential net(Linear(2, 8), ReLU(), Linear(8, 1));
        net.eval();                                  // eval 只管层的行为，不关 autograd
        const Tensor x({0.5f, -0.3f, 0.1f, 0.8f}, {2, 2});

        const Tensor y1 = net.forward(x);
        std::printf("  默认：is_grad_enabled=%s，输出 is_leaf=%s（还在建图）\n",
                    is_grad_enabled() ? "true" : "false", y1.is_leaf() ? "true" : "false");

        {
            NoGradGuard guard;                        // RAII：离开作用域自动恢复
            const Tensor y2 = net.forward(x);
            std::printf("  NoGradGuard 内：is_grad_enabled=%s，输出 is_leaf=%s，requires_grad=%s\n",
                        is_grad_enabled() ? "true" : "false",
                        y2.is_leaf() ? "true" : "false", y2.requires_grad ? "true" : "false");
            std::printf("  数值完全一样：%g vs %g\n", (double)y1.data()[0], (double)y2.data()[0]);
            try
            {
                y2.sum().backward();                  // no_grad 里没有图可反传
            }
            catch (const std::exception& e)
            {
                std::printf("  no_grad 里 backward() 会直接报错：%s\n", e.what());
            }
        }
        std::printf("  离开作用域：is_grad_enabled=%s（自动恢复）\n", is_grad_enabled() ? "true" : "false");

        // 训练照旧：no_grad 只影响建图，不影响优化器
        net.train();
        Adam opt(net.parameters(), 0.05f);
        const Tensor X({0, 0, 0, 1, 1, 0, 1, 1}, {4, 2});
        const Tensor Y({0, 1, 1, 0}, {4, 1});
        for (int i = 0; i < 1500; ++i)
        {
            opt.zero_grad();
            net.forward(X).bce_with_logits(Y).backward();
            opt.step();
        }
        net.eval();
        {
            NoGradGuard guard;                        // 推理：不建图，也不需要梯度
            const Tensor out = net.forward(X);
            std::printf("  训练后（no_grad 推理）logits = [%.2f, %.2f, %.2f, %.2f] -> %s\n",
                        (double)out.data()[0], (double)out.data()[1],
                        (double)out.data()[2], (double)out.data()[3],
                        (out.data()[0] < 0 && out.data()[1] > 0 && out.data()[2] > 0 && out.data()[3] < 0)
                            ? "4/4 全对" : "有错");
        }
    }

    std::printf("\n=== 指标与掩码：argmax / var / clamp / where / item ===\n");
    {
        // 小例子：逐元素看一遍
        const Tensor t({-2.0f, -0.5f, 0.0f, 0.5f, 3.0f, 10.0f}, {2, 3});
        show("t", t);
        show("t.clamp(-1, 1)", t.clamp(-1.0f, 1.0f));
        show("t.clamp_min(0)", t.clamp_min(0.0f));
        show("t.sign()", t.sign());
        show("t.gt(0)", t.gt(Tensor::scalar(0.0f)));
        show("where(t>0, t, 0)", Tensor::where(t.gt(Tensor::scalar(0.0f)), t, Tensor::zeros({2, 3})));
        std::printf("  t.argmax() = %g   t.argmin() = %g   t.var() = %.6f   t.std(biased) = %.6f\n",
                    (double)t.argmax().item(), (double)t.argmin().item(),
                    (double)t.var().item(), (double)t.std(VarCorrection::biased).item());
        std::printf("  t.argmax(1) = [%g, %g]（每行最大值的下标）\n",
                    (double)t.argmax(1).data()[0], (double)t.argmax(1).data()[1]);

        // 真正的用途：训练一个 3 分类器，然后用 argmax + eq + mean 算准确率
        const int64_t N = 15, C = 3;
        std::vector<float> xv, yv;
        std::vector<int64_t> labels;
        for (int64_t i = 0; i < N; ++i)
        {
            const int64_t cls = i % C;
            for (int64_t j = 0; j < C; ++j)
            {
                xv.push_back((j == cls ? 1.0f : 0.0f) + 0.1f * static_cast<float>((i * 5 + j) % 3));
            }
            yv.push_back(static_cast<float>(cls));
            labels.push_back(cls);
        }
        const Tensor X(xv, {N, C});
        const Tensor Yv(yv, {N});

        Sequential net(Linear(C, 12), ReLU(), Linear(12, C));
        Adam opt(net.parameters(), 0.05f);
        float ce = 0.0f;
        for (int i = 0; i < 600; ++i)
        {
            opt.zero_grad();
            Tensor loss = net.forward(X).cross_entropy(labels);
            loss.backward();
            opt.step();
            ce = loss.item();
        }
        const Tensor pred = net.forward(X).argmax(1);      // 预测类别（下标存在 float 张量里）
        const float acc = pred.eq(Yv).mean().item();       // 准确率
        std::printf("  3 分类：ce = %.6f，acc = %.4f（%g/%g 正确）\n",
                    (double)ce, (double)acc, (double)pred.eq(Yv).sum().item(), (double)N);
    }

    std::printf("\n=== nn.Conv2d（im2col 实现）===\n");
    {
        Conv2d conv(1, 4, 3, 1, 1);                      // in=1, out=4, k=3, stride=1, padding=1
        show("conv.weight", conv.weight);                // [O, C, k, k] = [4,1,3,3]
        show("conv.bias", conv.bias);                    // [4]

        const Tensor img({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                          0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                          0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
                          0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f,
                          0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f,
                          0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f}, {1, 1, 6, 6});
        show("img (N,C,H,W)", img);
        show("conv.forward(img)", conv.forward(img));    // [1,4,6,6]
        const Tensor col = img.im2col(3, 3, 1, 1);       // [N, C*k*k, L] = [1,9,36]
        std::printf("%-22s shape=[%lld,%lld,%lld]（N, C*k*k, L），col[0][0] 前 6 个 = [%g, %g, %g, %g, %g, %g]\n",
                    "img.im2col(3,3,1,1)", (long long)col.shape()[0], (long long)col.shape()[1],
                    (long long)col.shape()[2], (double)col.data()[0], (double)col.data()[1],
                    (double)col.data()[2], (double)col.data()[3], (double)col.data()[4],
                    (double)col.data()[5]);
        std::printf("  尺寸：6x6 -> %lldx%lld（k=3, s=1, p=1 保持高宽）\n",
                    (long long)conv.output_height(6), (long long)conv.output_width(6));

        // 端到端：识别竖条还是横条
        const int64_t N = 16, S = 6;
        std::vector<float> xv(static_cast<size_t>(N * S * S), 0.0f), yv(static_cast<size_t>(N), 0.0f);
        for (int64_t i = 0; i < N; ++i)
        {
            const bool vertical = (i % 2 == 0);
            const int64_t pos = 1 + (i / 2) % (S - 2);
            for (int64_t r = 0; r < S; ++r)
            {
                for (int64_t c = 0; c < S; ++c)
                {
                    const bool on = vertical ? (c == pos) : (r == pos);
                    xv[static_cast<size_t>((i * S + r) * S + c)] = on ? 1.0f : 0.0f;
                }
            }
            yv[static_cast<size_t>(i)] = vertical ? 1.0f : 0.0f;
        }
        const Tensor Xc(xv, {N, 1, S, S});
        const Tensor Ybar(yv, {N});

        Conv2d c2(1, 4, 3, 1, 1);
        BatchNorm bn(4);                                   // 放在卷积之后（卷积网的标准排法）
        ReLU act;
        MaxPool2d pool(2, 2);                              // [N,4,6,6] -> [N,4,3,3]
        Flatten flt;                                       // [N,4,3,3] -> [N,36]，不用手写 reshape
        Linear head(4 * 3 * 3, 1);                         // 池化后尺寸变小，全连接输入跟着变小
        std::vector<Tensor*> params = c2.parameters();     // 三层的参数一起交给优化器
        for (Tensor* q : bn.parameters()) { params.push_back(q); }
        for (Tensor* q : head.parameters()) { params.push_back(q); }
        Adam opt(params, 0.02f);
        std::printf("  BatchNorm 初始：running_mean = [%g, %g, ...]，running_var = [%g, %g, ...]\n",
                    (double)bn.running_mean.data()[0], (double)bn.running_mean.data()[1],
                    (double)bn.running_var.data()[0], (double)bn.running_var.data()[1]);
        float first = 0.0f, last = 0.0f;
        for (int step = 0; step < 400; ++step)
        {
            opt.zero_grad();
            Tensor flat = flt.forward(act.forward(pool.forward(act.forward(bn.forward(c2.forward(Xc))))));
            Tensor loss = head.forward(flat).reshape({N}).bce_with_logits(Ybar);
            loss.backward();
            opt.step();
            if (step == 0) { first = loss.item(); }
            last = loss.item();
        }
        std::printf("  训练 400 步后：running_mean = [%.4f, %.4f, %.4f, %.4f]（被滑动平均更新过）\n",
                    (double)bn.running_mean.data()[0], (double)bn.running_mean.data()[1],
                    (double)bn.running_mean.data()[2], (double)bn.running_mean.data()[3]);
        bn.eval();                                         // 推理：改用 running_mean / running_var
        const Tensor logits =
            head.forward(flt.forward(act.forward(pool.forward(act.forward(bn.forward(c2.forward(Xc))))))).reshape({N});
        const Tensor pred = logits.gt(Tensor::scalar(0.0f));
        std::printf("  Conv2d -> BatchNorm -> ReLU -> MaxPool2d -> ReLU -> Flatten -> Linear：loss %.4f -> %.6f，eval acc = %.4f（%g/%g）\n",
                    (double)first, (double)last, (double)pred.eq(Ybar).mean().item(),
                    (double)pred.eq(Ybar).sum().item(), (double)N);
    }

    std::printf("\n=== AvgPool2d / Embedding / 损失层 ===\n");
    {
        // 平均池化（和 MaxPool2d 同构，只是把 max 换成 mean）
        const Tensor img2({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, {1, 1, 4, 4});
        show("img2", img2);
        show("MaxPool2d(2,2)", MaxPool2d(2, 2).forward(img2));   // 每个 2x2 取最大
        show("AvgPool2d(2,2)", AvgPool2d(2, 2).forward(img2));   // 每个 2x2 取平均

        // 查表：weight[5,3] + 下标 -> 向量
        Embedding emb(5, 3);
        show("emb.weight", emb.weight);
        const Tensor ids({0, 2, 2}, {3});
        show("emb.forward([0,2,2])", emb.forward(ids));          // [3,3]

        // 损失层：能坐在 Sequential 末尾，前向直接得到标量损失
        Sequential net2(Linear(4, 8), ReLU(), Linear(8, 3));
        CrossEntropyLoss criterion({0, 2, 1});
        Adam opt2(net2.parameters(), 0.05f);
        const Tensor Xc2({0.5f, -1.0f, 2.0f, 0.3f, 1.5f, 0.2f, -0.7f, 0.9f,
                          -1.2f, 0.4f, 0.8f, -0.3f}, {3, 4});
        float first = 0.0f, last = 0.0f;
        for (int i = 0; i < 300; ++i)
        {
            opt2.zero_grad();
            Tensor loss = criterion.forward(net2.forward(Xc2));   // 网络 + 损失层一起前向
            loss.backward();
            opt2.step();
            if (i == 0) { first = loss.item(); }
            last = loss.item();
        }
        const Tensor pred = net2.forward(Xc2).argmax(1);
        std::printf("  损失层训练：loss %.5f -> %.6f，acc = %.4f（%g/3）\n",
                    (double)first, (double)last,
                    (double)pred.eq(Tensor({0, 2, 1}, {3})).mean().item(),
                    (double)pred.eq(Tensor({0, 2, 1}, {3})).sum().item());
    }
    std::printf("\n=== error handling ===\n");
    try
    {
        show("bad shape", a + Tensor({1, 2, 3, 4}, {2, 2}));
    }
    catch (const std::exception& e)
    {
        std::printf("a + {2,2} throws: %s\n", e.what());
    }

    return 0;
}
