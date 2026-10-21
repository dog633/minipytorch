"""pybind11 绑定的冒烟测试（在 dog 环境里跑）。

    D:\\ProgramData\\Anaconda3\\envs\\dog\\python.exe test_bindings.py

C++ 那边已经用有限差分验过每个算子的正确性，这里只验"绑定层有没有接错、有没有漏"：
数值经过 Python 走一圈后要和手算一致，autograd / 层 / 优化器 / no_grad 都要能用。
"""

import gc
import math

import minitorch as mt


FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  OK    {name:44s} {detail}")
    else:
        FAILS.append(name)
        print(f"  FAIL  {name:44s} {detail}")


def close(a, b, tol=1e-5):
    return abs(float(a) - float(b)) <= tol


def section(title):
    print(f"\n=== {title} ===")


def main():
    section("张量与运算")
    x = mt.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    check("构造 shape", x.shape == [2, 3])
    check("data 是展平列表", x.data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    check("len() = shape[0]", len(x) == 2)
    check("repr 能打印", "Tensor(shape=[2, 3]" in repr(x))
    check("zeros/ones", mt.Tensor.zeros([2, 2]).data == [0.0] * 4
          and mt.Tensor.ones([2, 2]).data == [1.0] * 4)
    check("eye", mt.Tensor.eye(2).data == [1.0, 0.0, 0.0, 1.0])
    check("arange", mt.Tensor.arange(3.0).data == [0.0, 1.0, 2.0])
    check("scalar + item", mt.Tensor.scalar(2.5).item() == 2.5)
    check("shape 省略就是标量", mt.Tensor([7.0]).shape == [])

    y = x + 1.0                      # 运算符 + 自动把数字变成标量张量
    check("x + 1.0", y.data == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    check("2.0 * x", (2.0 * x).data[:3] == [2.0, 4.0, 6.0])
    check("-x", (-x).data[:3] == [-1.0, -2.0, -3.0])
    check("x ** 2", (x ** 2).data[:3] == [1.0, 4.0, 9.0])
    a = mt.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = mt.Tensor([1.0, 0.0, 0.0, 1.0], [2, 2])
    check("matmul", (a @ b).data == [1.0, 2.0, 3.0, 4.0])
    check("广播 add", (a + mt.Tensor([10.0, 20.0], [2])).data == [11.0, 22.0, 13.0, 24.0])

    section("归约 / 形状 / 一元")
    check("sum", x.sum().item() == 21.0)
    check("sum(dim=1)", x.sum(1).data == [6.0, 15.0])
    check("mean", close(x.mean().item(), 3.5))
    check("max/min", x.max().item() == 6.0 and x.min().item() == 1.0)
    check("argmax(dim=1)", x.argmax(1).data == [2.0, 2.0])
    check("var(unbiased=False)", close(x.var(False).item(), 35.0 / 12.0))
    check("reshape", x.reshape([3, 2]).shape == [3, 2])
    check("transpose", x.transpose(0, 1).shape == [3, 2])
    check("unsqueeze/squeeze", x.unsqueeze(0).shape == [1, 2, 3]
          and x.unsqueeze(0).squeeze(0).shape == [2, 3])
    check("concat", x.concat(x, 1).shape == [2, 6])
    check("exp/log", close(x.exp().log().data[0], 1.0))
    check("softmax 和为 1", close(x.softmax(1).sum(1).data[0], 1.0))
    check("relu/clamp", mt.Tensor([-1.0, 2.0], [2]).relu().data == [0.0, 2.0]
          and mt.Tensor([-1.0, 2.0], [2]).clamp(0.0, 1.0).data == [0.0, 1.0])
    check("比较 + where", mt.Tensor([-1.0, 2.0], [2]).gt(0.0).data == [0.0, 1.0]
          and mt.Tensor.where(mt.Tensor([1.0, 0.0], [2]), mt.Tensor([5.0, 6.0], [2]),
                              mt.Tensor([7.0, 8.0], [2])).data == [5.0, 8.0])

    section("autograd")
    w = mt.Tensor([1.0, 2.0, 3.0], [3])
    w.requires_grad = True
    loss = w.mul(w).sum()            # d(sum(w^2))/dw = 2w
    loss.backward()
    check("requires_grad 可读写", w.requires_grad is True)
    check("grad 是 Tensor", isinstance(w.grad, mt.Tensor))
    check("dL/dw = 2w", w.grad.data == [2.0, 4.0, 6.0])
    check("非叶子张量 is_leaf=False", loss.is_leaf is False)
    w.zero_grad()
    check("zero_grad 后 grad 为 None", w.grad is None)
    check("detach 后不建图", w.detach().is_leaf is True)

    # 矩阵乘 + 广播的梯度也要对
    p = mt.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    p.requires_grad = True
    q = mt.Tensor([1.0, 1.0], [2])
    q.requires_grad = True
    out = (p @ q).sum()              # = sum over rows of 行和 -> dp 全是 1，dq 是列和 = [4, 6]
    out.backward()
    check("matmul 的 dL/dp", p.grad.data == [1.0, 1.0, 1.0, 1.0])
    check("matmul 的 dL/dq", q.grad.data == [4.0, 6.0])

    section("no_grad")
    z = mt.Tensor([1.0, 2.0], [2])
    z.requires_grad = True
    check("默认记录计算图", mt.is_grad_enabled() is True)
    with mt.no_grad():
        inside = z.mul(z)
        check("with no_grad(): 输出不需要梯度", inside.requires_grad is False)
        check("with no_grad(): 输出是叶子", inside.is_leaf is True)
        check("with no_grad(): 开关已关", mt.is_grad_enabled() is False)
    check("离开作用域自动恢复", mt.is_grad_enabled() is True)
    try:
        with mt.no_grad():
            z.mul(z).sum().backward()
        check("no_grad 里 backward 报错", False)
    except RuntimeError:
        check("no_grad 里 backward 报错", True)
    mt.set_grad_enabled(False)
    check("set_grad_enabled(False) 生效", mt.is_grad_enabled() is False)
    mt.set_grad_enabled(True)

    section("nn 层")
    lin = mt.Linear(3, 2)
    check("Linear: weight/bias 形状", lin.weight.shape == [3, 2] and lin.bias.shape == [2])
    check("Linear: 前向形状", lin(mt.Tensor.zeros([4, 3])).shape == [4, 2])
    check("Linear: parameters 2 个", len(lin.parameters()) == 2)
    check("Linear: parameters 指向同一张量", lin.parameters()[0].data == lin.weight.data)

    conv = mt.Conv2d(1, 2, 3, 1, 1)
    check("Conv2d: weight 4 维", conv.weight.shape == [2, 1, 3, 3])
    check("Conv2d: 前向尺寸不变", conv(mt.Tensor.zeros([1, 1, 5, 5])).shape == [1, 2, 5, 5])
    check("Conv2d: output_height", conv.output_height(5) == 5)

    # weight 属性返回的是层内部张量的引用（不是拷贝）：改动会反映到层上
    ref = lin.weight
    ref.requires_grad = False
    check("Linear: weight 是引用而非拷贝", lin.weight.requires_grad is False)
    ref.requires_grad = True

    pool = mt.MaxPool2d(2, 2)
    inp = mt.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                     9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], [1, 1, 4, 4])
    check("MaxPool2d", pool(inp).data == [6.0, 8.0, 14.0, 16.0])
    check("AvgPool2d", mt.AvgPool2d(2, 2)(inp).data == [3.5, 5.5, 11.5, 13.5])
    check("MaxPool2d(dim) 与 AvgPool2d 形状一致", pool(inp).shape == mt.AvgPool2d(2, 2)(inp).shape)

    bn = mt.BatchNorm(2)
    before = list(bn.running_mean.data)
    bn(mt.Tensor([1.0, 10.0, 3.0, 14.0], [2, 2]))
    check("BatchNorm: 训练后 running_mean 被更新",
          before == [0.0, 0.0] and bn.running_mean.data != [0.0, 0.0])
    bn.eval()
    check("BatchNorm: eval 后 training=False", bn.training is False)

    ln = mt.LayerNorm(4)
    check("LayerNorm: 输出均值≈0", close(ln(mt.Tensor([1.0, 2.0, 3.0, 4.0], [1, 4])).mean().item(), 0.0))

    drop = mt.Dropout(0.5)
    check("Dropout: 训练模式会置零", any(v == 0.0 for v in drop(mt.Tensor.ones([100])).data))
    drop.eval()
    check("Dropout: eval 模式恒等", drop(mt.Tensor.ones([100])).data == [1.0] * 100)

    flat = mt.Flatten()
    check("Flatten", flat(mt.Tensor.zeros([2, 3, 4, 4])).shape == [2, 48])

    emb = mt.Embedding(5, 3)
    check("Embedding: 查表形状", emb(mt.Tensor([0.0, 2.0], [2])).shape == [2, 3])
    check("Embedding: 查到的行等于 weight", emb.lookup([2]).data == emb.weight.data[6:9])

    section("损失层 + Sequential")
    ce = mt.CrossEntropyLoss([2, 0])
    logits = mt.Tensor([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], [2, 3])
    check("CrossEntropyLoss == Tensor.cross_entropy",
          close(ce(logits).item(), logits.cross_entropy([2, 0]).item(), 1e-7))
    mse = mt.MSELoss([1.0, 2.0], [2])
    check("MSELoss", close(mse(mt.Tensor([1.0, 0.0], [2])).item(), 2.0))

    net = mt.Sequential(mt.Linear(2, 8), mt.ReLU(), mt.Linear(8, 2))
    check("Sequential: len", len(net) == 3)
    check("Sequential: 索引到具体层", isinstance(net[1], mt.ReLU) and isinstance(net[0], mt.Linear))
    check("Sequential: 负索引", isinstance(net[-1], mt.Linear))
    check("Sequential: parameters 汇总 4 个", len(net.parameters()) == 4)
    try:
        net[9]
        check("Sequential: 越界抛 IndexError", False)
    except IndexError:
        check("Sequential: 越界抛 IndexError", True)

    section("端到端：用 Python 解 XOR")
    X = mt.Tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2])
    Y = [0.0, 1.0, 1.0, 0.0]
    Yt = mt.Tensor(Y, [4])
    net = mt.Sequential(mt.Linear(2, 8), mt.ReLU(), mt.Linear(8, 1))
    opt = mt.Adam(net.parameters(), lr=0.05)
    first = last = None
    for i in range(2000):
        opt.zero_grad()
        loss = net(X).reshape([4]).mse(Yt)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    net.eval()
    pred = net(X).reshape([4])
    acc = pred.gt(0.0).eq(Yt).mean().item()
    check("XOR 训练收敛", last < 0.01, f"loss {first:.4f} -> {last:.6f}")
    check("XOR 预测 4/4", close(acc, 1.0), f"acc={acc:.4f}")

    # SGD 也要能用
    net2 = mt.Sequential(mt.Linear(2, 1))
    opt2 = mt.SGD(net2.parameters(), lr=0.1, momentum=0.9)
    opt2.zero_grad()
    l2 = net2(X).reshape([4]).mse(Yt)
    l2.backward()
    opt2.step()
    check("SGD 能训练一步", l2.item() > 0.0)

    # 损失层放在 Sequential 末尾
    net3 = mt.Sequential(mt.Linear(4, 8), mt.ReLU(), mt.CrossEntropyLoss([0, 1]))
    X3 = mt.Tensor.zeros([2, 4])
    opt3 = mt.Adam(net3.parameters(), lr=0.05)
    opt3.zero_grad()
    l3 = net3(X3)               # 直接得到标量损失
    check("Sequential 里的损失层返回标量", l3.shape == [])
    l3.backward()
    opt3.step()

    section("生命周期")
    holder = mt.Linear(2, 2)
    params = holder.parameters()
    del holder
    gc.collect()
    check("参数张量在层被回收后仍可用（reference_internal 生效）",
          len(params[0].data) == 4)


if __name__ == "__main__":
    main()
    print()
    if FAILS:
        print(f"{len(FAILS)} 项失败：" + ", ".join(FAILS))
        raise SystemExit(1)
    print("ALL PASS")
