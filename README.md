# mini-pytorch

从零用 C++17 手写的极小张量库。**只依赖标准库**，存储、广播、矩阵乘、归约、形状变换、激活、损失、自动微分、SGD/Adam 全部自己实现，不用 Eigen / BLAS。

当前规模（**按组件分目录**：`tensor/` 张量本体、`nn/` 神经网络层、`optimizer/` 优化器；每个目录内部再按模块拆文件）：

| 目录 / 文件 | 行数 | 作用 |
|---|---|---|
| **tensor/** | | **张量本体** |
| [tensor/Tensor.hpp](tensor/Tensor.hpp) | 458 | `Tensor` 类：**声明 + 数据成员 + 必须留在头文件里的模板** |
| [tensor/TensorCore.cpp](tensor/TensorCore.cpp) | 40 | 构造函数、`detach`、`sub_inplace`（就地更新原语） |
| [tensor/TensorFactory.cpp](tensor/TensorFactory.cpp) | 79 | 生成函数：`zeros` / `ones` / `eye` / `arange` / `one_hot` / `scalar` |
| [tensor/TensorArith.cpp](tensor/TensorArith.cpp) | 185 | 逐元素算术与 `matmul` |
| [tensor/TensorReduce.cpp](tensor/TensorReduce.cpp) | 363 | `neg` / `abs` 与各种归约 |
| [tensor/TensorShape.cpp](tensor/TensorShape.cpp) | 233 | `reshape` / `flatten` / `unsqueeze` / `squeeze` / `concat` / `transpose` |
| [tensor/TensorUnary.cpp](tensor/TensorUnary.cpp) | 177 | `exp` / `log` / `sqrt` / `pow` / `sigmoid` / `ReLU` / `tanh` / `softmax` / `log_softmax` |
| [tensor/TensorLoss.cpp](tensor/TensorLoss.cpp) | 62 | `mse` / `cross_entropy` / `bce_with_logits` / `bce` |
| [tensor/TensorAutograd.cpp](tensor/TensorAutograd.cpp) | 99 | 自动微分：`backward` 与内部工具 |
| [tensor/TensorUtil.cpp](tensor/TensorUtil.cpp) | 96 | 内部公共工具：`check_dim` / `next_index` / `outer_inner` / `scalar` / `log_clamped` / `numel` / `slice_dim` |
| [tensor/TensorIndex.cpp](tensor/TensorIndex.cpp) | 106 | `index_select`（gather）+ 私有 `index_add_grad`（scatter-add 反向） |
| [tensor/TensorConv.cpp](tensor/TensorConv.cpp) | 151 | `im2col`（卷积展开）+ 私有 `col2im`（反向） |
| [tensor/TensorGradMode.hpp](tensor/TensorGradMode.hpp) | 46 | `is_grad_enabled` / `set_grad_enabled` / `GradModeGuard` / `NoGradGuard` |
| [tensor/TensorGradMode.cpp](tensor/TensorGradMode.cpp) | 24 | 开关本体（`thread_local`，默认开） |
| [tensor/detail/TensorDetail.hpp](tensor/detail/TensorDetail.hpp) | 98 | 形状工具：广播推导、步长、`matmul` 形状规划 |
| [tensor/detail/AutogradMeta.hpp](tensor/detail/AutogradMeta.hpp) | 24 | `AutogradMeta`（梯度槽 + `grad_fn`） |
| [tensor/detail/AutogradNode.hpp](tensor/detail/AutogradNode.hpp) | 52 | `Node` + `BackwardCtx`（必须在 `Tensor` 之后包含） |
| **nn/** | | **神经网络层** |
| [nn/Module.hpp](nn/Module.hpp) | 49 | 层基类：`forward` / `parameters` / `zero_grad` / `train` / `eval` |
| [nn/Module.cpp](nn/Module.cpp) | 13 | `zero_grad()` 的默认实现 |
| [nn/Linear.hpp](nn/Linear.hpp) | 29 | 全连接层（`weight: [in, out]`、`bias: [out]`） |
| [nn/Linear.cpp](nn/Linear.cpp) | 64 | 可复现初始化 + `x @ W + b` |
| [nn/ReLU.hpp](nn/ReLU.hpp) | 16 | 激活层 `ReLU` + 激活函数 `ReLUctant` |
| [nn/ReLU.cpp](nn/ReLU.cpp) | 14 | 两者都转发到张量算子 `Tensor::ReLU()` |
| [nn/SiLU.hpp](nn/SiLU.hpp) | 15 | 激活层 `SiLU` + 激活函数 `Silu` |
| [nn/SiLU.cpp](nn/SiLU.cpp) | 14 | `x·σ(x)`，两个现成算子组合 |
| [nn/GELU.hpp](nn/GELU.hpp) | 19 | 激活层 `GELU` + 激活函数 `Gelu`（tanh 近似） |
| [nn/GELU.cpp](nn/GELU.cpp) | 20 | `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` |
| [nn/LayerNorm.hpp](nn/LayerNorm.hpp) | 41 | LayerNorm（`normalized_shape` / `eps` / `elementwise_affine`） |
| [nn/LayerNorm.cpp](nn/LayerNorm.cpp) | 94 | 末尾 k 维标准化 + 仿射；三种构造重载 |
| [nn/Dropout.hpp](nn/Dropout.hpp) | 28 | Dropout（`p` / `seed`，训练时生效） |
| [nn/Dropout.cpp](nn/Dropout.cpp) | 34 | inverted dropout：掩码张量 + 逐元素乘 |
| [nn/Sequential.hpp](nn/Sequential.hpp) | 72 | 顺序容器（可变参数构造 / `add` / `operator[]` / `size`） |
| [nn/Sequential.cpp](nn/Sequential.cpp) | 33 | 逐层前向 + 参数汇总 + 模式下发 |
| [nn/AvgPool2d.hpp](nn/AvgPool2d.hpp) | 39 | 2D 平均池化（含 `count_include_pad`） |
| [nn/AvgPool2d.cpp](nn/AvgPool2d.cpp) | 74 | im2col(填 0) + reshape + mean + reshape |
| [nn/Embedding.hpp](nn/Embedding.hpp) | 42 | 查表层 `[V,D]` + 下标 -> 下标形状 + `[D]` |
| [nn/Embedding.cpp](nn/Embedding.cpp) | 78 | 一次 `index_select`（自带 scatter-add 反向） |
| [nn/CrossEntropyLoss.hpp](nn/CrossEntropyLoss.hpp) | 30 | 交叉熵损失层（目标固定在层里，能进 Sequential） |
| [nn/CrossEntropyLoss.cpp](nn/CrossEntropyLoss.cpp) | 27 | 转发到 `Tensor::cross_entropy` |
| [nn/MSELoss.hpp](nn/MSELoss.hpp) | 27 | 均方误差损失层 |
| [nn/MSELoss.cpp](nn/MSELoss.cpp) | 24 | 转发到 `Tensor::mse` |
| [nn/Conv2d.hpp](nn/Conv2d.hpp) | 48 | 2D 卷积层（im2col 实现，weight 保持 4 维 `[O,C,k,k]`） |
| [nn/MaxPool2d.hpp](nn/MaxPool2d.hpp) | 39 | 2D 最大池化（`stride = 0` 表示跟 kernel 一样） |
| [nn/MaxPool2d.cpp](nn/MaxPool2d.cpp) | 67 | `im2col`(填 `-inf`) + `reshape` + `max` + `reshape` |
| [nn/Flatten.hpp](nn/Flatten.hpp) | 24 | 摊平层（Flatten(start_dim = 1, end_dim = -1)） |
| [nn/Flatten.cpp](nn/Flatten.cpp) | 71 | 合并 [start,end] 这几维；就是一次 reshape |
| [nn/BatchNorm.hpp](nn/BatchNorm.hpp) | 52 | 批归一化（一份实现覆盖 BatchNorm1d/2d/3d，带 running 统计量） |
| [nn/BatchNorm.cpp](nn/BatchNorm.cpp) | 126 | 训练用 batch 统计量 + 更新滑动平均；推理用 running_* |
| [nn/Conv2d.cpp](nn/Conv2d.cpp) | 118 | 前向 = im2col + matmul + 广播加 + reshape |
| **optimizer/** | | **优化器** |
| [optimizer/SGD.hpp](optimizer/SGD.hpp) | 31 | SGD 声明（动量 / 权重衰减 / Nesterov） |
| [optimizer/SGD.cpp](optimizer/SGD.cpp) | 58 | SGD 实现 |
| [optimizer/Adam.hpp](optimizer/Adam.hpp) | 36 | Adam 声明 |
| [optimizer/Adam.cpp](optimizer/Adam.cpp) | 61 | Adam 实现 |

| **根目录** | | |
| [main.cpp](main.cpp) | 534 | 入口 + 全功能演示（算子/autograd/损失/训练/nn 各层/模式切换/no_grad 推理/argmax 算准确率） |
| **Python 绑定** | | **pybind11** |
| [bindings.cpp](bindings.cpp) | 393 | 全部绑定：Tensor / no_grad / nn 各层 / SGD·Adam |
| [minitorch/__init__.py](minitorch/__init__.py) | 22 | 把 `_minitorch` 的符号转出去，`import minitorch` 就能用 |
| [setup.py](setup.py) | 59 | 用 MSVC + pybind11 编扩展；`packages=["minitorch"]` |
| [pyproject.toml](pyproject.toml) | 3 | 构建依赖（setuptools / wheel / pybind11>=3.0） |
| [test_bindings.py](test_bindings.py) | 243 | 85 项 Python 冒烟测试 |
| [CMakeLists.txt](CMakeLists.txt) | 44 | 备选构建路径（暂未验证） |
| [.clang-format](.clang-format) | 21 | 代码风格（Allman 大括号、4 空格、100 列） |
| [.vscode/tasks.json](.vscode/tasks.json) | 32 | VS Code 构建任务（`Ctrl+Shift+B`） |

设计要点：
- **float32 一维连续存储 + 形状向量**（行优先，无 stride / 无视图），算子返回新张量（**值语义**）；唯一的例外是 `sub_inplace`，只给优化器就地改参数用。
- **自动微分**：每个 `Tensor` 持有一份共享的 `AutogradMeta`（`requires_grad` / 梯度槽 / `grad_fn`），**反向公式由每个算子自己用 lambda 传进去**（详见 1.11）。
- **拆分原则**：先按"组件"分目录（`tensor/`、`optimizer/`），目录内再按"读者会一起找的东西"分文件；类里只放声明，模板的实现必须留在头文件（`binary_element` / `reduce_*` / `unary_elementwise` / `attach_grad`）。

---

## 构建与运行

环境：MinGW-w64 g++ 8.1（`D:\mingw64\bin\g++.exe`），C++17。

**方式一：VS Code 任务**
1. 任意文件活动状态下 `Ctrl+Shift+B`（任务用的是 `${workspaceFolder}`，与活动文件无关）
2. 生成根目录的 `main.exe`
3. 集成终端 `.\main.exe`，或点右上角 ▷

**方式二：命令行**（三个目录一起交给 g++；MinGW 的 g++ 会自己展开 `*.cpp`）

```bat
D:\mingw64\bin\g++.exe -std=c++17 -O2 -Wall -Wextra tensor\*.cpp nn\*.cpp optimizer\*.cpp main.cpp -o main.exe
main.exe
```

只编译两个组件（不含 `main.cpp`，用于自带 `main` 的测试程序）：

```bat
D:\mingw64\bin\g++.exe -std=c++17 -O2 -I. test.cpp tensor\*.cpp nn\*.cpp optimizer\*.cpp -o test.exe
```

注意：
- **不要单独编译某个 `.cpp` 或 `Tensor.hpp`**：实现分散在同组件的多个文件里，只编一个会 `undefined reference`；头文件没有 `main` 则报 `WinMain` 未定义。
- 组件之间的依赖只有一条：`optimizer/SGD.hpp` / `optimizer/Adam.hpp` → `../tensor/Tensor.hpp`（用相对路径，所以不需要任何 `-I` 参数）。
- 想格式化代码：VS Code 里 `Shift+Alt+F`（C/C++ 扩展自带 clang-format，会读 `.clang-format`）。

---

## 一、tensor/（Tensor 本体）

> 下表"位置"列的格式是 `文件:行号`（例如 `tensor/TensorArith.cpp:5`）；改动代码后行号会偏移。

### 1.1 构造与访问

| 位置 | 签名 | 作用 |
|---|---|---|
| tensor/TensorCore.cpp:5 | `Tensor(std::vector<float> data, std::vector<int64_t> shape)` | 唯一构造函数（按值接收 + 移动）。校验 `data.size() == numel(shape)`，不一致抛 `Tensor: data size != shape product` |
| tensor/Tensor.hpp:149 | `const std::vector<int64_t>& shape() const` | 返回形状（const 引用，不拷贝） |
| tensor/Tensor.hpp:150 | `const std::vector<float>& data() const` | 返回底层数据（行优先展平，const 引用） |
| tensor/TensorCore.cpp:10 | `float item() const` | 单元素张量取值（等价于 PyTorch 的 `.item()`）；元素数不为 1 抛异常。省得每处都写 `data()[0]` |

### 1.2 生成函数（工厂，全部 static）

| 位置 | 签名 | 作用 |
|---|---|---|
| tensor/TensorFactory.cpp:5 | `zeros(shape)` | 全 0 张量；`zeros({})` 是标量 |
| tensor/TensorFactory.cpp:11 | `ones(shape)` | 全 1 张量 |
| tensor/TensorFactory.cpp:17 | `eye(n)` | n×n 单位阵 |
| tensor/TensorFactory.cpp:22 | `eye(n, m)` | n×m，对角线上 `min(n,m)` 个 1，其余 0 |
| tensor/TensorFactory.cpp:37 | `arange(end)` | `[0, 1, ..., <end)`，步长 1 |
| tensor/TensorFactory.cpp:42 | `arange(start, end, step = 1)` | `[start, end)` 不含 end；`step == 0` 抛异常；步长与区间方向相反时返回空张量 `shape=[0]` |
| tensor/TensorFactory.cpp:61 | `one_hot(targets, num_classes)` | `[N, C]`，第 i 行第 `targets[i]` 列为 1。校验 `num_classes > 0`、`0 <= targets[i] < C` |

### 1.3 逐元素算术（支持 NumPy 广播）

| 位置 | 签名 | 作用 |
|---|---|---|
| tensor/TensorArith.cpp:39 | `add / sub / mul / div(const Tensor& other)` | 逐元素加减乘除，形状按 NumPy 规则右对齐广播；不可广播时抛 `broadcast: shape mismatch`。**`add` / `sub` / `mul` / `div` 都已支持反向**（含广播归约） |
| tensor/Tensor.hpp:61 | `friend operator+ - * /` | 转发到上面的成员函数，与 PyTorch 一致：**`*` 是逐元素乘，不是矩阵乘** |

### 1.4 矩阵乘法

| 位置 | 签名 | 作用 |
|---|---|---|
| tensor/TensorArith.cpp:44 | `matmul(const Tensor& other)` | `[..., m, k] × [..., k, n] → [..., m, n]`。1 维操作数按 NumPy 规则提升（`[k]→[1,k]` / `[k]→[k,1]`）并在结果中压回；批次维广播；0 维操作数抛异常。形状判定全部交给 `plan_matmul`，循环里零分支。**已支持反向**：`dA = g @ Bᵀ`、`dB = Aᵀ @ g`，并处理 1 维提升与批次广播的归约 |
| tensor/TensorArith.cpp:64 | `gt` / `lt` / `ge` / `le` / `eq` / `ne` | 逐元素比较，返回 1.0 / 0.0（支持广播）。不可导所以不建图；想和常数比就 `x.gt(Tensor::scalar(0.0f))` |
| tensor/TensorArith.cpp:94 | `static Tensor where(cond, a, b)` | 逐元素三选一：`cond != 0` 取 a 否则取 b。先把 cond 归一成严格 0/1，所以 `cond = 0.5` 也是"选 a"而不是插值。**有反向**：`da = g·mask`、`db = g·(1-mask)`，不给 cond 梯度（与 PyTorch 一致） |
| tensor/TensorUnary.cpp:21 | `clamp(lo, hi)` / `clamp_min(lo)` / `clamp_max(hi)` | 逐元素夹值。**有反向**：区间内 1、被夹住 0（边界算区间内，与 PyTorch 一致）；`lo > hi` 抛异常 |
| tensor/TensorUnary.cpp:12 | `sign()` | 逐元素 -1 / 0 / +1。导数恒为 0，所以不建图 |

### 1.5 归约

**整体归约**（结果都是 0 维标量）：

| 位置 | 签名 | 作用 |
|---|---|---|
| tensor/TensorReduce.cpp:5 | `neg()` | 逐元素取负（无一元 `operator-`） |
| tensor/TensorReduce.cpp:10 | `abs()` | 逐元素绝对值（`std::fabs`） |
| tensor/TensorReduce.cpp:15 | `sum()` | 全部元素求和；空张量返回 0。**有反向**（梯度广播回输入形状） |
| tensor/TensorReduce.cpp:27 | `prod()` | 全部元素连乘；空张量返回 1 |
| tensor/TensorReduce.cpp:32 | `max()` | 全局最大；**空张量抛异常**（没有单位元） |
| tensor/TensorReduce.cpp:37 | `min()` | 全局最小；空张量抛异常 |
| tensor/TensorReduce.cpp:42 | `mean()` | 全局均值 = `sum()/numel`；空张量得到 `nan`。**有反向**（`dx = g/n`） |

**按维度归约**（`keepdim = true` 时保留被归约的那一维、长度 1，便于直接和输入广播）：

| 位置 | 签名 | 作用 |
|---|---|---|
| tensor/TensorReduce.cpp:56 | `sum(int64_t dim, bool keepdim = false)` | 沿 dim 求和，默认丢掉该维。**有反向** |
| tensor/TensorReduce.cpp:74 | `prod(int64_t dim, bool keepdim = false)` | 沿 dim 连乘 |
| tensor/TensorReduce.cpp:79 | `max(int64_t dim, bool keepdim = false)` | 沿 dim 取最大 |
| tensor/TensorReduce.cpp:84 | `min(int64_t dim, bool keepdim = false)` | 沿 dim 取最小 |
| tensor/TensorReduce.cpp:89 | `mean(int64_t dim, bool keepdim = false)` | 沿 dim 求均值（内部 `sum(dim)/shape[dim]`）。**有反向**（`1/n` 折进了反向里） |
| tensor/TensorReduce.cpp:233/243 | `argmax(dim, keepdim = false)` / `argmax()` | 沿 dim（或整体展平）取最大值的**下标**。并列取第一个；不可导、不建图 |
| tensor/TensorReduce.cpp:238/260 | `argmin(dim, keepdim = false)` / `argmin()` | 同上，取最小 |
| tensor/TensorReduce.cpp:277/287/310 | `var()` / `var(correction)` / `var(dim, keepdim, correction)` | 方差，用 `mean/sub/mul` 组合（**反向自动**）。修正方式用强类型 `VarCorrection::{unbiased, biased}`（无偏除以 n-1，是 PyTorch 的默认）；无偏但元素不足 2 个时抛异常，避免悄悄给 NaN |
| tensor/TensorReduce.cpp:282/305/334 | `std()` / `std(correction)` / `std(dim, keepdim, correction)` | 标准差 = 方差开根号（`sqrt` 有反向） |

### 1.6 形状变换

| 位置 | 签名 | 作用 |
|---|---|---|
| tensor/TensorShape.cpp:5 | `reshape(std::vector<int64_t> new_shape)` | 改变形状，元素总数必须一致；支持**一个** `-1` 自动推断；重复 `-1` / 无法整除 / 负数维度分别抛异常 |
| tensor/TensorShape.cpp:49 | `flatten()` | 展平成 `[numel]` |
| tensor/TensorShape.cpp:54 | `unsqueeze(int64_t dim)` | 在 dim 处插入长度 1 的维；允许 `dim == ndim`（插到末尾）、允许负 dim |
| tensor/TensorShape.cpp:83 | `squeeze()` | 去掉**所有**长度为 1 的维 |
| tensor/TensorShape.cpp:96 | `squeeze(int64_t dim)` | 去掉指定维（该维必须为 1，否则抛异常） |
| tensor/TensorShape.cpp:117 | `concat(const Tensor& other, int64_t dim)` | 沿 dim 拼接；要求维数相同、除 dim 外各维相等；内层用 `std::copy` 整段拷贝 |
| tensor/TensorShape.cpp:154 | `transpose(int64_t dim0, int64_t dim1)` | 交换两维（**真搬数据**，不是视图）；`dim0 == dim1` 时直接返回自身 |
| tensor/TensorIndex.cpp:37 | `index_select(dim, indices)` | 沿 dim 按下标取（gather），形状把 `shape[dim]` 换成 `indices.size()`。Embedding 的地基。**有反向**：梯度按同样的下标**累加**回原张量（同一行选多次就累加，没选中的为 0） |
| tensor/TensorConv.cpp:38 | `im2col(kh, kw, stride = 1, padding = 0, pad_value = 0)` | 把 `[N,C,H,W]` 按 k×k 滑窗摊成 `[N, C*k*k, L]`（L = OH*OW），越界位置取 0 —— 零填充不必真 pad 出输入。布局同 PyTorch `F.unfold`，于是 `weight[O,K].matmul(col[N,K,L])` 一次矩阵乘就是卷积。**有反向**（私有 `col2im`，重叠 patch 相加） |

### 1.7 一元算子 / 激活（**全部支持反向**）

| 位置 | 签名 | 作用 |
|---|---|---|
| tensor/TensorUnary.cpp:5 | `exp()` | 逐元素 e^x |
| tensor/TensorUnary.cpp:10 | `log()` | 逐元素自然对数 |
| tensor/TensorUnary.cpp:15 | `sqrt()` | 逐元素平方根 |
| tensor/TensorUnary.cpp:20 | `pow(float p)` | 逐元素 x^p |
| tensor/TensorUnary.cpp:25 | `sigmoid()` | 1/(1+e^-x) |
| tensor/TensorUnary.cpp:30 | `ReLU()` | max(x, 0) |
| tensor/TensorUnary.cpp:35 | `tanh()` | 双曲正切 |

### 1.8 归一化与损失

| 位置 | 签名 | 作用 |
|---|---|---|
| tensor/TensorUnary.cpp:40 | `softmax(int64_t dim)` | 沿 dim 归一化；先减该维 max 再取 exp，避免上溢 |
| tensor/TensorUnary.cpp:55 | `log_softmax(int64_t dim)` | `(x - max) - log(Σexp(x - max))`；比 `softmax().log()` 数值稳定 |
| tensor/TensorLoss.cpp:5 | `mse(const Tensor& other)` | 均方误差，要求形状完全相同 |
| tensor/TensorLoss.cpp:15 | `cross_entropy(const Tensor& targets)` | 交叉熵，target 是 **one-hot / 软标签**（与 logits 同形状 `[B, C]`）。`mean_i(-Σ_c y[i,c]·log_softmax[i,c])` |
| tensor/TensorLoss.cpp:28 | `cross_entropy(const std::vector<int64_t>& targets)` | 交叉熵，target 是**整数类别下标**（长度 = B）。内部转成 `one_hot` 再走上面那条加权求和；校验目标个数 = batch、下标范围 |
| tensor/TensorLoss.cpp:41 | `bce_with_logits(const Tensor& targets)` | 二元交叉熵（输入 logits，推荐）：`mean(softplus(x) - x·y)`，其中 `softplus(x) = max(x,0) + log(1+e^{-|
| tensor/TensorLoss.cpp:52 | `bce(const Tensor& targets)` | 二元交叉熵（输入概率 p）：`mean(-(y·log p + (1-y)·log(1-p)))`；log 结果截到 ≥ -100，对齐 PyTorch 的 `BCELoss` |

### 1.9 私有工具函数（外部不可调用）

| 位置 | 签名 | 作用 |
|---|---|---|
| tensor/TensorUtil.cpp:5 | `static int64_t check_dim(dim, ndim, op)` | 所有带 dim 的算子共用：范围检查 + 负 dim 归一化到 `[0, ndim)`，越界抛 `<op>: dim out of range` |
| tensor/TensorUtil.cpp:14 | `static void next_index(idx, shape)` | 行优先多维下标进位（`matmul` 的批次遍历、`transpose` 共用） |
| tensor/Tensor.hpp:172 | `struct OuterInner { outer, inner; }` | 沿某维看时"外层个数 × 内层连续段长度"的组合 |
| tensor/TensorUtil.cpp:26 | `OuterInner outer_inner(int64_t dim) const` | 计算上面那个 outer/inner（`concat`、`reduce_dim` 共用） |
| tensor/Tensor.hpp:181 | `template<BinaryOp> static Tensor binary_element(a, b, op)` | 逐元素二元运算的公共实现：**形状相同走快路径**（顺序遍历），否则走广播 + 步长索引推进 |
| tensor/TensorUtil.cpp:41 | `static Tensor scalar(float v)` | 造一个 0 维单元素张量，用于标量广播（如 `1 - p`） |
| tensor/TensorUtil.cpp:46 | `static Tensor log_clamped(const Tensor& t)` | 逐元素 log 并截到 ≥ -100，避免 p=0/1 时 `-inf` 传播成 `nan` |
| tensor/Tensor.hpp:237 | `template<Accumulate> Tensor reduce_all_identity(acc, identity)` | 整体归约，有空输入单位元（sum→0、prod→1） |
| tensor/Tensor.hpp:249 | `template<Accumulate> Tensor reduce_all_nonempty(acc, op)` | 整体归约，无单位元（max/min），空张量抛异常 |
| tensor/Tensor.hpp:264 | `template<Accumulate> Tensor reduce_dim(dim, acc, keepdim)` | 按维度归约的公共实现，`keepdim` 逻辑在这里 |
| tensor/TensorUtil.cpp:55 | `static int64_t numel(const std::vector<int64_t>& shape)` | 形状的元素总数；遇到负数维度抛异常 |
| tensor/Tensor.hpp:318 | `template<ActivateOp> static Tensor unary_elementwise(a, op)` | 一元逐元素的公共实现（输入输出同形状，直接顺序遍历） |

> **模板（`binary_element` / `reduce_*` / `unary_elementwise` / `outer_inner`）的定义必须留在 Tensor.hpp**：别的 .cpp 要用就得看得见定义，搬进 .cpp 会 `undefined reference`。

### 1.10 数据成员

| 行号 | 成员 | 作用 |
|---|---|---|
| tensor/Tensor.hpp:153 | `std::vector<float> data_` | 行优先连续存储的所有元素 |
| tensor/Tensor.hpp:154 | `std::vector<int64_t> shape_` | 形状；`{}` 表示 0 维标量（`numel({}) == 1`） |
| tensor/Tensor.hpp:155 | `std::shared_ptr<tensor_detail::AutogradMeta> meta_` | 自动微分元信息，**拷贝时共享**（见 1.11） |

### 1.11 自动微分（autograd）

**设计**：每个 `Tensor` 在构造时创建一份 `AutogradMeta`；`Tensor` 拷贝时这个 `shared_ptr` 被一起复制，于是"同一个变量"的所有副本**共用一份梯度槽**。这是值语义下梯度能被外面看见的关键——图里保存的输入副本和用户手上的 `x` 指向同一个梯度对象。

反向公式**不由 Tensor 统一实现**（一个函数写 `if (op == ADD) ... else if (op == MUL)` 是灾难），而是由**每个算子**传给 `attach_grad` 一个"怎么算局部导数"的 lambda。

**所有算子现在都有反向**：`add` / `sub` / `mul` / `div`（含广播）、`matmul`（含 1 维提升与批次广播）、`neg` / `abs`、`sum` / `prod` / `max` / `min` / `mean`（整体形式与 `(dim, keepdim)` 形式）、`reshape` / `flatten` / `unsqueeze` / `squeeze` / `concat` / `transpose`、`exp` / `log` / `sqrt` / `pow` / `sigmoid` / `ReLU` / `tanh` / `softmax` / `log_softmax`，以及全部损失函数。

其中**损失函数一行反向代码都没写**——它们是组合出来的（`mse = sub→mul→mean`、`cross_entropy = log_softmax→mul→sum→neg→mean`、`bce_with_logits = relu/abs/exp/log/…`），前面的算子有了反向，链式法则自动生效。

三个已知的语义提醒：
- `prod` 的反向是 `g·P/x_i`，**`x_i = 0` 时会出 inf/nan**（真正的实现要单独算"除它之外的连乘"）。
- `max` / `min` 的反向把梯度给到该维的**所有**并列最大值（PyTorch 只给其中一个）。
- 反向里的 `exp`/`sigmoid`/`tanh`/`softmax` 是**重算**的，不是复用前向输出（原因见本节末尾）。

**公开接口**

| 位置 | 签名 | 作用 |
|---|---|---|
| tensor/Tensor.hpp:24 | `bool requires_grad` | 叶子标记。**必须在参与运算之前设置**（它是普通 bool，只按当时的值拷贝） |
| tensor/Tensor.hpp:27 | `bool is_leaf() const` | 没有 `grad_fn` 即为叶子 |
| tensor/Tensor.hpp:28 | `const Tensor* grad() const` | 读梯度；还没反向过时为 `nullptr` |
| tensor/Tensor.hpp:29 | `void zero_grad()` | 清掉梯度（释放梯度槽） |
| tensor/TensorAutograd.cpp:65 | `void backward()` | 反向传播：播种 `d(loss)/d(loss) = 1`，再按逆拓扑序逐节点回传 |
| tensor/TensorCore.cpp:5 | `Tensor detach() const` | 脱离计算图的副本（同数据、`requires_grad = false`、无 `grad_fn`） |

**已支持反向**：全部算子（见上一段）。**没有反向的只剩不需要的**：工厂函数（`zeros` 等，没有输入）、`detach`（本来就脱离图）、以及 autograd 自己的接口（`backward` / `grad` / `zero_grad`）。

**内部实现**

| 行号 | 名字 | 作用 |
|---|---|---|
| tensor/detail/AutogradMeta.hpp:18 | `struct tensor_detail::AutogradMeta` | 元信息：`grad_fn`（产生该张量的反向节点，叶子为 null）+ `grad`（梯度槽，反向时懒创建） |
| tensor/detail/AutogradNode.hpp:29 | `struct tensor_detail::BackwardCtx` | **反向下文**：把"取输入值 / 取上游梯度 / 写回梯度"收进来，见下方"怎么写反向" |
| tensor/detail/AutogradNode.hpp:57 | `struct tensor_detail::Node` | 反向节点：`inputs`（前向输入副本）+ `out_meta`（取上游梯度）+ `backward`（接收 `BackwardCtx`）+ `visited` |
| tensor/Tensor.hpp:160 | `template<Backward> static void attach_grad(out, inputs, backward)` | **唯一的"自动微分函数"**：任一输入需要梯度时给 `out` 挂上 `Node`，并把算子传来的反向 lambda 存进去 |
| tensor/TensorAutograd.cpp:5 | `static void accumulate_grad(t, g)` | 梯度**累加**（首次先初始化成全 0）；`t.requires_grad == false` 直接跳过 |
| tensor/TensorAutograd.cpp:28 | `static Tensor reduce_to(g, target)` | **广播算子的反向**：把梯度沿被广播的维 `sum` 回原形状 |
| tensor/TensorAutograd.cpp:45 | `static Tensor expand_like(g, target)` | **归约算子的反向**：`ones(target) * g`，把梯度广播回输入形状 |
| tensor/TensorAutograd.cpp:50 | `static void collect_topo(n, order)` | 后序 DFS 收集节点，反转后即为反向传播顺序；用 `visited` 去重以支持菱形结构 |
| tensor/TensorUtil.cpp:69 | `static Tensor slice_dim(src, dim, start, count)` | 沿 dim 切片，`concat` 的反向用它把梯度切回两块 |
| tensor/TensorAutograd.cpp:1 起 | 上述工具的**实现** | 非模板的实现在 TensorAutograd.cpp；模板 `attach_grad` 的定义留在 Tensor.hpp（模板定义必须对使用点可见） |

### 1.12 no_grad（关掉计算图记录）

推理时不需要梯度，但默认每个算子都会挂反向节点、并且**把输入深拷贝一份存进节点**。`no_grad` 让这些工作全部省掉。

| 位置 | 签名 | 作用 |
|---|---|---|
| tensor/TensorGradMode.cpp:15 | `bool is_grad_enabled()` | 当前是否记录计算图（`thread_local`，默认 `true`） |
| tensor/TensorGradMode.cpp:20 | `void set_grad_enabled(bool)` | 手动开关 |
| tensor/TensorGradMode.hpp:27 | `class GradModeGuard` | RAII：构造时设成给定值，析构时恢复（可嵌套） |
| tensor/TensorGradMode.hpp:41 | `class NoGradGuard` | 最常用：`GradModeGuard(false)` |
| tensor/Tensor.hpp:351 | `Tensor::attach_grad` | 建图的唯一入口，第一行就是 `if (!is_grad_enabled()) return;` |
| tensor/TensorAutograd.cpp:65 | `Tensor::backward()` | `no_grad` 里调用会**直接报错**，而不是静默什么都不做 |

```cpp
{
    NoGradGuard guard;                     // 进入作用域后不再建图
    Tensor y = net.forward(x);             // y.requires_grad == false，且 y 是叶子（没有 grad_fn）
}
// 离开作用域自动恢复（嵌套安全）

set_grad_enabled(false);  ...  set_grad_enabled(true);      // 手动
{ GradModeGuard g(true);  ... }                             // 在 no_grad 里临时重新打开
```

**和 `Module::eval()` 是两件事**（PyTorch 也是如此）：`eval()` 只切层的行为（关掉 Dropout 等），**不关 autograd**——验证集上想算梯度/损失就得让 autograd 开着；`no_grad` 只管建图，不影响层的训练/推理行为，也不影响优化器更新参数。

**实测收益**（同一份代码，只切开关）：

| 负载 | 建图 | no_grad | 说明 |
|---|---|---|---|
| 小模型（16→16→16，batch 32）20000 次前向 | 940 ms | **700 ms（1.34×）** | 算子多、张量小 → 节点分配与输入拷贝占比高，这时最划算 |
| 大张量逐元素链（1M 元素 × 4 个算子） | 3071 ms | 3122 ms | 带宽受限，差距在噪声里 |
| 512×512 两层 Linear | 3187 ms | 2961 ms | 计算受限，建图开销可忽略 |

> 附带好处是**内存**：不建图就没有 `Node`（连同它保存的输入副本），也不会分配中间梯度。
### 1.13 怎么写一个新算子的反向

反向下文 `tensor_detail::BackwardCtx` 把重复动作都收走了，算子只需要回答"局部导数是什么"：

| 成员 | 作用 |
|---|---|
| `c.x(i)` | 第 i 个输入的值（**已 detach**，参与反向计算不会再建图） |
| `c.g` | 上游梯度（**已 detach**） |
| `c.grad(i, v)` | 把 `v` 累加为第 i 个输入的梯度，**自动把形状归约回该输入**（被广播过的输入不用自己处理） |
| `c.grad_expand(i, v)` | 同上，但先把 `v` **广播**到输入形状（归约类算子用） |

于是四类算子的反向都只有几行：

```cpp
// 二元逐元素（TensorArith.cpp）：da = g·b，db = g·a
attach_grad(out, {*this, other}, [](const tensor_detail::BackwardCtx& c)
{
    c.grad(0, c.g.mul(c.x(1)));
    c.grad(1, c.g.mul(c.x(0)));
});

// 一元逐元素（TensorUnary.cpp）：dx = g·f'(x)
attach_grad(out, {*this}, [](const tensor_detail::BackwardCtx& c)
{
    c.grad(0, c.g.mul(c.x(0).exp()));      // exp 的局部导数
});

// 归约（TensorReduce.cpp）：先把被归约掉的那一维补回来，再广播
attach_grad(out, {*this}, [d, keepdim](const tensor_detail::BackwardCtx& c)
{
    Tensor gg = c.g;
    if (!keepdim) { gg = gg.unsqueeze(d); }
    c.grad_expand(0, gg);
});

// 形状算子（TensorShape.cpp）：数据没动，梯度 reshape 回去就行
attach_grad(res, {*this}, [orig](const tensor_detail::BackwardCtx& c)
{
    c.grad(0, c.g.reshape(orig));
});
```


**用法**

```cpp
Tensor x({1, 2, 3, 4, 5, 6}, {2, 3});
x.requires_grad = true;                  // 叶子：要在参与运算之前设置
Tensor w({10, 20, 30}, {3});
w.requires_grad = true;

Tensor loss = x.mul(w).sum(1).mean();    // 只用到有反向的算子
loss.backward();

x.grad()->data();    // dl/dx
w.grad()->data();    // dl/dw —— w 被广播过，梯度会被 sum 回 [3]
```

**各算子的反向公式**（分散在 TensorArith / TensorUnary / TensorReduce / TensorShape / TensorUtil.cpp 里，由算子自己传给 `attach_grad`）

| 算子 | 局部导数 | 实现要点 |
|---|---|---|
| `add` / `sub` | `da = g`，`db = ±g` | 被广播过的一方由 `c.grad` 自动归约 |
| `mul` | `da = g·b`，`db = g·a` | 用 `c.x(1)` / `c.x(0)` 取前向输入值 |
| `div` | `da = g/b`，`db = −g·a/b²` | |
| `matmul` | `dA = g @ Bᵀ`，`dB = Aᵀ @ g` | 先 `reshape` 回 `[..., m, n]`，再 `transpose(-2,-1)`，最后 `reduce_to` + `reshape` 收尾 |
| `neg` / `abs` | `−g` / `g·sign(x)` | `sign` 用 `unary_elementwise` 造（`x = 0` 处取 0） |
| `sum` / `mean` | `g`（mean 再乘 `1/n`） | 归约类：`c.grad_expand` 广播回输入形状；`keepdim = false` 时先 `unsqueeze(d)` |
| `prod` | `g·P/x_i` | ⚠️ **`x_i = 0` 会出 inf/nan**（要求"除它之外的连乘"才行） |
| `max` / `min` | `g·(x == 极值)` | 掩码用 `binary_element` 造；并列时每项都拿到完整梯度（PyTorch 只给一个） |
| `reshape` / `flatten` / `unsqueeze` / `squeeze` | `g.reshape(输入形状)` | 数据没动，只是换个看法 |
| `transpose` | `g.transpose(d0, d1)` | transpose 是自己的逆运算 |
| `concat` | 沿 dim 切两块 | `slice_dim(g, dim, 0, a_dim)` 与 `slice_dim(g, dim, a_dim, b_dim)` |
| `exp` | `g·exp(x)` | 反向里重算 `exp`（见下方说明） |
| `log` | `g/x` | |
| `sqrt` | `g/(2√x)` | |
| `pow(p)` | `g·p·x^(p−1)` | `p` 由 lambda 捕获 |
| `sigmoid` | `g·σ(x)·(1−σ(x))` | |
| `ReLU` | `g·(x>0)` | `x = 0` 处导数为 0，与 PyTorch 一致 |
| `tanh` | `g·(1−tanh²x)` | |
| `softmax` | `y ⊙ (g − Σ_dim(g ⊙ y))` | `y = softmax(x)` 重算，`sum(dim, keepdim=true)` |
| `log_softmax` | `g − softmax(x)·Σ_dim(g)` | 同上 |
| `log_clamped`（内部） | 未截断区域 `g/x`，截断区域 `0` | 必须在**一次**逐元素运算里同时判断+做除法，否则 `x = 0` 时算出 `inf*0 = nan` |
| `mse` / `cross_entropy` / `bce*` | —— | **不写反向**：由上面的算子组合而成，链式法则自动生效 |

> **为什么反向里"重算"而不是复用前向输出**：如果让反向 lambda 捕获前向的输出张量，会形成
> `Tensor → Node → lambda → Tensor` 的 `shared_ptr` 环，永远不释放。所以代价换成"多算一次 exp/sigmoid/softmax"——
> 对教学实现完全够用；真要紧可以后续把中间值单独存进 `Node`（不持有输出的 meta）。

**验证方式**：中心差分 `(f(x+h) − f(x−h)) / (2h)`，对**每个输入张量的每个元素**逐个核对解析梯度。

最近一次全量检查：**226 个偏导、0 个不符**，覆盖 `neg`/`abs`、13 种归约（整体 + 按维 + keepdim，含 `prod`/`max`/`min`）、9 种形状算子（含 `concat` 的三种拼法）、9 个一元算子与 `softmax`/`log_softmax`（2 维/3 维、正负 dim）、`add`/`sub`/`mul`/`div` 的广播、`matmul` 的 8 种形状组合（1 维提升、单/双侧批次广播），以及 5 个损失函数。
另外专门验证了这些容易错的点：
- 同一张量被用两次（`y.mul(y)`）时梯度要**累加**成 `2y`；
- `requires_grad = false` 的输入不产生梯度；
- `zero_grad()` 后再次 `backward()` 能正常工作（`visited` 标记会被清掉）；
- `bce` 在 `p = 0 / 1`（触发 `log` 截断）时梯度仍然有限，不出 `inf`/`nan`。

---

## 二、tensor/detail/（内部工具）

`tensor/detail/` 放的都是**内部细节**：形状/广播/matmul 规划（`TensorDetail.hpp`）与自动微分的元信息、反向节点（`AutogradMeta.hpp` / `AutogradNode.hpp`）。所有自由函数都是 **`inline`**（头文件里不加 `inline` 会被每个包含它的 .cpp 各定义一份 → `multiple definition`）。

| 行号 | 名字 | 作用 |
|---|---|---|
| tensor/detail/TensorDetail.hpp:13 | `struct MatmulPlan` | matmul 的执行计划：`m / k / n`、广播后的 `batch`、两个操作数的批次步长 `a_stride`/`b_stride`（单位 = 矩阵块，广播维为 0）、结果形状 `out_shape` |
| tensor/detail/TensorDetail.hpp:22 | `inline strides_for(own, out)` | 把 `own` 的连续布局步长右对齐到 `out`；自身为 1 或缺维的位置取 0 —— **步长 0 就是广播**，数据一次都不复制 |
| tensor/detail/TensorDetail.hpp:37 | `inline broadcast_shape(a, b)` | NumPy 广播判定：从最后一维往前，每对必须相等或其中一个为 1；返回结果形状，不兼容抛 `broadcast: shape mismatch` |
| tensor/detail/TensorDetail.hpp:56 | `inline plan_matmul(a, b)` | 矩阵乘判定：先做 1 维提升，再查内维 `k == k'`，批次维复用 `broadcast_shape`，最后推出结果形状。失败分别抛 `matmul: dim not exist` / `matmul: inner dim mismatch` |

---

## 三、nn/（神经网络层）

层都继承 `Module`。写一个新层只要管两件事：`forward(x)` 和 `parameters()`；`zero_grad()` 和训练/推理模式由基类负责。

### 3.1 训练 / 推理模式

| 位置 | 签名 | 作用 |
|---|---|---|
| nn/Module.hpp:32 | `void train(bool mode = true)` | 切模式（默认切到训练），并**递归**通知子层 |
| nn/Module.hpp:37 | `void eval()` | 等价于 `train(false)` |
| nn/Module.hpp:38 | `bool is_training() const` | 当前模式；需要区分行为的层在前向里读它 |
| nn/Module.hpp:45 | `virtual void set_training(bool)` | 有子层的层重写它把模式转发下去（无子层不用管） |
| nn/Sequential.cpp:26 | `Sequential::set_training` | 逐个 `layer->train(mode)` |

```cpp
Sequential net(Linear(2, 16), LayerNorm(16), GELU(), Dropout(0.1f), Linear(16, 1));
net.is_training();   // true（默认训练模式）
net.eval();          // 整个网络切推理：连里面的 Dropout 一起关掉
net.train();         // 切回训练
```

**默认训练模式**（和 PyTorch 一致）；`eval()` 之后再 `add()` 进来的层会跟随容器当前模式。目前只有 `Dropout` 真正关心这个开关（`LayerNorm` 不依赖 batch 统计量，训练/推理行为相同）。

### 3.2 层一览

| 位置 | 签名 | 作用 |
|---|---|---|
| nn/Module.hpp:15 | `class Module` | 层基类（虚析构 + 纯虚 `forward`） |
| nn/Module.hpp:22 | `virtual Tensor forward(const Tensor&) = 0` | 前向。**故意不是 const**：Dropout 这类层前向时要改自身状态 |
| nn/Module.hpp:25 | `virtual std::vector<Tensor*> parameters()` | 默认返回空（无参层不用写） |
| nn/Module.hpp:28 / nn/Module.cpp:5 | `virtual void zero_grad()` | 默认实现：遍历 `parameters()` 逐个清梯度槽 |
| nn/Module.hpp:41 | `Tensor operator()(const Tensor&)` | 让层能像函数一样用：`net(x)` ≡ `net.forward(x)` |
| nn/Linear.hpp:12 / nn/Linear.cpp:42 | `class Linear` | `y = x @ W + b`；`weight: [in, out]`、`bias: [out]`，按 `U(-1/√fan_in, +1/√fan_in)` 初始化（seed 固定可复现） |
| nn/ReLU.hpp:9 / nn/ReLU.cpp:5 | `Tensor ReLUctant(const Tensor&)` | `max(x, 0)`，转发到 `Tensor::ReLU()` |
| nn/ReLU.hpp:12 | `class ReLU` | 把 `ReLUctant` 包成一层（无参） |
| nn/SiLU.hpp:9 / nn/SiLU.cpp:5 | `Tensor Silu(const Tensor&)` | `x · σ(x)`，用 `mul` + `sigmoid` 组合出来 |
| nn/SiLU.hpp:11 | `class SiLU` | 无参层 |
| nn/GELU.hpp:13 / nn/GELU.cpp:7 | `Tensor Gelu(const Tensor&)` | **tanh 近似**：`0.5x(1+tanh(√(2/π)(x+0.044715x³)))` |
| nn/GELU.hpp:15 | `class GELU` | 无参层 |
| nn/LayerNorm.hpp:15 / nn/LayerNorm.cpp:47 | `class LayerNorm` | 对末尾 k 维标准化再仿射 |
| nn/LayerNorm.hpp:24-28 | `LayerNorm(128)` / `LayerNorm({3,4})` / `LayerNorm(vec)` | 三种写法都对：`γ` 初值全 1、`β` 初值全 0 |
| nn/LayerNorm.hpp:31 / nn/LayerNorm.cpp:87 | `forward` / `parameters()` | 关掉 `elementwise_affine` 时 `parameters()` 返回空 |
| nn/Dropout.hpp:15 / nn/Dropout.cpp:16 | `class Dropout` | `Dropout(p = 0.5, seed)`；推理模式恒等 |
| nn/Dropout.hpp:19 | `explicit Dropout(float p, uint32_t seed)` | `p` 必须在 `[0, 1)`；每个实例自带随机状态 |
| nn/AvgPool2d.hpp:23 / nn/AvgPool2d.cpp:44 | `class AvgPool2d` | `[N,C,H,W] -> [N,C,OH,OW]`；前向 = `im2col`(填 0) → `reshape` → `mean(2)` → `reshape`，**零反向代码**（mean 的反向把梯度均分给窗口内每个位置） |
| nn/AvgPool2d.hpp:27 | `AvgPool2d(kernel_size, stride = 0, padding = 0, count_include_pad = true)` | `stride = 0` 跟 kernel 一样；`count_include_pad = false` 时分母只算落在输入里的位置（PyTorch 的参数） |
| nn/Embedding.hpp:19 / nn/Embedding.cpp:50 | `class Embedding` | `weight[V, D]` + 整数下标 → 下标形状 + `[D]`。本体是一次 `index_select(0, ids)`，所以**零反向代码**：只有被查到的行拿到梯度，同一行查多次就累加 |
| nn/Embedding.hpp:26 | `Embedding(num_embeddings, embedding_dim, seed = 12345)` | 权重 `U(-1,1)`（PyTorch 用 N(0,1)，本库没有正态 RNG）；`forward` 接整数下标的 float 张量，另给了 `lookup(vector<int64_t>)` |
| nn/CrossEntropyLoss.hpp:17 / nn/CrossEntropyLoss.cpp:22 | `class CrossEntropyLoss` | 把 `Tensor::cross_entropy` 包成层，**能坐在 `Sequential` 末尾**：网络前向直接得到标量损失。目标固定在层里（`set_target` 可换） |
| nn/MSELoss.hpp:13 / nn/MSELoss.cpp:23 | `class MSELoss` | 同上，包的是 `Tensor::mse`。`Tensor` 没有默认构造函数，所以目标必须构造时给 |
| nn/Conv2d.hpp:20 / nn/Conv2d.cpp:69 | `class Conv2d` | `[N,C,H,W] -> [N,O,OH,OW]`。前向只有四步：`im2col` → `weight.reshape({O,C·k·k}).matmul(col)` → 广播加 `bias` → `reshape`；**层里零反向代码** |
| nn/Conv2d.hpp:26 | `Conv2d(in_channels, out_channels, kernel_size, stride = 1, padding = 0, seed = 12345)` | 参数按 `U(±1/√(C·k·k))` 初始化（PyTorch 默认 kaiming 的等价形式），seed 固定可复现；kernel/stride/padding 目前两个方向相同 |
| nn/Conv2d.hpp:36-37 | `Tensor weight` / `bias` | `weight: [O, C, k, k]`（**和 PyTorch 同布局**，可直接对照）、`bias: [O]`；`parameters()` 返回这两个 |
| nn/Conv2d.hpp:33 | `int64_t output_height(h)` / `output_width(w)` | `(h + 2p - k)/s + 1`，算下一层形状用 |
| nn/BatchNorm.hpp:31 / nn/BatchNorm.cpp:62 | `class BatchNorm` | 逐通道归一化。`[N,C,...]` 都吃：`[N,C]` 等价 BatchNorm1d、`[N,C,H,W]` 等价 BatchNorm2d；统计量取自**通道维以外的所有维** |
| nn/BatchNorm.hpp:35 | `BatchNorm(num_features, eps = 1e-5, momentum = 0.1, affine = true)` | 和 PyTorch 同签名、同默认值 |
| nn/BatchNorm.cpp:80 | 训练分支 | μ/σ² 用当前 batch（**带计算图**，反向要算 ∂μ/∂x、∂σ²/∂x）；同时按滑动平均更新 `running_*`（detach 掉、不参与反向） |
| nn/BatchNorm.cpp:74 | 推理分支 | 直接用 `running_mean` / `running_var`，不再更新 |
| nn/BatchNorm.hpp:47-50 | `weight` / `bias` / `running_mean` / `running_var` | γ 初值 1、β 初值 0、running_mean 初值 0、running_var 初值 1；**只有 γ/β 进 `parameters()`**，running_* 是统计量 |
| nn/BatchNorm.cpp:26 | `reduce_other_dims` / `channel_broadcast_shape` | 对"除通道维外的所有维"求均值；把 `[C]` 变成 `[1,C,1,...]`（直接拿 `[C]` 广播会错位到最后一维） |
| nn/MaxPool2d.hpp:27 / nn/MaxPool2d.cpp:37 | `class MaxPool2d` | `[N,C,H,W] -> [N,C,OH,OW]`；前向 = `im2col`(填 `-inf`) → `reshape({N,C,k*k,L})` → `max(2)` → `reshape`，**零反向代码**（`max` 的反向把梯度送回窗口里的最大值） |
| nn/MaxPool2d.hpp:31 | `MaxPool2d(kernel_size, stride = 0, padding = 0)` | `stride = 0` 表示"跟 kernel 一样"（PyTorch 的 `stride=None`）；无参数 |
| nn/MaxPool2d.cpp:50 | 填充值 | **必须是 `-inf` 不能是 0**：否则窗口里全是负数时 0 会当上最大值。为此给 `im2col` 加了 `pad_value` 参数（卷积仍用默认的 0） |
| nn/Flatten.hpp:16 / nn/Flatten.cpp:29 | `class Flatten` | 把 `[start_dim, end_dim]` 合并成一维；默认 `[N,C,H,W] -> [N,C*H*W]`（等价 PyTorch `nn.Flatten`），无参数，前向就是一次 `reshape` |
| nn/Sequential.hpp:27 / nn/Sequential.cpp:5 | class Sequential | 顺序容器：逐层前向 |
| nn/Sequential.hpp:32 | `explicit Sequential(Layers&&...)` | `Sequential(Linear(3,8), ReLU(), Linear(8,2))` |
| nn/Sequential.hpp:40 / :48 | `void add(...)` | 加层（值/右值走 `make_shared`；`shared_ptr` 重载则共享所有权） |
| nn/Sequential.hpp:57 / nn/Sequential.cpp:15 | `parameters()` | 汇总所有层的参数（顺序 = 层顺序） |
| nn/Sequential.hpp:59-60 | `size()` / `operator[]` | 层数 / 取层（返回 `Module&`，要具体类型就 `dynamic_cast`） |

三个实现在细节上值得看一眼：

**① 权重布局故意和 PyTorch 相反**：PyTorch 存 `[out, in]`、前向写 `x @ W.T`；本库没有转置视图（`transpose` 真搬数据），所以存 `[in, out]`，前向就是一次干净的 `x.matmul(weight)`。副作用是训练出来的 `weight` 能和真值**直接逐元素比**。

**② GELU 用的是 tanh 近似**，因为张量库现在没有 `erf` 算子。要精确式 `0.5x(1+erf(x/√2))` 就得给 `Tensor` 加一个 `erf`/`gelu` 算子（你可以自己加）。两者实测最大差 **4.1e-04**，日常够用。

**③ Dropout 是 inverted dropout**：训练时按 `p` 置零，存活元素放大 `1/(1-p)`，所以输出的期望等于输入、推理时不需要任何补偿。实现只用一个 0 / `1/(1-p)` 的掩码张量做逐元素乘，梯度自然就是 `dx = g·mask`（又是零反向代码）。

### 3.3 用法

```cpp
#include "nn/LayerNorm.hpp"
#include "nn/GELU.hpp"
#include "nn/Dropout.hpp"
#include "nn/Linear.hpp"
#include "nn/Sequential.hpp"
#include "optimizer/Adam.hpp"
#include "optimizer/SGD.hpp"

Sequential net(Linear(2, 16), LayerNorm(16), GELU(), Dropout(0.1f), Linear(16, 1));
Adam opt(net.parameters(), 0.03f);        // 容器汇总所有层的参数

for (int i = 0; i < 3000; ++i)
{
    opt.zero_grad();
    Tensor loss = net.forward(X).bce_with_logits(Y);   // 训练：Dropout 开着
    loss.backward();
    opt.step();
}
net.eval();                                // 推理：Dropout 关掉，结果确定
Tensor logits = net.forward(X);
// 实测：XOR 的 loss 3e-05，eval 下 4/4 点全对
```

### 3.4 验证（48 项全过）

| 组 | 检查 |
|---|---|
| train/eval（8 项） | 默认训练模式；`eval()` 把自己和**子层**都切过去；`train()` / `train(false)`；事后 `add` 的层跟随当前模式；再 `train()` 时新层一起切回来 |
| SiLU（4 项） | `silu(x) == x·σ(x)`（手算）；层 == 函数；无参数；梯度与有限差分一致（6.7e-06） |
| GELU（5 项） | 与 tanh 近似式逐元素一致；**与精确 erf 版最大差 4.1e-04**；两端行为 `GELU(-6)≈0`、`GELU(5)≈5`；层 == 函数；梯度与差分一致（1.9e-05） |
| LayerNorm（12 项） | 形状不变；`γ=1,β=0` 时每行均值≈0（1.5e-08）、方差≈1（7.9e-06）；`γ=2,β=3` 时均值 3.00000、标准差 1.99999；`{3,4}` 多维持标准化；`dL/dx`、`dL/dγ`、`dL/dβ` 与有限差分一致；`affine=false` 时无参数；不匹配/非法 shape/`eps≤0` 都抛异常；三种构造写法不歧义、不误绑 |
| Dropout（11 项） | p=0.5 时置零比例 0.4940、存活值精确放大到 `x/(1-p)`；输出均值≈输入均值（2.0000 vs 2.0240）；`eval` 恒等（0 个置零）；同实例连续调用掩码不同、同 seed 相同、换 seed 不同；`p=0` 恒等；梯度 = `g·mask`；`p=1`/`p>1`/`p<0` 抛异常 |
| 端到端（5 项） | `Linear+LayerNorm+GELU+Dropout` 解 XOR：loss 0.798 → **3e-05**；`eval` 下 4/4 全对且**两次前向完全一致**；`train()` 后 Dropout 又打开；另外 ReLU / SiLU / GELU 三种激活各训一次都收敛到 1e-06 量级 |
| **AvgPool2d / Embedding / 损失层 / 优化器拆分（36 项）** | AvgPool2d：5 种配置（含 `count_include_pad` 两种）与**自己另写的参考实现**逐个一致（最大差 0），梯度 = 1/窗口元素数；Embedding：查表逐行对上 weight（含重复下标）、`[L]→[L,D]` 与 `[N,L]→[N,L,D]`、**梯度只落在被查的行且重复下标累加（2 和 1）**、非整数/越界下标抛异常；`index_select` 沿 dim 0/1/2 与手算一致；损失层：与 `Tensor::cross_entropy` / `Tensor::mse` 数值相同、返回 0 维标量、无参数、没设目标抛异常、**能塞进 Sequential 端到端训练（loss 0.92 → 0.004）**；优化器拆分后 SGD 与 Adam 各自单独可用 |
| **MaxPool2d（21 项）** | 5 种配置（k / stride / padding / 默认 stride）前向与**自己另写的直接窗口最大值实现**逐个一致（最大差 0）；**全负输入 + padding 时输出仍全为负**（证明 -inf 填充生效，用 0 会错）；反向只在各窗口最大值处拿到梯度，有限差分一致（2.5e-05）；`stride=0` 用 kernel、非法参数抛异常；**端到端 `Conv → BN → ReLU → MaxPool → ReLU → Flatten → Linear` 收敛（loss 2.3e-05，acc 1.0000）** |
| **BatchNorm（25 项）** | 训练模式逐通道输出均值≈0/方差≈1（`[N,C]` 与 `[N,C,H,W]` 都验了）；γ=2,β=5 时均值 5、标准差 2；**`running_*` 的滑动平均与手算逐元素一致，含"归一化用有偏方差、running_var 用无偏方差"这个易错点**（momentum=0.1 一步后 running_var = 0.9+0.1·无偏）；推理模式两次前向完全一致且不再改动 `running_*`；`dL/dx`、`dL/dγ`、`dL/dβ` 与有限差分一致（6.1e-05 / 1.7e-05 / 6.3e-06，**含穿过 batch 统计量的那部分**）；`affine=false` 时 `parameters()` 为空；1 维输入 / 通道不匹配 / 训练时每通道只有 1 个值 / 非法 eps / 非法 momentum 都抛异常，推理模式下 N=1 允许；**端到端：`Conv2d → BatchNorm → ReLU → Linear` 训练收敛（loss 0.7094 → 1.1e-05，eval acc 1.0000）** |
| **Conv2d / im2col（38 项）** | im2col 布局与手算逐元素一致（覆盖 k / stride / padding 与多通道，含"同一 patch 在 col 里按 k 分行"这个易错点）；`dL/dx` 穿过 col2im 与有限差分一致（2.1e-04，k=3/s=1 有重叠，验证了重叠累加）；**Conv2d 前向与自己另写的朴素卷积在 6 种配置下最大差 ~1e-7**（k=1/2/3/5、s=1/2、p=0/1/2、C=1/2/3）；`dL/dW`、`dL/db`、`dL/dx` 与差分一致（4.0e-05 / 1.1e-05 / 1.8e-05）；weight 梯度形状保持 4 维 `[O,C,k,k]`（reshape 的反向在工作）；初始化范围 / 同 seed 可复现 / 各种非法参数都抛异常；**端到端：卷积网学会区分竖条与横条（loss 0.69 → 6.1e-05，acc 1.0000）** |
| Linear/ReLU/Sequential（24 项） | 见之前的记录：形状、初始化、手算前向、有限差分、参数汇总顺序、RTTI、`forward` 等价手工逐层调用、空容器、`add(shared_ptr)` 共享而非拷贝、XOR 对照实验（纯线性停在 ln2） |

> **层的地址必须稳定**：`Sequential` 用 `shared_ptr` 持有层，所以 `parameters()` 交给优化器的 `Tensor*` 不会因为容器增长而失效（如果按值存进 `vector`，扩容就会把指针搬走）。反过来说，**容器要活得比优化器久**。

> **most vexing parse 提醒**：只传一个临时对象时 `Sequential net(ReLU());` 会被编译器当成函数声明。用 `Sequential net{ReLU()};` 或 `Sequential net(ReLU(), Linear(2,1));` 就行。

---

## 四、optimizer/（优化器）

优化器持有**参数指针**，`step()` 时用 `Tensor::sub_inplace` 就地更新参数（不建图）。参数的生存期必须长于优化器。

| 位置 | 签名 | 作用 |
|---|---|---|
| optimizer/SGD.hpp:15 | `class SGD` | 随机梯度下降（支持动量 / 权重衰减 / Nesterov） |
| optimizer/SGD.hpp:18 | `SGD(params, lr, momentum = 0, weight_decay = 0, nesterov = false)` | 构造；`momentum != 0` 时会按参数形状分配动量缓冲 |
| optimizer/SGD.cpp:22 | `void step()` | 一步更新：`g = grad + wd·p`；`v = mom·v + g`；`g' = nesterov ? g + mom·v : v`；`p -= lr·g'` |
| optimizer/SGD.cpp:52 | `void zero_grad()` | 把所有参数的梯度槽清空（等价于逐个调 `p->zero_grad()`） |
| optimizer/Adam.hpp:16 | `class Adam` | Adam（Kingma & Ba, 2015） |
| optimizer/Adam.hpp:19 | `Adam(params, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0)` | 构造；按参数形状分配一阶/二阶矩缓冲 |
| optimizer/Adam.cpp:21 | `void step()` | `m = b1·m + (1-b1)·g`；`v = b2·v + (1-b2)·g²`；偏差修正后 `p -= lr·m̂/(√v̂+eps)` |
| optimizer/Adam.cpp:54 | `void zero_grad()` | 同上 |
| tensor/Tensor.hpp:39 | `void Tensor::sub_inplace(const Tensor& other, float scale = 1)` | **全库唯一的就地操作**：`this -= scale * other`，直接改 `data_`。只给优化器用 |
| tensor/Tensor.hpp:60 | `static Tensor::scalar(float v)` | 造 0 维张量（优化器里到处要和标量广播） |

**为什么就地更新不会污染计算图**：图里保存的是输入的**副本**（`inputs` 是 `std::vector<Tensor>` 按值存），而 `data_` 是深拷贝。所以 `opt.step()` 改了参数之后，已经建好的那张图仍然拿着旧值——这是值语义带来的意外好处（PyTorch 里就地改参数是会破坏图的，得靠 `p.data` 绕开）。

**怎么验证的**：
- 用"处方梯度"（`loss = (p ⊙ g).sum()` 让 `dL/dp` 恰好等于 `g`）逐驱动 7 种配置 × 8 步，和 **NumPy 独立实现的递推公式**逐步对比：**56 次全部一致**（覆盖 `lr`、动量、Nesterov、权重衰减、Adam、Adam+wd、自定义 eps）。
- 端到端训练一个线性回归（`y = X @ w_true + b_true`，16 样本 × 3 特征，从全 0 开始）：SGD 与 Adam 都**精确学回真值**（`max|Δw| < 1e-6`，loss 从 2.75 降到 ~0）。

---

## 五、main.cpp（根目录）

| 行号 | 名字 | 作用 |
|---|---|---|
| main.cpp:8 | `static void show(const char* tag, const Tensor& t)` | 打印一行 `tag  shape=[...]  data=[...]`，仅用于演示/肉眼核对 |
| main.cpp:24 | `int main()` | 依次演示：逐元素广播 → matmul → 一元/形状算子 → 整体归约 → softmax → 工厂函数 → 损失函数 → **手写算子的训练示例（SGD/Adam）** → **`nn.Linear` 的训练示例** → 错误处理 |

---

## 六、Python 绑定（pybind11）

把整个库编成 CPython 扩展 `_minitorch`，再由 `minitorch/` 包转出去，于是能在 Python 里用同一套 C++ 实现：

```python
import minitorch as mt

x = mt.Tensor([1.0, 2.0, 3.0], [3])
x.requires_grad = True
loss = x.mul(x).sum()
loss.backward()
print(x.grad.data)                    # [2, 4, 6]

with mt.no_grad():                    # 作用域内不建图
    y = mt.Tensor.ones([2, 2]) * 3.0

net = mt.Sequential(mt.Linear(2, 8), mt.ReLU(), mt.Linear(8, 1))
opt = mt.Adam(net.parameters(), lr=0.05)
```

| 文件 | 作用 |
|---|---|
| [bindings.cpp](bindings.cpp) | 全部绑定：Tensor / no_grad / nn 各层 / SGD·Adam |
| [minitorch/\_\_init\_\_.py](minitorch/__init__.py) | 把 `_minitorch` 的符号转出去，让 `import minitorch` 能用 |
| [setup.py](setup.py) | 用 MSVC + pybind11 编扩展（`packages=["minitorch"]`） |
| [pyproject.toml](pyproject.toml) | 构建依赖：`setuptools>=64` / `wheel` / `pybind11>=3.0` |
| [test_bindings.py](test_bindings.py) | 85 项冒烟测试，跑完输出 `ALL PASS` |
| [CMakeLists.txt](CMakeLists.txt) | 备选构建路径（**暂未验证**，日常用 setup.py 就够了） |

**构建与运行**（要的是装了 pybind11 的那个解释器，这里是 conda 的 `dog` 环境）：

```bat
:: 原地构建：产出 _minitorch.cp310-win_amd64.pyd 到仓库根目录
D:\ProgramData\Anaconda3\envs\dog\python.exe setup.py build_ext --inplace

:: 跑冒烟测试
D:\ProgramData\Anaconda3\envs\dog\python.exe test_bindings.py

:: 想让别处也能 import：装进环境（-e 是"改完 C++ 重新 build 即可，不用再装"）
D:\ProgramData\Anaconda3\envs\dog\python.exe -m pip install -e .
```

- 环境：Python 3.10 + pybind11 3.1 + MSVC（VS 2026 Community）。`setup.py` 里的 `/utf-8`、`/bigobj` 是 **MSVC 参数**，改用 MinGW 要换成 `-finput-charset=UTF-8`、`-Wa,-mbig-obj` 之类。
- 改了任何 `.cpp` 之后都要重新 `build_ext`（扩展是静态编进去的）。

**暴露了什么**

| 分组 | 内容 |
|---|---|
| Tensor | 构造（`shape` 可省 → 0 维标量）、工厂（`zeros/ones/eye/arange/one_hot/scalar`）、算术、比较、`where/clamp/sign`、一元、归约（整体 + 按维 + `keepdim`）、`argmax/argmin/var/std`、形状（`reshape/flatten/unsqueeze/squeeze/concat/transpose/im2col/index_select`）、损失（`mse/cross_entropy×2/bce_with_logits/bce`）、`backward/zero_grad/detach/sub_inplace` |
| 属性 | `shape`、`data`（展平 list）、`grad`（Tensor 或 `None`）、`requires_grad`（可写）、`is_leaf` |
| 运算符 | `+ - * / @ ** -x`，右操作数**可以是数字**（内部转成 0 维标量）；支持 `x + 1.0`、`2.0 * x` |
| 计算图开关 | `is_grad_enabled()` / `set_grad_enabled(bool)` / `with no_grad():`（上下文管理器） |
| nn | `Module`（`forward` / `__call__` / `parameters` / `zero_grad` / `train` / `eval` / `training`）、`Linear`、`Conv2d`、`MaxPool2d`、`AvgPool2d`、`BatchNorm`、`LayerNorm`、`Dropout`、`Flatten`、`Embedding`、`ReLU`/`SiLU`/`GELU`、`CrossEntropyLoss`、`MSELoss`、`Sequential`（`len` / 索引 / 负索引 / `add`） |
| optimizer | `SGD(params, lr, momentum, weight_decay, nesterov)`、`Adam(params, lr, betas, eps, weight_decay)`，都有 `step()` / `zero_grad()` |

**绑定层的几个约定**（和 C++ 略有不同，都是为了让 Python 顺手）

1. **`parameters()` 返回引用，不是拷贝** —— 元素的绑法是 `reference_internal`，所以张量会顺带把层"拉住"，层也不会因为 Python 侧没引用而被提前回收。优化器存的就是这些指针，所以**层必须活得比优化器久**（`net` 和 `opt` 放在同一作用域即可）。
2. **`var` / `std` 用 `unbiased=True` 的 bool**，内部再转成 C++ 的 `VarCorrection`（Python 侧不用碰那个枚举）。
3. **`is_leaf` / `training` / `shape` / `data` / `grad` 都是属性**（`grad` 是拷贝，`zero_grad()` 之后不会悬空）；`item()` 还是方法。
4. 没有 `__getitem__` / 切片 —— 库本身没有索引视图，取元素请用 `data` 或 `reshape` 后再取。

**验证**：`test_bindings.py` 的 85 项覆盖张量运算与手算对拍、`dL/dw = 2w`、matmul 双向梯度、`no_grad` 的四种行为、12 个层的前向/属性、损失层与 `Sequential`（含负索引、越界 `IndexError`）、用 Python 端到端解 XOR（loss 0.31 → 0.003，acc 1.0）、以及"层被回收后参数张量仍可用"的生命周期检查 —— 全部通过。

---

## 七、读代码前必看的约定

1. **广播**：NumPy 风格——从最后一维右对齐，每对相等或其中一个为 1；结果取较大者。缺失的维视为 1。标量（`shape = {}`）与任何形状兼容。
2. **0 维标量**：`Tensor({2.5f}, {})`，`numel({}) == 1`。所有整体归约返回这种形状，所以 `x - x.mean()` 这类写法天然可广播。
3. **归约默认丢维**：`sum(1)` 把 `[B,C]` 变成 `[B]`；要保留成 `[B,1]` 用 `sum(1, true)`。`softmax`/`log_softmax` 内部用的就是 `keepdim = true`。
4. **值语义**：算子返回**新张量**，没有视图——`reshape`/`unsqueeze`/`squeeze`/`transpose` 都会复制整个数据（因为还没有 stride/offset）。唯一的就地操作是 `sub_inplace`，只给优化器用。
5. **错误处理**：一律 `throw std::runtime_error("算子名: 原因")`，例如 `matmul: inner dim mismatch`、`softmax: dim out of range`、`one_hot: target out of range`。
6. **dtype 只有 float32**；形状这类整数信息用 `std::vector<int64_t>` 传递，不进张量。**唯一的例外是 `argmax` / `argmin`**：它们把下标放进张量（float 存整数），这样就能直接和 `eq` / `sum` / `mean` 组合算准确率：`net(X).argmax(1).eq(y).mean().item()`。
7. **`*` 与 `matmul`**：`a * b` 是逐元素乘（和 PyTorch 一致），矩阵乘必须写 `a.matmul(b)`。
8. **autograd 的三条使用约束**（值语义带来的）：
   - `requires_grad` 要在**参与运算之前**设置好——它是普通 `bool`，只按当时的值被拷进计算图；事后再改原张量不会影响已经建好的图。
   - **拷贝 = 同一个变量**：`Tensor y = x;` 之后 `y` 和 `x` 的**数据和梯度都是分开的**（`data_` 是深拷贝），但**梯度槽是共享的**（`meta_` 是 `shared_ptr`）——所以反向时往图里那份副本累加的梯度，你用 `x.grad()` 也能读到。想要一份完全独立的副本用 `detach()`。
   - **所有算子都会建图**（见 1.11），梯度不会静默断开。

---

## 八、快速上手

```cpp
#include "Tensor.hpp"
#include <cstdio>

int main()
{
    // 1) 逐元素广播：{2,3} + {3} -> {2,3}
    const Tensor a({1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor b({10, 20, 30}, {3});
    const float c0 = (a + b).data()[0];

    // 2) 线性层：y = x @ w + bias（bias 广播到每一行）
    const Tensor x({1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor w({1, 0, 0, 1, 1, 1}, {3, 2});
    const Tensor bias({10, 20}, {2});
    const Tensor y = x.matmul(w) + bias;              // [2,3] @ [3,2] + [2] -> [2,2]

    // 3) 多分类交叉熵（整数类别标签）
    const Tensor logits({0.5f, -1.5f, 2.0f, 3.0f, 0.0f, -0.5f}, {2, 3});
    const float ce = logits.cross_entropy(std::vector<int64_t>{0, 2}).data()[0];

    // 4) 二分类（在 logits 上算，数值最稳）
    const Tensor lg({-8, -3, 3, 8}, {2, 2});
    const Tensor tg({1, 0, 0, 1}, {2, 2});
    const float bce = lg.bce_with_logits(tg).data()[0];

    std::printf("(a+b)[0]=%g  y[0]=%g  y=%lldx%lld  ce=%g  bce=%g\n",
                c0, y.data()[0], (long long)y.shape()[0], (long long)y.shape()[1],
                ce, bce);
    return 0;
}
```

输出（已实测）：

```
(a+b)[0]=11  y[0]=14  y=2x2  ce=2.65137  bce=2.77446
```

运行 `main.exe` 还能看到每一类算子的完整输出（含形状和数据），例如：

```
=== losses ===
lg.cross_entropy(onehot) shape=[]     data=[2.65137]
lg.cross_entropy({0,2})  shape=[]     data=[2.65137]     ← one-hot 加权求和 == 整数下标版
xb.bce_with_logits(yb)   shape=[]     data=[2.77446]
xb.sigmoid().bce(yb)     shape=[]     data=[2.77446]     ← logits 版与概率版一致到 5 位小数
```

**用 `nn.Linear` 写更省事**（下面的手写版只是为了展示底层算子）：

```cpp
#include "nn/Linear.hpp"
#include "optimizer/Adam.hpp"
#include "optimizer/SGD.hpp"

Linear lin(3, 2);                       // [in, out]，参数自动初始化并带上 requires_grad
SGD opt(lin.parameters(), 0.1f, 0.9f);

for (int i = 0; i < 600; ++i)
{
    opt.zero_grad();
    Tensor loss = lin.forward(X).mse(target);   // X: [N,3] -> [N,2]
    loss.backward();
    opt.step();
}
// 实测：loss 1.879 -> ~0，weight/bias 与真值完全一致
```

**搭多层网络**（`nn.Sequential` + `nn.ReLU`，最上面那段 XOR 演示就是这个）：

```cpp
#include "nn/Linear.hpp"
#include "nn/ReLU.hpp"
#include "nn/Sequential.hpp"

Sequential net(Linear(2, 8), ReLU(), Linear(8, 1));   // 层按顺序串起来
Adam opt(net.parameters(), 0.05f);                    // 容器汇总所有参数，直接喂优化器
// 实测：XOR 的 loss 0.7077 -> 0.00004，4/4 点全对

Sequential deep(Linear(2, 16), LayerNorm(16), GELU(), Dropout(0.1f), Linear(16, 1));
deep.eval();   // 推理：Dropout 关掉，前向结果确定
deep.train();  // 训练：Dropout 打开
```

**等价的手写版**（`main.exe` 里那段，展示底层算子怎么串）：

```cpp
#include "optimizer/Adam.hpp"
#include "optimizer/SGD.hpp"
#include "Tensor.hpp"

// 目标：y = X @ w_true + b_true
const Tensor X({0.1f, -0.8f, 0.5f,  0.6f, 0.2f, -0.7f,
                -0.4f, 0.9f, 0.3f,  0.2f, -0.1f, 0.8f}, {4, 3});
const Tensor target = X.matmul(Tensor({0.5f, -1.0f, 2.0f}, {3})).add(Tensor({0.3f}, {}));

Tensor w = Tensor::zeros({3});   w.requires_grad = true;
Tensor b = Tensor::zeros({});    b.requires_grad = true;

SGD opt({&w, &b}, 0.05f, /*momentum=*/0.9f);
for (int i = 0; i < 300; ++i)
{
    opt.zero_grad();                                  // 1) 清梯度
    Tensor loss = X.matmul(w).add(b).mse(target);     // 2) 前向
    loss.backward();                                  // 3) 反向
    opt.step();                                       // 4) 更新参数
}
// 实测：loss 2.518 -> ~0，学到的 w/b 与真值完全一致
```

---

## 九、尚未实现 / 后续计划

**底层能力**
- `strides_` + `offset_`（视图）：让 `reshape`/`unsqueeze`/`squeeze`/`transpose` 零拷贝，也是 `broadcast_to`、`permute` 的地基
- 整数索引：`gather` / `index_select`（现在的整数标签交叉熵是靠 `one_hot` + 加权求和绕过去的，`C` 很大时费内存）
- `numel` / `reshape` 的整数溢出检查

**工程**
- 常驻测试：现在只有 `main.cpp` 的打印式演示，没有 assert 式回归测试（开发过程中用的是临时的"有限差分对拍"脚本，每次改反向都会重跑）
- 构建系统：现在是 `g++ tensor\*.cpp nn\*.cpp optimizer\*.cpp main.cpp`；文件再多可以上 Makefile / CMake + Ninja（本机都有）
- `namespace minipt`（现在类在全局命名空间，内部工具在 `tensor_detail`）

**功能**
- **autograd 可用**：所有算子都有反向，`backward()` + 逆拓扑遍历 + 梯度累加都能跑。已知的粗糙点：
  - `prod` 的反向在 `x_i = 0` 时给 inf/nan
  - `max`/`min` 并列极值时梯度会重复给每个并列项
  - 反向里会重算 `exp`/`sigmoid`/`softmax`（为了避开 `shared_ptr` 环，见 1.11）
- **`nn/` 已有**：`Module`（含 `train()`/`eval()` 模式）、`Linear`、`ReLU` / `SiLU` / `GELU`、`LayerNorm`、`Dropout`、`Sequential`。待补：
  - 更多薄封装：`Sigmoid` / `Tanh` / `Softmax` / `Flatten`（照 `ReLU` 那样，一个函数 + 一层）
  - 自适应池化、空洞卷积 / 分组卷积、`BatchNorm` 的 `track_running_stats=false` 变体
  - 损失层（把 `cross_entropy` / `mse` 包成 `Module`，就能塞进 `Sequential`）
  - 网络保存/加载、`Module` 结构的打印（`to_string`）
- 训练配套：dataloader / 分 batch、学习率调度、梯度裁剪
