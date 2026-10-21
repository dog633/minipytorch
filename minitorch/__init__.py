"""mini-pytorch 的 Python 包。

真正的实现是 C++ 扩展 ``_minitorch``（由 ``bindings.cpp`` + ``setup.py`` 编出来），
这里只是把它的符号原样导出，让 ``import minitorch`` 能直接用：

    import minitorch as mt

    x = mt.Tensor([1.0, 2.0, 3.0], [3])
    x.requires_grad = True
    loss = x.mul(x).sum()
    loss.backward()
    print(x.grad.data)          # [2, 4, 6]

    with mt.no_grad():          # 作用域内不建计算图
        y = mt.Tensor.ones([2, 2]) * 3.0

    net = mt.Sequential(mt.Linear(2, 8), mt.ReLU(), mt.Linear(8, 1))
    opt = mt.Adam(net.parameters(), lr=0.05)
"""

from _minitorch import *        # noqa: F401,F403
from _minitorch import __version__  # noqa: F401
