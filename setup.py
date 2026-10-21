from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

sources = [
    # tensor 核心
    "tensor/TensorArith.cpp",
    "tensor/TensorAutograd.cpp",
    "tensor/TensorConv.cpp",
    "tensor/TensorCore.cpp",
    "tensor/TensorFactory.cpp",
    "tensor/TensorGradMode.cpp",
    "tensor/TensorIndex.cpp",
    "tensor/TensorLoss.cpp",
    "tensor/TensorReduce.cpp",
    "tensor/TensorShape.cpp",
    "tensor/TensorUnary.cpp",
    "tensor/TensorUtil.cpp",
    # nn
    "nn/AvgPool2d.cpp",
    "nn/BatchNorm.cpp",
    "nn/Conv2d.cpp",
    "nn/CrossEntropyLoss.cpp",
    "nn/Dropout.cpp",
    "nn/Embedding.cpp",
    "nn/Flatten.cpp",
    "nn/GELU.cpp",
    "nn/LayerNorm.cpp",
    "nn/Linear.cpp",
    "nn/MaxPool2d.cpp",
    "nn/Module.cpp",
    "nn/MSELoss.cpp",
    "nn/ReLU.cpp",
    "nn/Sequential.cpp",
    "nn/SiLU.cpp",
    # optimizer
    "optimizer/Adam.cpp",
    "optimizer/SGD.cpp",
    # 绑定层
    "bindings.cpp",
]

ext_modules = [
    Pybind11Extension(
        "_minitorch",
        sources,
        include_dirs=[".", "nn", "optimizer", "tensor", "tensor/detail"],
        cxx_std=17,
        extra_compile_args=["/utf-8"], 
    ),
]

setup(
    name="minitorch",
    version="0.2.0",
    # minitorch/__init__.py 把 _minitorch 的符号转出去，这样 import minitorch 就能用
    packages=["minitorch"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)