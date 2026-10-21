// pybind11 绑定：把 mini-pytorch 暴露成 Python 模块 `_minitorch`。
//
// 分四块：
//   1) Tensor：构造 / 工厂 / 运算 / 归约 / 形状 / 一元 / 损失 / autograd
//   2) 计算图开关：is_grad_enabled / set_grad_enabled / with no_grad()
//   3) nn：Module 基类 + 各个层（都用 shared_ptr 持有，才能塞进 Sequential）
//   4) optimizer：SGD / Adam
//
// 关于生命周期的约定（和 C++ 一样）：层拥有参数张量，所以
// **层必须活得比它的参数张量、以及持有这些指针的优化器久** ——
// `net.parameters()` 返回的张量是用 reference_internal 绑定的，会顺带keep住 net。
#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nn/AvgPool2d.hpp"
#include "nn/BatchNorm.hpp"
#include "nn/Conv2d.hpp"
#include "nn/CrossEntropyLoss.hpp"
#include "nn/Dropout.hpp"
#include "nn/Embedding.hpp"
#include "nn/Flatten.hpp"
#include "nn/GELU.hpp"
#include "nn/LayerNorm.hpp"
#include "nn/Linear.hpp"
#include "nn/MSELoss.hpp"
#include "nn/MaxPool2d.hpp"
#include "nn/Module.hpp"
#include "nn/ReLU.hpp"
#include "nn/Sequential.hpp"
#include "nn/SiLU.hpp"
#include "optimizer/Adam.hpp"
#include "optimizer/SGD.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/TensorGradMode.hpp"

namespace py = pybind11;

namespace
{
    // 二元运算符的右操作数：既可以是 Tensor，也可以是 Python 的 int / float。
    // 后者会变成一个 0 维张量（广播），于是 x * 2 这种写法能直接用在 Python 里。
    Tensor as_tensor(const py::object& obj)
    {
        if (py::isinstance<Tensor>(obj))
        {
            return obj.cast<Tensor>();
        }
        if (py::isinstance<py::bool_>(obj) || py::isinstance<py::int_>(obj)
            || py::isinstance<py::float_>(obj))
        {
            return Tensor::scalar(obj.cast<float>());
        }
        throw py::type_error("expected Tensor or a number");
    }

    // 张量的简洁表示：shape=[2,3] data=[...]
    std::string tensor_repr(const Tensor& t)
    {
        std::string s = "Tensor(shape=[";
        for (size_t i = 0; i < t.shape().size(); ++i)
        {
            s += (i ? ", " : "") + std::to_string(t.shape()[i]);
        }
        s += "], data=[";
        const std::vector<float>& d = t.data();
        for (size_t i = 0; i < d.size(); ++i)
        {
            if (i >= 8)
            {
                s += ", ... (" + std::to_string(d.size()) + " 个元素)";
                break;
            }
            s += (i ? ", " : "") + std::to_string(d[i]);
        }
        s += "])";
        return s;
    }
}  // namespace

PYBIND11_MODULE(_minitorch, m)
{
    m.doc() = "mini-pytorch minimal binding（张量 / autograd / nn 层 / 优化器）";
    m.attr("__version__") = "0.2.0";

    // ---------------- 枚举 ----------------
    py::enum_<VarCorrection>(m, "VarCorrection", "方差/标准差的修正方式")
        .value("unbiased", VarCorrection::unbiased)
        .value("biased", VarCorrection::biased);

    // ---------------- Tensor ----------------
    py::class_<Tensor>(m, "Tensor")
        // 构造：Tensor(data, shape)，shape 省略就是 0 维标量
        .def(py::init<std::vector<float>, std::vector<int64_t>>(),
             py::arg("data"), py::arg("shape") = std::vector<int64_t>{})
        .def("__repr__", &tensor_repr)

        // 工厂
        .def_static("zeros", &Tensor::zeros, py::arg("shape"))
        .def_static("ones", &Tensor::ones, py::arg("shape"))
        .def_static("eye", py::overload_cast<int64_t>(&Tensor::eye), py::arg("n"))
        .def_static("eye", py::overload_cast<int64_t, int64_t>(&Tensor::eye), py::arg("n"), py::arg("m"))
        .def_static("arange", py::overload_cast<float>(&Tensor::arange), py::arg("end"))
        .def_static("arange", py::overload_cast<float, float, float>(&Tensor::arange),
                    py::arg("start"), py::arg("end"), py::arg("step") = 1.0f)
        .def_static("one_hot", &Tensor::one_hot, py::arg("targets"), py::arg("num_classes"))
        .def_static("scalar", &Tensor::scalar, py::arg("v"))

        // 属性 / 访问
        .def_property_readonly("shape", [](const Tensor& t) { return t.shape(); }, "形状（list）")
        .def_property_readonly("data", [](const Tensor& t) { return t.data(); }, "展平后的数据（list）")
        .def_property_readonly("grad", [](const Tensor& t) -> py::object
        {
            const Tensor* g = t.grad();
            return g ? py::cast(*g) : py::none();   // 返回一份拷贝，zero_grad 之后也不会悬空
        }, "梯度（没有就是 None）")
        .def_property("requires_grad",
                      [](const Tensor& t) { return t.requires_grad; },
                      [](Tensor& t, bool v) { t.requires_grad = v; },
                      "是否需要梯度（叶子标记，要在参与运算之前设置）")
        .def_property_readonly("is_leaf", &Tensor::is_leaf, "是不是叶子（没有 grad_fn）")
        .def("item", &Tensor::item, "单元素张量取值")
        .def("__len__", [](const Tensor& t) { return t.shape().empty() ? 0 : t.shape()[0]; })

        // 逐元素算术 + 运算符（参数可以是 Tensor，也可以是数字）
        .def("add", [](const Tensor& a, py::object b) { return a.add(as_tensor(b)); }, py::arg("other"))
        .def("sub", [](const Tensor& a, py::object b) { return a.sub(as_tensor(b)); }, py::arg("other"))
        .def("mul", [](const Tensor& a, py::object b) { return a.mul(as_tensor(b)); }, py::arg("other"))
        .def("div", [](const Tensor& a, py::object b) { return a.div(as_tensor(b)); }, py::arg("other"))
        .def("matmul", [](const Tensor& a, py::object b) { return a.matmul(as_tensor(b)); }, py::arg("other"))
        .def("__add__", [](const Tensor& a, py::object b) { return a.add(as_tensor(b)); }, py::is_operator())
        .def("__sub__", [](const Tensor& a, py::object b) { return a.sub(as_tensor(b)); }, py::is_operator())
        .def("__mul__", [](const Tensor& a, py::object b) { return a.mul(as_tensor(b)); }, py::is_operator())
        .def("__truediv__", [](const Tensor& a, py::object b) { return a.div(as_tensor(b)); }, py::is_operator())
        .def("__matmul__", [](const Tensor& a, py::object b) { return a.matmul(as_tensor(b)); }, py::is_operator())
        .def("__radd__", [](const Tensor& a, py::object b) { return as_tensor(b).add(a); }, py::is_operator())
        .def("__rmul__", [](const Tensor& a, py::object b) { return as_tensor(b).mul(a); }, py::is_operator())
        .def("__neg__", [](const Tensor& a) { return a.neg(); }, py::is_operator())
        .def("__pow__", [](const Tensor& a, float p) { return a.pow(p); }, py::is_operator())

        // 掩码 / 条件（同样接受数字，方便写 x.gt(0)）
        .def("gt", [](const Tensor& a, py::object b) { return a.gt(as_tensor(b)); }, py::arg("other"))
        .def("lt", [](const Tensor& a, py::object b) { return a.lt(as_tensor(b)); }, py::arg("other"))
        .def("ge", [](const Tensor& a, py::object b) { return a.ge(as_tensor(b)); }, py::arg("other"))
        .def("le", [](const Tensor& a, py::object b) { return a.le(as_tensor(b)); }, py::arg("other"))
        .def("eq", [](const Tensor& a, py::object b) { return a.eq(as_tensor(b)); }, py::arg("other"))
        .def("ne", [](const Tensor& a, py::object b) { return a.ne(as_tensor(b)); }, py::arg("other"))
        .def_static("where", &Tensor::where, py::arg("cond"), py::arg("a"), py::arg("b"))
        .def("sign", &Tensor::sign)
        .def("clamp", &Tensor::clamp, py::arg("lo"), py::arg("hi"))
        .def("clamp_min", &Tensor::clamp_min, py::arg("lo"))
        .def("clamp_max", &Tensor::clamp_max, py::arg("hi"))

        // 一元
        .def("neg", &Tensor::neg)
        .def("abs", &Tensor::abs)
        .def("exp", &Tensor::exp)
        .def("log", &Tensor::log)
        .def("sqrt", &Tensor::sqrt)
        .def("pow", &Tensor::pow, py::arg("p"))
        .def("sigmoid", &Tensor::sigmoid)
        .def("relu", &Tensor::ReLU)
        .def("tanh", &Tensor::tanh)
        .def("softmax", &Tensor::softmax, py::arg("dim"))
        .def("log_softmax", &Tensor::log_softmax, py::arg("dim"))

        // 整体归约
        .def("sum", py::overload_cast<>(&Tensor::sum, py::const_))
        .def("prod", py::overload_cast<>(&Tensor::prod, py::const_))
        .def("max", py::overload_cast<>(&Tensor::max, py::const_))
        .def("min", py::overload_cast<>(&Tensor::min, py::const_))
        .def("mean", py::overload_cast<>(&Tensor::mean, py::const_))
        // 按维归约
        .def("sum", py::overload_cast<int64_t, bool>(&Tensor::sum, py::const_),
             py::arg("dim"), py::arg("keepdim") = false)
        .def("prod", py::overload_cast<int64_t, bool>(&Tensor::prod, py::const_),
             py::arg("dim"), py::arg("keepdim") = false)
        .def("max", py::overload_cast<int64_t, bool>(&Tensor::max, py::const_),
             py::arg("dim"), py::arg("keepdim") = false)
        .def("min", py::overload_cast<int64_t, bool>(&Tensor::min, py::const_),
             py::arg("dim"), py::arg("keepdim") = false)
        .def("mean", py::overload_cast<int64_t, bool>(&Tensor::mean, py::const_),
             py::arg("dim"), py::arg("keepdim") = false)

        // 指标
        .def("argmax", py::overload_cast<>(&Tensor::argmax, py::const_))
        .def("argmin", py::overload_cast<>(&Tensor::argmin, py::const_))
        .def("argmax", py::overload_cast<int64_t, bool>(&Tensor::argmax, py::const_),
             py::arg("dim"), py::arg("keepdim") = false)
        .def("argmin", py::overload_cast<int64_t, bool>(&Tensor::argmin, py::const_),
             py::arg("dim"), py::arg("keepdim") = false)
        // var/std：Python 侧用 bool 更顺手，内部转成 VarCorrection
        .def("var", [](const Tensor& t) { return t.var(); })
        .def("var", [](const Tensor& t, bool unbiased)
             { return t.var(unbiased ? VarCorrection::unbiased : VarCorrection::biased); },
             py::arg("unbiased") = true)
        .def("var", [](const Tensor& t, int64_t dim, bool keepdim, bool unbiased)
             { return t.var(dim, keepdim, unbiased ? VarCorrection::unbiased : VarCorrection::biased); },
             py::arg("dim"), py::arg("keepdim") = false, py::arg("unbiased") = true)
        .def("std", [](const Tensor& t) { return t.std(); })
        .def("std", [](const Tensor& t, bool unbiased)
             { return t.std(unbiased ? VarCorrection::unbiased : VarCorrection::biased); },
             py::arg("unbiased") = true)
        .def("std", [](const Tensor& t, int64_t dim, bool keepdim, bool unbiased)
             { return t.std(dim, keepdim, unbiased ? VarCorrection::unbiased : VarCorrection::biased); },
             py::arg("dim"), py::arg("keepdim") = false, py::arg("unbiased") = true)

        // 形状
        .def("reshape", &Tensor::reshape, py::arg("shape"))
        .def("flatten", &Tensor::flatten)
        .def("unsqueeze", &Tensor::unsqueeze, py::arg("dim"))
        .def("squeeze", py::overload_cast<>(&Tensor::squeeze, py::const_))
        .def("squeeze", py::overload_cast<int64_t>(&Tensor::squeeze, py::const_), py::arg("dim"))
        .def("concat", &Tensor::concat, py::arg("other"), py::arg("dim"))
        .def("transpose", &Tensor::transpose, py::arg("dim0"), py::arg("dim1"))
        .def("im2col", &Tensor::im2col, py::arg("kh"), py::arg("kw"), py::arg("stride") = 1,
             py::arg("padding") = 0, py::arg("pad_value") = 0.0f)
        .def("index_select", &Tensor::index_select, py::arg("dim"), py::arg("indices"))

        // 损失
        .def("mse", &Tensor::mse, py::arg("other"))
        .def("cross_entropy", py::overload_cast<const Tensor&>(&Tensor::cross_entropy, py::const_),
             py::arg("targets"))
        .def("cross_entropy", py::overload_cast<const std::vector<int64_t>&>(&Tensor::cross_entropy, py::const_),
             py::arg("targets"))
        .def("bce_with_logits", &Tensor::bce_with_logits, py::arg("targets"))
        .def("bce", &Tensor::bce, py::arg("targets"))

        // autograd + 就地更新
        .def("backward", &Tensor::backward)
        .def("zero_grad", &Tensor::zero_grad)
        .def("detach", &Tensor::detach)
        .def("sub_inplace", &Tensor::sub_inplace, py::arg("other"), py::arg("scale") = 1.0f);

    // ---------------- 计算图开关 ----------------
    m.def("is_grad_enabled", &is_grad_enabled, "当前是否记录计算图");
    m.def("set_grad_enabled", &set_grad_enabled, py::arg("enabled"), "手动开关计算图记录");

    // with minitorch.no_grad(): ...
    py::class_<NoGradGuard>(m, "no_grad", "上下文管理器 / 装饰器：作用域内不建计算图")
        .def(py::init<>())
        .def("__enter__", [](NoGradGuard&) { })
        .def("__exit__", [](NoGradGuard&, py::object, py::object, py::object) { return false; });

    // ---------------- nn：Module 基类 ----------------
    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def("forward", &Module::forward, py::arg("x"))
        .def("__call__", [](Module& self, const Tensor& x) { return self.forward(x); }, py::arg("x"))
        .def("parameters", &Module::parameters, py::return_value_policy::reference_internal,
             "可训练参数（list[Tensor]，引用层内部的张量）")
        .def("zero_grad", &Module::zero_grad)
        .def("train", &Module::train, py::arg("mode") = true)
        .def("eval", &Module::eval)
        .def_property_readonly("training", &Module::is_training);

    // ---------------- nn：具体的层 ----------------
    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear", "全连接层 y = x @ W + b")
        .def(py::init<int64_t, int64_t, uint32_t>(),
             py::arg("in_features"), py::arg("out_features"), py::arg("seed") = 12345u)
        .def_property_readonly("weight", [](Linear& l) -> Tensor& { return l.weight; },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("bias", [](Linear& l) -> Tensor& { return l.bias; },
                               py::return_value_policy::reference_internal);

    py::class_<Conv2d, Module, std::shared_ptr<Conv2d>>(m, "Conv2d", "2D 卷积（im2col 实现）")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, uint32_t>(),
             py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"),
             py::arg("stride") = 1, py::arg("padding") = 0, py::arg("seed") = 12345u)
        .def_property_readonly("weight", [](Conv2d& l) -> Tensor& { return l.weight; },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("bias", [](Conv2d& l) -> Tensor& { return l.bias; },
                               py::return_value_policy::reference_internal)
        .def("output_height", &Conv2d::output_height, py::arg("h"))
        .def("output_width", &Conv2d::output_width, py::arg("w"));

    py::class_<MaxPool2d, Module, std::shared_ptr<MaxPool2d>>(m, "MaxPool2d", "2D 最大池化")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("kernel_size"), py::arg("stride") = 0, py::arg("padding") = 0)
        .def("output_height", &MaxPool2d::output_height, py::arg("h"))
        .def("output_width", &MaxPool2d::output_width, py::arg("w"));

    py::class_<AvgPool2d, Module, std::shared_ptr<AvgPool2d>>(m, "AvgPool2d", "2D 平均池化")
        .def(py::init<int64_t, int64_t, int64_t, bool>(),
             py::arg("kernel_size"), py::arg("stride") = 0, py::arg("padding") = 0,
             py::arg("count_include_pad") = true)
        .def("output_height", &AvgPool2d::output_height, py::arg("h"))
        .def("output_width", &AvgPool2d::output_width, py::arg("w"));

    py::class_<BatchNorm, Module, std::shared_ptr<BatchNorm>>(m, "BatchNorm", "批归一化 [N, C, ...]")
        .def(py::init<int64_t, float, float, bool>(),
             py::arg("num_features"), py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f,
             py::arg("affine") = true)
        .def_property_readonly("weight", [](BatchNorm& l) -> Tensor& { return l.weight; },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("bias", [](BatchNorm& l) -> Tensor& { return l.bias; },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("running_mean", [](BatchNorm& l) -> Tensor& { return l.running_mean; },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("running_var", [](BatchNorm& l) -> Tensor& { return l.running_var; },
                               py::return_value_policy::reference_internal);

    py::class_<LayerNorm, Module, std::shared_ptr<LayerNorm>>(m, "LayerNorm", "层归一化")
        .def(py::init<int64_t, float, bool>(),
             py::arg("normalized_size"), py::arg("eps") = 1e-5f, py::arg("elementwise_affine") = true)
        .def(py::init<const std::vector<int64_t>&, float, bool>(),
             py::arg("normalized_shape"), py::arg("eps") = 1e-5f, py::arg("elementwise_affine") = true)
        .def_property_readonly("weight", [](LayerNorm& l) -> Tensor& { return l.gamma; },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("bias", [](LayerNorm& l) -> Tensor& { return l.beta; },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("gamma", [](LayerNorm& l) -> Tensor& { return l.gamma; },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("beta", [](LayerNorm& l) -> Tensor& { return l.beta; },
                               py::return_value_policy::reference_internal);

    py::class_<Dropout, Module, std::shared_ptr<Dropout>>(m, "Dropout", "Dropout（推理模式恒等）")
        .def(py::init<float, uint32_t>(), py::arg("p") = 0.5f, py::arg("seed") = 20240601u)
        .def_property_readonly("p", &Dropout::p);

    py::class_<Flatten, Module, std::shared_ptr<Flatten>>(m, "Flatten", "摊平层")
        .def(py::init<int64_t, int64_t>(), py::arg("start_dim") = 1, py::arg("end_dim") = -1);

    py::class_<Embedding, Module, std::shared_ptr<Embedding>>(m, "Embedding", "查表层")
        .def(py::init<int64_t, int64_t, uint32_t>(),
             py::arg("num_embeddings"), py::arg("embedding_dim"), py::arg("seed") = 12345u)
        .def_property_readonly("weight", [](Embedding& l) -> Tensor& { return l.weight; },
                               py::return_value_policy::reference_internal)
        .def("lookup", &Embedding::lookup, py::arg("ids"));

    // 无参激活层
    py::class_<ReLU, Module, std::shared_ptr<ReLU>>(m, "ReLU", "ReLU 激活层").def(py::init<>());
    py::class_<SiLU, Module, std::shared_ptr<SiLU>>(m, "SiLU", "SiLU 激活层").def(py::init<>());
    py::class_<GELU, Module, std::shared_ptr<GELU>>(m, "GELU", "GELU 激活层（tanh 近似）").def(py::init<>());

    // 损失层（目标固定在层里，所以能坐在 Sequential 末尾）
    py::class_<CrossEntropyLoss, Module, std::shared_ptr<CrossEntropyLoss>>(m, "CrossEntropyLoss", "交叉熵损失层")
        .def(py::init<>())
        .def(py::init<std::vector<int64_t>>(), py::arg("targets"))
        .def("set_target", &CrossEntropyLoss::set_target, py::arg("targets"));

    py::class_<MSELoss, Module, std::shared_ptr<MSELoss>>(m, "MSELoss", "均方误差损失层")
        .def(py::init<Tensor>(), py::arg("target"))
        .def(py::init<std::vector<float>, std::vector<int64_t>>(), py::arg("data"), py::arg("shape"))
        .def("set_target", &MSELoss::set_target, py::arg("target"))
        .def_property_readonly("target", [](MSELoss& l) -> Tensor& { return const_cast<Tensor&>(l.target()); },
                               py::return_value_policy::reference_internal);

    // 顺序容器：Sequential(Linear(2,8), ReLU(), Linear(8,1))
    py::class_<Sequential, Module, std::shared_ptr<Sequential>>(m, "Sequential", "顺序容器")
        .def(py::init([](py::args layers)
        {
            auto seq = std::make_shared<Sequential>();
            for (py::handle h : layers)
            {
                seq->add(h.cast<std::shared_ptr<Module>>());
            }
            return seq;
        }))
        .def("add", [](Sequential& self, std::shared_ptr<Module> layer) { self.add(std::move(layer)); },
             py::arg("layer"))
        .def("__len__", &Sequential::size)
        .def("__getitem__", [](Sequential& self, py::ssize_t i) -> Module&
        {
            const py::ssize_t n = static_cast<py::ssize_t>(self.size());
            if (i < 0)
            {
                i += n;
            }
            if (i < 0 || i >= n)
            {
                throw py::index_error("Sequential index out of range");
            }
            return self[static_cast<size_t>(i)];
        }, py::return_value_policy::reference_internal);

    // ---------------- 优化器 ----------------
    py::class_<SGD>(m, "SGD", "随机梯度下降（动量 / 权重衰减 / Nesterov）")
        .def(py::init<std::vector<Tensor*>, float, float, float, bool>(),
             py::arg("params"), py::arg("lr"), py::arg("momentum") = 0.0f,
             py::arg("weight_decay") = 0.0f, py::arg("nesterov") = false)
        .def("step", &SGD::step)
        .def("zero_grad", &SGD::zero_grad);

    py::class_<Adam>(m, "Adam", "Adam")
        .def(py::init<std::vector<Tensor*>, float, float, float, float, float>(),
             py::arg("params"), py::arg("lr") = 1e-3f, py::arg("beta1") = 0.9f,
             py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
        .def("step", &Adam::step)
        .def("zero_grad", &Adam::zero_grad);
}
