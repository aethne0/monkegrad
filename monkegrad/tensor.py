from enum import Enum, auto
from math import tanh

import numpy as np

np.random.seed(sum([ord(ch) for ch in "monke"]))
_DEBUG = False


class Op_Enum(Enum):
    Matmul = auto()
    Tanh = auto()


class Op:
    def __init__(self):
        self.args = ()
        self.arity = 0
        self.out = 0.0
        self.grad = 0.0

    def __repr__(self):
        return f"{type(self)}:(out={self.out:.2f}, grad={self.grad:.2f})"


def Op_disp(op_or_nd, indent=0):
    if type(op_or_nd) is not np.ndarray:
        print(" " * indent * 4 + str(op_or_nd))
        for i in op_or_nd.args:
            Op_disp(i, indent + 1)
    else:
        print(" " * indent * 4 + "TENSOR:shape=" + str(op_or_nd.shape))


class Tanh(Op):
    def __init__(self, args):
        super().__init__()
        self.arity = 1
        self.args = args

    def fw(self):
        self.out = np.vectorize(tanh)(self.args[0].out)

    def bw(self, out):
        self.grad = (np.vectorize(lambda x: 1 - tanh(x) ** 2)(out),)


class Matmul(Op):
    def __init__(self, args):
        super().__init__()
        self.arity = 2
        self.args = args

    def fw(self):
        self.out = np.matmul(self.args[0].out, self.args[1].out)

    def bw(self):
        self.grad = (self.args[1].transpose(), self.args[0].transpose)


ops_fw = {
    Op_Enum.Matmul: np.matmul,
    Op_Enum.Tanh: np.vectorize(tanh),
}

# do/dA ( A B ) = B.transpose
# (allegedly)


def fw_expr(expr):
    if type(expr) is tuple:
        if expr[0] == Op_Enum.Matmul:
            return ops_fw[Op_Enum.Matmul](fw_expr(expr[1]), fw_expr(expr[2]))
        elif expr[0] == Op_Enum.Tanh:
            return ops_fw[Op_Enum.Tanh](fw_expr(expr[1]))
        else:
            raise RuntimeError("invalid op?")
    else:
        print(expr)
        return expr


def expr_str(expr, indent=0):
    _spacing = 3
    _in, _in2 = "|" + _spacing * " ", "|" + _spacing * "-"
    if type(expr) is tuple:
        out = f"{_in * (indent - 1) + (_in2 if indent > 0 else '')}{str(expr[0])}"
        for e in expr[1:]:
            out += f"\n{expr_str(e, indent + 1)}"
    else:
        out = f"{_in * (indent - 1) + (_in2 if indent > 0 else '')}TENSOR:shape={expr.shape}"
    return out


def print_stuff(inputs, layers):
    print("Inputs:")
    print(inputs, "\n")
    for i, lay in enumerate(layers):
        if i == len(layers) - 1:
            print("output layer weights\n" + str(lay), "\n")
        else:
            print(f"hidden-layer-{i} weights\n" + str(lay), "\n")


if __name__ == "__main__":
    mlp = (2, (2, 1))
    # inputs = np.random.uniform(0, 1, [1, mlp[0]])
    inputs = np.array([[-0.25, 0.75]])

    prev_layer = inputs
    layers = []
    for i in mlp[1]:
        left_dim = prev_layer.shape[1]
        layers.append(np.random.uniform(0, 1, [left_dim, i]))
        prev_layer = layers[-1]

    t_tanh = np.vectorize(lambda x: tanh(x))
    t_tanh_r = np.vectorize(lambda x: 1 - tanh(x) ** 2)

    # build expr
    expr = inputs
    for lay in layers:
        expr = (Op_Enum.Tanh, (Op_Enum.Matmul, expr, lay))

    if _DEBUG:
        print("MLP with structure:", mlp, "\n")

        print_stuff(inputs, layers)

        print("expression tree:")
        print(expr_str(expr))

        print("\ninferance:")
        print(fw_expr(expr))
    print(expr_str(expr), "\n")

    # build expr 2
    expr_obj = inputs
    for lay in layers:
        expr_obj = Tanh((Matmul((expr_obj, lay)),))

    Op_disp(expr_obj)
