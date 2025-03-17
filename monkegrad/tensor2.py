from enum import Enum, auto
from math import tanh

import numpy as np

np.random.seed(sum([ord(ch) for ch in "monke"]))
_DEBUG = True


class Op_Enum(Enum):
    Matmul = auto()
    Tanh = auto()


class Op:
    def __init__(self):
        self.arity = 0


class Tanh(Op):
    def __init__(self):
        self.arity = 1

    def fw(self, x):
        return np.vectorize(tanh)(x)

    def bw(self, out):
        return (np.vectorize(lambda x: 1 - tanh(x) ** 2)(out),)


class Matmul(Op):
    def __init__(self, args):
        self.arity = 2
        self.args = args

    def fw(self):
        return np.matmul(self.args[0], self.args[1])

    def bw(self):
        return (self.args[1].transpose(), self.args[0].transpose)


ops_fw = {
    Op_Enum.Matmul: np.matmul,
    Op_Enum.Tanh: np.vectorize(tanh),
}

# do/dA ( A B ) = B.transpose

_INDENT_MOD = 3


def expr_str(expr, indent=0):
    if type(expr) is tuple:
        out = f"{' ' * indent}{str(expr[0])}"
        for e in expr[1:]:
            out += f"\n{expr_str(e, indent + _INDENT_MOD)}"
    else:
        out = f"{' ' * indent}TENSOR:{expr.shape}"

    return out


def fw_expr(expr):
    if type(expr) is tuple:
        if expr[0] == Op_Enum.Matmul:
            return ops_fw[Op_Enum.Matmul](fw_expr(expr[1]), fw_expr(expr[2]))
        elif expr[0] == Op_Enum.Tanh:
            return ops_fw[Op_Enum.Tanh](fw_expr(expr[1]))
        else:
            raise RuntimeError("invalid op?")
    else:
        expr = expr
        return expr


def print_stuff(inputs, layers):
    print("Inputs:")
    print(inputs, "\n")
    for i, lay in enumerate(layers):
        if i == len(layers) - 1:
            print("output layer weights\n" + str(lay), "\n")
        else:
            print(f"hidden-layer-{i} weights\n" + str(lay), "\n")


if __name__ == "__main__":
    mlp = (2, (1,))
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
