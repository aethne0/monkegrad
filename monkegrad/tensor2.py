from enum import Enum, auto
from math import tanh

import numpy as np


class Op(Enum):
    Matmul = auto()
    Tanh = auto()


ops = {Op.Matmul: np.matmul, Op.Tanh: np.vectorize(tanh)}

_INDENT_MOD = 2


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
        if expr[0] == Op.Matmul:
            return ops[Op.Matmul](fw_expr(expr[1]), fw_expr(expr[2]))
        elif expr[0] == Op.Tanh:
            return ops[Op.Tanh](fw_expr(expr[1]))
        else:
            raise RuntimeError("invalid op?")
    else:
        return expr


if __name__ == "__main__":
    mlp = (2, (1,))
    inputs = np.random.uniform(0, 1, [1, mlp[0]])
    prev_layer = inputs
    layers = []
    for i in mlp[1]:
        left_dim = prev_layer.shape[1]
        layers.append(np.random.uniform(0, 1, [left_dim, i]))
        prev_layer = layers[-1]

    print(inputs, "\n")
    for lay in layers:
        print(lay, "\n")

    t_tanh = np.vectorize(lambda x: tanh(x))
    t_tanh_r = np.vectorize(lambda x: 1 - tanh(x) ** 2)

    # build expr
    expr = inputs
    for lay in layers:
        expr = (Op.Tanh, (Op.Matmul, expr, lay))

    print(expr_str(expr))
    print(fw_expr(expr))
