from enum import Enum, auto
from math import tanh

import numpy as np

np.random.seed(sum([ord(ch) for ch in "monke"]))


class Ops(Enum):
    Matmul = auto()
    Tanh = auto()


class Op:
    def __init__(self, args=()):
        self.args = args
        self.arity = 0
        self.out = 0.0
        self.grad = 1.0

    def __repr__(self):
        return (
            f"{type(self)}".ljust(30)
            + f"| out={self.out}, grad={self.grad} args={len(self.args)} |"
        )


class Literal(Op):
    def __init__(self, args, name=""):
        super().__init__()
        self.name = name
        self.args = args
        self.out = args[0]

    def fw(self):
        pass

    def bw(self):
        pass

    def __repr__(self):
        return f"LITERAL: {self.name}\n" + str(self.out)


def print_op(op_or_nd, indent=0):
    if type(op_or_nd) is not np.ndarray:
        print(" " * indent * 4 + str(op_or_nd))
        for i in op_or_nd.args:
            print_op(i, indent + 1)
    else:
        print(" " * indent * 4 + "TENSOR:shape=" + str(op_or_nd.shape))


class Tanh(Op):
    def __init__(self, args):
        super().__init__()
        self.arity = 1
        self.args = args

    def fw(self):
        self.out = np.vectorize(tanh)(self.args[0].out)

    def bw(self):
        self.grad = (np.vectorize(lambda x: 1 - tanh(x) ** 2)(self.out),)


class Matmul(Op):
    def __init__(self, args):
        super().__init__()
        self.arity = 2
        self.args = args

    def fw(self):
        self.out = np.matmul(self.args[0].out, self.args[1].out)

    def bw(self):
        self.grad = (self.args[1].out.transpose(), self.args[0].out.transpose())


# do/dA ( A B ) = B.transpose
# (allegedly)

if __name__ == "__main__":
    mlp = (2, (1,))
    inputs = np.array([[-0.5, 0.5]])

    prev_layer = inputs
    layers = []
    for i in mlp[1]:
        left_dim = prev_layer.shape[1]
        layers.append(np.random.uniform(0, 1, [left_dim, i]))
        prev_layer = layers[-1]

    # build expr 2
    expr_obj = Literal((inputs,), "inputs")
    for i, lay in enumerate(layers):
        expr_obj = Tanh(
            (
                Matmul(
                    (
                        expr_obj,
                        Literal((lay,), f"weights-{i}"),
                    )
                ),
            )
        )

    seen = set()
    topo = []

    def add_to_topo(e):
        # global seen, topo
        if type(e) is np.ndarray or e in seen:
            return
        else:
            seen.add(e)
            topo.append(e)
            for i in e.args:
                add_to_topo(i)

    add_to_topo(expr_obj)

    print("topologically sorted -> forward pass ->")
    print("# literals will appear in non-meaningful order")
    print("---------------------------------------")
    for n in reversed(topo):
        n.fw()
        print(f"{str(n)}")

    print()
    # print()
    # print()

    print("backward pass")
    print("---------------------------------------")
    for n in topo:
        n.bw()
        print(f"{str(n)}")
