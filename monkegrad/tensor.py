import math
from typing import List
import numpy as np


"""
def case():
    x, y = uni(0.0, 1.0), uni(0.0, 1.0)
    x_b, y_b = x > 0.5, y > 0.5
    return ((x, y), (1.0 if (x_b and not y_b) or (y_b and not x_b) else 0.0,))


cases = 10
train_data = [case() for _ in range(cases)]
test_data = [case() for _ in range(cases)]

relu = np.vectorize(lambda x: max(0, x))  # relu

w_inner = np.array([[0.638, 0.366], [0.880, 0.878]])
w_output = np.array([[0.282], [0.763]])


for datum in train_data:
    inputs = np.array(datum[0])
    # infer
    output = np.matmul(inputs, w_inner)
    output = relu(output)
    output = np.matmul(output, w_output)
    output = relu(output)
    quantized_output = 1.0 if output[0] > 0.5 else 0.0
    # expected is already quantized
    print(
        f"{output[0]:.2f}",
        f"- {quantized_output:.2f}",
        f" | matches: {abs(quantized_output - datum[1][0]) > 0.01}",
    )
"""


def relu(x):
    return max(0, x)


def tanh(x):
    return (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)


class MLP:
    def __init__(self, input_dim: int, layer_dims: List[int], act_fn):
        self._input_dim = input_dim
        self._layer_dims = layer_dims
        self._act_fn = act_fn

        self.inputs = np.random.uniform(0.0, 1.0, [1, input_dim])
        self.act = np.vectorize(act_fn)
        # aka weights
        self.layers = []
        for i in range(len(layer_dims)):
            prev_dim = input_dim if i == 0 else layer_dims[i - 1]
            self.layers.append(np.random.uniform(0.0, 1.0, [prev_dim, layer_dims[i]]))

    def __repr__(self):
        out = f"Multilayer perceptron:\nactivation fn: {self._act_fn}\n"
        out += f"input (dim: {self._input_dim})\n"
        out += str(self.inputs) + "\n"
        for i, layer in enumerate(self.layers):
            out += f"layer-weights {i} (dim: {self._layer_dims[i]}):\n"
            out += str(layer) + "\n"
        return out

    def fw(self):
        print("forward pass...")
        output = self.inputs
        for layer in self.layers:
            print("mul:")
            print(output)
            print(layer)
            output = np.matmul(output, layer)
            print("result")
            print(output)
            output = self.act(output)
            print("act(result)")
            print(output)

        return output


xor_nn = MLP(2, [3, 1], tanh)
# print(xor_nn)
print("\n" + str(xor_nn) + "\n")
# xor_nn.inputs = np.array([[0.25, 0.75]])
xor_nn.fw()
