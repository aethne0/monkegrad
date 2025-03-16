import math
from typing import List
import numpy as np


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
        #
        # aka weights
        # layer[:-1] IS the output layer
        self.layers = []
        for i in range(len(layer_dims)):
            prev_dim = input_dim if i == 0 else layer_dims[i - 1]
            self.layers.append(np.random.uniform(0.0, 1.0, [prev_dim, layer_dims[i]]))

        self.layer_grads = []
        for i in range(len(layer_dims)):
            prev_dim = input_dim if i == 0 else layer_dims[i - 1]
            self.layer_grads.append(np.zeros([prev_dim, layer_dims[i]]))

    def __repr__(self):
        out = f"Multilayer perceptron:\nactivation fn: {self._act_fn}\n"
        out += f"input (dim: {self._input_dim})\n"
        out += str(self.inputs) + "\n"
        for i, layer in enumerate(self.layers):
            out += f"layer-{i} weights (dim: {self._layer_dims[i]}):\n"
            out += str(layer) + "\n"
        return out

    def fw(self):
        output = self.inputs
        for layer in self.layers:
            output = self.act(np.matmul(output, layer))
        return output

    def bw(self):
        for i, layer in enumerate(self.layer_grads[:-1]):
            self.layer_grads[i] = np.zeros(layer.shape)
        # output layer ones
        self.layer_grads[-1] = np.ones(self.layer_grads[-1].shape)

        for i, layer in enumerate(self.layer_grads):
            print(i, self.layer_grads[i])


if __name__ == "__main__":
    xor_nn = MLP(2, [2, 2], relu)
    print("\n" + str(xor_nn) + "\n")

    print(xor_nn.fw())
    print("\n")
    xor_nn.bw()
