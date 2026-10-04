import numpy as np

from .losses import softmax


class Network:
    """A feed-forward network: a sequence of layers applied in order.

    The network outputs logits. Pair it with SoftmaxCrossEntropy for training,
    or use predict_proba / predict for inference.
    """

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def step(self, lr):
        for layer in self.layers:
            layer.step(lr)

    def predict_proba(self, x):
        return softmax(self.forward(x))

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)
