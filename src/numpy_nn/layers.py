"""Layers of the network. Each layer caches what it needs in `forward`
and uses it in `backward` to compute gradients (backpropagation)."""

import numpy as np


class Layer:
    """Base class. Layers without parameters only override forward/backward."""

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError

    def step(self, lr):
        """Update parameters with gradient descent. No-op for parameter-free layers."""


class Linear(Layer):
    """Fully connected layer: y = x @ W + b."""

    def __init__(self, in_features, out_features, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        # He initialisation: keeps activation variance stable through ReLU layers
        self.weight = rng.normal(0.0, np.sqrt(2.0 / in_features), size=(in_features, out_features))
        self.bias = np.zeros(out_features)
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.weight + self.bias

    def backward(self, grad_out):
        """Given dL/dy, store dL/dW and dL/db and return dL/dx."""
        self.weight_grad = self._x.T @ grad_out
        self.bias_grad = grad_out.sum(axis=0)
        return grad_out @ self.weight.T

    def step(self, lr):
        self.weight -= lr * self.weight_grad
        self.bias -= lr * self.bias_grad


class ReLU(Layer):
    """f(x) = max(0, x)."""

    def __init__(self):
        self._x = None

    def forward(self, x):
        self._x = x
        return np.maximum(0, x)

    def backward(self, grad_out):
        return grad_out * (self._x > 0)


class LeakyReLU(Layer):
    """f(x) = x if x > 0 else alpha * x. Avoids 'dead' neurons with zero gradient."""

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self._x = None

    def forward(self, x):
        self._x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad_out):
        return grad_out * np.where(self._x > 0, 1.0, self.alpha)
