"""Convolutional layers. Inputs are image batches shaped (N, C, H, W)."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .layers import Layer


class Conv2D(Layer):
    """2D convolution implemented with im2col: every k x k patch of the input
    becomes a row of a matrix, so the convolution is one matrix multiplication."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        fan_in = in_channels * kernel_size * kernel_size
        self.weight = rng.normal(0.0, np.sqrt(2.0 / fan_in),
                                 size=(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros(out_channels)
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)
        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        self._cols = None
        self._x_shape = None

    def forward(self, x):
        n, c, _, _ = x.shape
        p, k, s = self.padding, self.k, self.stride
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        # (N, C, H_out, W_out, k, k): every k x k window, taking every s-th one
        windows = sliding_window_view(xp, (k, k), axis=(2, 3))[:, :, ::s, ::s]
        h_out, w_out = windows.shape[2], windows.shape[3]
        # im2col: one row per output position, one column per (channel, ki, kj)
        cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n * h_out * w_out, c * k * k)

        self._cols = cols
        self._x_shape = x.shape
        out = cols @ self.weight.reshape(len(self.bias), -1).T + self.bias
        return out.reshape(n, h_out, w_out, -1).transpose(0, 3, 1, 2)

    def backward(self, grad_out):
        n, c, h, w = self._x_shape
        p, k, s = self.padding, self.k, self.stride
        out_channels, h_out, w_out = grad_out.shape[1:]
        g = grad_out.transpose(0, 2, 3, 1).reshape(-1, out_channels)

        self.weight_grad = (g.T @ self._cols).reshape(self.weight.shape)
        self.bias_grad = g.sum(axis=0)

        # col2im: send each patch's gradient back to the input pixels it came from.
        # Overlapping patches touch the same pixel, so contributions are summed.
        dcols = (g @ self.weight.reshape(out_channels, -1)).reshape(n, h_out, w_out, c, k, k)
        dxp = np.zeros((n, c, h + 2 * p, w + 2 * p))
        for i in range(k):
            for j in range(k):
                dxp[:, :, i:i + s * h_out:s, j:j + s * w_out:s] += dcols[..., i, j].transpose(0, 3, 1, 2)
        return dxp[:, :, p:p + h, p:p + w]

    def step(self, lr):
        self.weight -= lr * self.weight_grad
        self.bias -= lr * self.bias_grad


class MaxPool2D(Layer):
    """Non-overlapping max pooling: keeps the largest value in each size x size block."""

    def __init__(self, size=2):
        self.size = size
        self._argmax = None
        self._x_shape = None

    def forward(self, x):
        n, c, h, w = x.shape
        s = self.size
        assert h % s == 0 and w % s == 0, "input height and width must be divisible by the pool size"
        # (N, C, H/s, W/s, s*s): the values of each pooling block in the last axis
        blocks = x.reshape(n, c, h // s, s, w // s, s).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h // s, w // s, s * s)
        self._argmax = blocks.argmax(axis=-1)
        self._x_shape = x.shape
        return np.take_along_axis(blocks, self._argmax[..., None], axis=-1)[..., 0]

    def backward(self, grad_out):
        # Only the max element of each block affected the output, so only it gets gradient
        n, c, h, w = self._x_shape
        s = self.size
        dblocks = np.zeros((n, c, h // s, w // s, s * s))
        np.put_along_axis(dblocks, self._argmax[..., None], grad_out[..., None], axis=-1)
        return dblocks.reshape(n, c, h // s, w // s, s, s).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)


class Flatten(Layer):
    """Reshapes (N, C, H, W) feature maps into (N, C*H*W) vectors for Linear layers."""

    def __init__(self):
        self._shape = None

    def forward(self, x):
        self._shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, grad_out):
        return grad_out.reshape(self._shape)
