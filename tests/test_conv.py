import numpy as np
import pytest

from numpy_nn import Conv2D, Flatten, Linear, MaxPool2D, Network, ReLU, SoftmaxCrossEntropy

from test_numpy_nn import numerical_grad


def naive_conv(x, weight, bias, stride, padding):
    """Direct loop implementation of convolution, used as a reference."""
    n, c, h, w = x.shape
    out_c, _, k, _ = weight.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1
    out = np.zeros((n, out_c, h_out, w_out))
    for b in range(n):
        for o in range(out_c):
            for i in range(h_out):
                for j in range(w_out):
                    patch = xp[b, :, i * stride:i * stride + k, j * stride:j * stride + k]
                    out[b, o, i, j] = np.sum(patch * weight[o]) + bias[o]
    return out


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1)])
def test_conv_matches_naive_implementation(stride, padding):
    rng = np.random.default_rng(0)
    conv = Conv2D(2, 3, 3, stride=stride, padding=padding, rng=rng)
    conv.bias = rng.normal(size=3)
    x = rng.normal(size=(2, 2, 6, 6))
    np.testing.assert_allclose(conv.forward(x), naive_conv(x, conv.weight, conv.bias, stride, padding))


@pytest.mark.parametrize("stride", [1, 2])
def test_cnn_backprop_matches_numerical_gradients(stride):
    rng = np.random.default_rng(0)
    conv = Conv2D(2, 3, 3, stride=stride, padding=1, rng=rng)
    out_size = 4 // stride
    net = Network([conv, ReLU(), MaxPool2D(2), Flatten(),
                   Linear(3 * (out_size // 2) ** 2, 3, rng)])
    loss_fn = SoftmaxCrossEntropy()
    x = rng.normal(size=(2, 2, 4, 4))
    y = np.array([0, 2])

    def loss():
        return loss_fn.forward(net.forward(x), y)

    loss()
    grad_x = net.backward(loss_fn.backward())

    np.testing.assert_allclose(conv.weight_grad, numerical_grad(loss, conv.weight), atol=1e-6)
    np.testing.assert_allclose(conv.bias_grad, numerical_grad(loss, conv.bias), atol=1e-6)
    np.testing.assert_allclose(grad_x, numerical_grad(loss, x), atol=1e-6)


def test_maxpool_routes_gradient_to_max_only():
    pool = MaxPool2D(2)
    x = np.array([[[[1.0, 5.0], [3.0, 2.0]]]])
    assert pool.forward(x).item() == 5.0
    np.testing.assert_array_equal(pool.backward(np.array([[[[7.0]]]])), [[[[0, 7], [0, 0]]]])
