import numpy as np
import pytest

from numpy_nn import (
    LeakyReLU, Linear, Network, ReLU, SoftmaxCrossEntropy,
    confusion_matrix, softmax, train, train_val_test_split,
)


def numerical_grad(f, param, h=1e-6):
    """Central-difference estimate of df/dparam, perturbing `param` in place."""
    grad = np.zeros_like(param)
    for i in np.ndindex(param.shape):
        old = param[i]
        param[i] = old + h
        plus = f()
        param[i] = old - h
        minus = f()
        param[i] = old
        grad[i] = (plus - minus) / (2 * h)
    return grad


@pytest.mark.parametrize("activation", [ReLU, LeakyReLU])
def test_backprop_matches_numerical_gradients(activation):
    rng = np.random.default_rng(0)
    net = Network([Linear(4, 5, rng), activation(), Linear(5, 3, rng)])
    loss_fn = SoftmaxCrossEntropy()
    x = rng.normal(size=(6, 4))
    y = rng.integers(0, 3, size=6)

    def loss():
        return loss_fn.forward(net.forward(x), y)

    loss()
    grad_x = net.backward(loss_fn.backward())

    for layer in (net.layers[0], net.layers[2]):
        np.testing.assert_allclose(layer.weight_grad, numerical_grad(loss, layer.weight), atol=1e-6)
        np.testing.assert_allclose(layer.bias_grad, numerical_grad(loss, layer.bias), atol=1e-6)
    np.testing.assert_allclose(grad_x, numerical_grad(loss, x), atol=1e-6)


def test_softmax_is_stable_for_large_logits():
    probs = softmax(np.array([[1000.0, 1000.0, -1000.0]]))
    assert np.all(np.isfinite(probs))
    np.testing.assert_allclose(probs, [[0.5, 0.5, 0.0]], atol=1e-12)


def test_split_is_a_partition():
    x = np.arange(100).reshape(-1, 1)
    y = np.arange(100)
    x_tr, x_val, x_te, y_tr, y_val, y_te = train_val_test_split(
        x, y, rng=np.random.default_rng(0))
    assert (len(y_tr), len(y_val), len(y_te)) == (80, 10, 10)
    assert sorted(np.concatenate([y_tr, y_val, y_te])) == list(range(100))
    np.testing.assert_array_equal(x_tr[:, 0], y_tr)


def test_training_learns_a_separable_problem():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    net = Network([Linear(2, 8, rng), ReLU(), Linear(8, 2, rng)])
    history = train(net, SoftmaxCrossEntropy(), x, y, x, y,
                    epochs=50, lr=0.1, batch_size=32, rng=rng)
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["val_acc"][-1] > 0.95


def test_confusion_matrix_counts():
    cm = confusion_matrix(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2]), 3)
    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
