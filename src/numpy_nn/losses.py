import numpy as np


def softmax(logits):
    """Row-wise softmax. Subtracting the row max avoids overflow in exp()
    without changing the result."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """Softmax followed by cross-entropy loss, computed together.

    Combining them gives a simple, numerically stable gradient:
    dL/dlogits = (softmax(logits) - one_hot(labels)) / N
    """

    def __init__(self, eps=1e-15):
        self.eps = eps
        self._probs = None
        self._labels = None

    def forward(self, logits, labels):
        """Mean cross-entropy loss. `labels` are integer class indices 0..C-1."""
        probs = softmax(logits)
        self._probs = probs
        self._labels = labels
        correct = probs[np.arange(len(labels)), labels]
        return -np.mean(np.log(np.clip(correct, self.eps, 1.0)))

    def backward(self):
        grad = self._probs.copy()
        grad[np.arange(len(self._labels)), self._labels] -= 1
        return grad / len(self._labels)
