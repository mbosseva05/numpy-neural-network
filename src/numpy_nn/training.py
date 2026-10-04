import numpy as np


def train_val_test_split(x, y, val_ratio=0.1, test_ratio=0.1, rng=None):
    """Shuffle and split into train / validation / test sets."""
    rng = rng if rng is not None else np.random.default_rng()
    n = len(x)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test, val, train = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    return x[train], x[val], x[test], y[train], y[val], y[test]


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def train(net, loss_fn, x_train, y_train, x_val, y_val,
          epochs=100, lr=0.1, batch_size=None, rng=None):
    """Train with (mini-batch) gradient descent.

    batch_size=None uses the full training set per step (full-batch gradient descent).
    Returns a history dict with per-epoch loss and accuracy for train and validation.
    """
    rng = rng if rng is not None else np.random.default_rng()
    n = len(x_train)
    batch_size = batch_size or n
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for _ in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            logits = net.forward(x_train[batch])
            loss_fn.forward(logits, y_train[batch])
            net.backward(loss_fn.backward())
            net.step(lr)

        for split, x, y in (("train", x_train, y_train), ("val", x_val, y_val)):
            loss, acc = evaluate(net, loss_fn, x, y)
            history[f"{split}_loss"].append(loss)
            history[f"{split}_acc"].append(acc)

    return history


def evaluate(net, loss_fn, x, y, batch_size=1000):
    """Mean loss and accuracy over a dataset, computed in batches to limit memory use."""
    total_loss, correct = 0.0, 0
    for start in range(0, len(x), batch_size):
        xb, yb = x[start:start + batch_size], y[start:start + batch_size]
        logits = net.forward(xb)
        total_loss += loss_fn.forward(logits, yb) * len(yb)
        correct += np.sum(np.argmax(logits, axis=1) == yb)
    return total_loss / len(x), correct / len(x)


def confusion_matrix(y_true, y_pred, n_classes):
    """Row i, column j counts samples of true class i predicted as class j."""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix
