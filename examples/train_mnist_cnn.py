"""Train a convolutional neural network on MNIST (28x28 handwritten digits, 10 classes).

The dataset is downloaded once through scikit-learn (from OpenML) and cached;
the network itself is pure NumPy.
"""

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.datasets import fetch_openml

from numpy_nn import (
    Conv2D, Flatten, Linear, MaxPool2D, Network, ReLU, SoftmaxCrossEntropy,
    accuracy, confusion_matrix, train,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURES = Path(__file__).resolve().parent.parent / "figures"


def load_mnist():
    x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    x = (x / 255.0).reshape(-1, 1, 28, 28)  # (N, channels, height, width), scaled to 0..1
    y = y.astype(int)
    # Standard split: first 60,000 images for training, last 10,000 for testing
    return x[:60000], y[:60000], x[60000:], y[60000:]


def build_cnn(rng):
    return Network([
        Conv2D(1, 8, 3, padding=1, rng=rng), ReLU(), MaxPool2D(2),   # 8 x 14 x 14
        Conv2D(8, 16, 3, padding=1, rng=rng), ReLU(), MaxPool2D(2),  # 16 x 7 x 7
        Flatten(),
        Linear(16 * 7 * 7, 64, rng), ReLU(),
        Linear(64, 10, rng),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-size", type=int, default=55000)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    x_train_all, y_train_all, x_test, y_test = load_mnist()
    # Hold out the last 5,000 training images for validation
    x_train, y_train = x_train_all[:args.train_size], y_train_all[:args.train_size]
    x_val, y_val = x_train_all[55000:], y_train_all[55000:]

    net = build_cnn(rng)
    start = time.time()
    history = train(net, SoftmaxCrossEntropy(), x_train, y_train, x_val, y_val,
                    epochs=args.epochs, lr=0.05, batch_size=64, rng=rng)
    minutes = (time.time() - start) / 60

    test_pred = predict_in_batches(net, x_test)
    print(f"{args.epochs} epochs in {minutes:.1f} min | "
          f"val acc {history['val_acc'][-1]:.4f} | test acc {accuracy(y_test, test_pred):.4f}")

    FIGURES.mkdir(exist_ok=True)
    plot_confusion(confusion_matrix(y_test, test_pred, 10))
    plot_predictions(x_test, y_test, test_pred)
    print(f"Figures saved to {FIGURES}")


def predict_in_batches(net, x, batch_size=1000):
    return np.concatenate([net.predict(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(10), yticks=range(10),
           xlabel="predicted", ylabel="true", title="MNIST confusion matrix (10,000 test images)")
    fig.tight_layout()
    fig.savefig(FIGURES / "mnist_confusion_matrix.png", dpi=120)


def plot_predictions(x, y, pred, n=12):
    """Show some test images with predictions; mistakes are titled in red."""
    wrong = np.flatnonzero(pred != y)[:n // 2]
    right = np.flatnonzero(pred == y)[:n - len(wrong)]
    fig, axes = plt.subplots(2, n // 2, figsize=(n, 4.5))
    for ax, i in zip(axes.flat, np.concatenate([right, wrong])):
        ax.imshow(x[i, 0], cmap="gray_r")
        ax.set_title(f"pred {pred[i]} / true {y[i]}", color="black" if pred[i] == y[i] else "red", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "mnist_predictions.png", dpi=120)


if __name__ == "__main__":
    main()
