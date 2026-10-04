"""Train a small network on the scikit-learn handwritten digits dataset
(1,797 8x8 images, 10 classes) and save training curves and a confusion matrix.

scikit-learn is used only to load the dataset; the network itself is pure NumPy.
"""

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.datasets import load_digits

from numpy_nn import (
    Linear, Network, ReLU, SoftmaxCrossEntropy,
    accuracy, confusion_matrix, train, train_val_test_split,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURES = Path(__file__).resolve().parent.parent / "figures"


def main():
    rng = np.random.default_rng(42)
    digits = load_digits()
    x = digits.data / 16.0  # pixel values are 0..16; scale to 0..1
    y = digits.target

    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y, rng=rng)

    net = Network([
        Linear(64, 64, rng), ReLU(),
        Linear(64, 32, rng), ReLU(),
        Linear(32, 10, rng),
    ])
    history = train(net, SoftmaxCrossEntropy(), x_train, y_train, x_val, y_val,
                    epochs=60, lr=0.05, batch_size=32, rng=rng)

    test_acc = accuracy(y_test, net.predict(x_test))
    print(f"train acc {history['train_acc'][-1]:.3f} | "
          f"val acc {history['val_acc'][-1]:.3f} | test acc {test_acc:.3f}")

    FIGURES.mkdir(exist_ok=True)
    plot_curves(history)
    plot_confusion(confusion_matrix(y_test, net.predict(x_test), 10))
    print(f"Figures saved to {FIGURES}")


def plot_curves(history):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    for split in ("train", "val"):
        ax_loss.plot(history[f"{split}_loss"], label=split)
        ax_acc.plot(history[f"{split}_acc"], label=split)
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    ax_loss.legend()
    ax_acc.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "training_curves.png", dpi=120)


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(10), yticks=range(10),
           xlabel="predicted", ylabel="true", title="Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix.png", dpi=120)


if __name__ == "__main__":
    main()
