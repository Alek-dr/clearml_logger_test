from typing import Tuple

from clearml import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification


def make_data(n_features: int, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_features=n_features,
        n_redundant=0,
        n_informative=n_features,
        random_state=1,
        n_clusters_per_class=1,
        n_classes=n_classes,
        scale=10,
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return X, y


def visualize_data(X, y):
    label_colors = ["b", "r"]
    for label, c in enumerate(label_colors):
        ind = y == label
        x_label = X[ind]
        plt.plot(x_label[:, 0], x_label[:, 1], "o", color=c)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    n_features = 2
    n_classes = 2
    X, y = make_data(n_features, n_classes)
    visualize_data(X, y)

    data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    np.save("./data", data)

    dataset = Dataset.create(dataset_project="test", dataset_name="two-class-classification")
    dataset.add_files("./data.npy")
    dataset.upload(verbose=True)
    dataset.finalize()
