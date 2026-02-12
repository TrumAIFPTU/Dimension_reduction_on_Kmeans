import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist

def load_mnist_images(n_samples: int, random_state: int = 42):
    (x_train, y_train), _ = mnist.load_data()

    rng = np.random.default_rng(random_state)
    n = min(n_samples, x_train.shape[0])
    idx = rng.permutation(x_train.shape[0])[:n]

    X_img = x_train[idx]
    y = y_train[idx].astype(int)
    return X_img, y

def load_fashion_mnist_images(n_samples: int, random_state: int = 42):
    (x_train, y_train), _ = fashion_mnist.load_data()

    rng = np.random.default_rng(random_state)
    n = min(n_samples, x_train.shape[0])
    idx = rng.permutation(x_train.shape[0])[:n]

    X_img = x_train[idx]
    y = y_train[idx].astype(int)
    return X_img, y
