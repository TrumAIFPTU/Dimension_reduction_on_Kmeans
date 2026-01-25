import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

def load_fashion_mnist(n_samples: int, random_state: int = 42):
    """
    Fashion-MNIST (28x28 = 784 dims), nhãn 0..9
    """
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)

    X, y = shuffle(X, y, random_state=random_state)
    X = X[:n_samples]
    y = y[:n_samples]

    X = StandardScaler().fit_transform(X)
    return X.astype(np.float32), y
