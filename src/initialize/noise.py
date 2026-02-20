import numpy as np

def add_SaltPeper_noise(X_img: np.ndarray, amount: float, seed: int) -> np.ndarray:
    if amount <= 0:
        return X_img
    
    rng = np.random.default_rng(seed)
    X = X_img.copy()
    N, H, W = X.shape

    mask = rng.random((N, H, W)) < amount
    salt = rng.random((N, H, W)) < 0.5

    X[mask & salt] = 255
    X[mask & (~salt)] = 0

    return X