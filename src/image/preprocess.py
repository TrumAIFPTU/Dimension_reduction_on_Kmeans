import numpy as np
from scipy.signal import convolve2d


def normalize_uint8_to_float01(X_img: np.ndarray) -> np.ndarray:
    return (X_img.astype(np.float32) / 255.0) #Normalize value from (0 -> 255) to (0 -> 1)


def sharpen_kernel_3x3() -> np.ndarray:
    return np.array(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]],
        dtype=np.float32,
    )


def apply_sharpen_batch(X01: np.ndarray) -> np.ndarray:
    k = sharpen_kernel_3x3()
    out = np.empty_like(X01, dtype=np.float32)
    for i in range(X01.shape[0]):
        out[i] = convolve2d(X01[i], k, mode="same", boundary="symm")
    return np.clip(out, 0.0, 1.0)


def preprocess_and_sharpen(X_img: np.ndarray) -> np.ndarray:
    return apply_sharpen_batch(normalize_uint8_to_float01(X_img))#Nỏmalize + sharpen
