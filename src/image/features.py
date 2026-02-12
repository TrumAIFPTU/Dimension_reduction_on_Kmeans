import numpy as np
from skimage.feature import hog


def hog_features(
    img01: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: int = 4,
    cells_per_block: int = 2,
) -> np.ndarray:
    #HOG feature cho 1 ảnh (float32, [0,1])
    return hog(
        img01,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        block_norm="L2-Hys",
        feature_vector=True,).astype(np.float32)

def extract_feature_matrix(
    X01: np.ndarray,
    hog_orientations: int = 9,
    hog_pixels_per_cell: int = 4,
    hog_cells_per_block: int = 2,) -> np.ndarray:
    feats = []
    for i in range(X01.shape[0]):
        img = X01[i]
        f = hog_features(
                    img,
                    orientations=hog_orientations,
                    pixels_per_cell=hog_pixels_per_cell,
                    cells_per_block=hog_cells_per_block,)
        feats.append(f)
    return np.vstack(feats).astype(np.float32)
