import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler

def load_swiss_roll(n_samples: int, n_clusters: int, random_state: int = 42, noise: float = 0.05):
    # make_swiss_roll là hàm chuẩn trong sklearn để tạo dữ liệu manifold phi tuyến :contentReference[oaicite:1]{index=1}
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
    X = StandardScaler().fit_transform(X).astype(np.float32)

    # Tạo nhãn "ground-truth" rời rạc bằng cách chia t thành K bins (quantile bins)
    qs = np.quantile(t, q=np.linspace(0, 1, n_clusters + 1))
    y = np.digitize(t, qs[1:-1], right=True).astype(int)  # 0..K-1
    return X, y
