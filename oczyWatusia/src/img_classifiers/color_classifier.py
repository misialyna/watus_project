import os
import numpy as np
import torch
from kmeans_gpu  import KMeans

def findDominantColor(img, n_clusters: int = 16) -> tuple[float, float, float]:
    os.environ["OMP_NUM_THREADS"] = "8"
    img = np.asarray(img)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img = np.asarray(img)

    if img.ndim == 3:
        h, w, c = img.shape
        if c < 3:
            raise ValueError(f"Oczekiwano ≥3 kanałów (RGB), a jest {c}.")
        pixels = img[..., :3].reshape(-1, 3)
    elif img.ndim == 2:
        if img.shape[1] < 3:
            raise ValueError(f"Oczekiwano ≥3 kanałów (RGB), a jest {img.shape[1]}.")
        pixels = img[:, :3]
    else:
        raise ValueError("Oczekiwano obrazu (H, W, 3) lub listy pikseli (N, 3).")

    if pixels.size == 0:
        raise ValueError("Obraz nie zawiera żadnych pikseli.")

    n_clusters_eff = int(min(n_clusters, pixels.shape[0]))

    pixels_f = torch.from_numpy(pixels.astype(np.float32)).to(device)

    points = pixels_f.unsqueeze(0)  # (1, N, 3)
    features = points.permute(0, 2, 1)  # (1, 3, N)

    kmeans = KMeans(
        n_clusters=n_clusters_eff,
        max_iter=100,
        tolerance=1e-2,
        distance="euclidean",
        sub_sampling=None,
        max_neighbors=5,
    ).to(device)

    centroids_t, _ = kmeans(points, features)

    centroids = centroids_t[0].detach().cpu().numpy()  # (K, 3)

    # diff: (N, K, 3)
    diff = pixels[:, None, :] - centroids[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)  # (N, K)
    labels = np.argmin(dist2, axis=1)  # (N,)

    # Liczności klastrów
    counts = np.bincount(labels, minlength=n_clusters_eff)
    dominant_label = int(np.argmax(counts))

    dominant_color = centroids[dominant_label]
    dominant_color = np.clip(dominant_color, 0, 255).astype(np.uint8)  # (3,)

    return dominant_color