# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image

# --- Wybór implementacji KMeans: GPU (jeśli jest) lub CPU (sklearn) ---
try:
    # jeżeli masz moduł kmeans_gpu, będzie użyty
    from kmeans_gpu import KMeans as _KMeans  # type: ignore
    _KMEANS_IMPL = "gpu"
except Exception:
    # domyślnie CPU – działa na macOS
    from sklearn.cluster import KMeans as _KMeans  # type: ignore
    _KMEANS_IMPL = "cpu"


def _fit_kmeans(X: np.ndarray, n_clusters: int, seed: int = 42):
    """Ujednolicony fit KMeans z bezpiecznym n_init (część wersji akceptuje 'auto')."""
    X = X.astype(np.float32, copy=False)
    try:
        return _KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(X)
    except TypeError:
        # kompatybilność z implementacjami oczekującymi n_init="auto"
        return _KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed).fit(X)


def _rgb_to_hex(c: np.ndarray | List[int] | Tuple[int, int, int]) -> str:
    r, g, b = [int(x) for x in list(c)[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def dominant_colors(
    img: Image.Image | np.ndarray,
    top_k: int = 5,
    sample_limit: int = 50_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Zwraca dominujące kolory z obrazu.

    Parametry:
      img          : PIL.Image.Image lub np.ndarray (H,W,3)
      top_k        : ile najlepszych kolorów zwrócić (1..10)
      sample_limit : maks. liczba pikseli do próbkowania dla szybkości
      seed         : seed do KMeans

    Zwraca:
      {
        "impl": "cpu" | "gpu",
        "top": [{"rgb":[r,g,b], "hex":"#rrggbb", "ratio":0.23}, ...],
        "size": [H, W]
      }
    """
    # --- konwersja do RGB ndarray ---
    if isinstance(img, Image.Image):
        im = img.convert("RGB")
        arr = np.asarray(im)
    else:
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] != 3:
            raise ValueError("dominant_colors: oczekuję obrazu w RGB (H,W,3)")

    H, W = int(arr.shape[0]), int(arr.shape[1])
    X = arr.reshape(-1, 3)

    # --- próbkowanie dla szybkości ---
    n = X.shape[0]
    if n > sample_limit:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, sample_limit, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    k = max(1, min(int(top_k), 10))
    km = _fit_kmeans(X_sample, n_clusters=k, seed=seed)

    centers = km.cluster_centers_.clip(0, 255).astype(np.uint8)

    # przybliżamy proporcje udziałami w próbie
    labels = km.labels_
    counts = np.bincount(labels, minlength=k).astype(np.float32)
    ratios = counts / max(1.0, float(counts.sum()))

    order = np.argsort(ratios)[::-1]
    out: List[Dict[str, Any]] = []
    for i in order[:k]:
        color = centers[i]
        out.append({
            "rgb": color.tolist(),
            "hex": _rgb_to_hex(color),
            "ratio": float(ratios[i]),
        })

    return {"impl": _KMEANS_IMPL, "top": out, "size": [H, W]}
