import os
from typing import Dict, Any, List, Optional, Literal, Tuple

import numpy as np
from PIL import Image
import torch
from kmeans_gpu  import KMeans
from .utils import _get_pipe


def dominant_color_pillow(
    img: Image.Image,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Zwraca dominujący kolor wg modelu HF oraz listę najlepszych typów.
    Parametry:
        img: PIL.Image.Image – wejściowy obraz
        top_k: ile najlepszych wyników zwrócić (ranking)
    Zwraca:
        {
          "label": "<najbardziej prawdopodobny kolor>",
          "score": <pewność (0..1)>,
          "candidates": List[{"label": str, "score": float}]
        }
    """
    if not isinstance(img, Image.Image):
        raise TypeError("Oczekiwano obiektu PIL.Image.Image")

    # Upewnij się, że obraz jest w formacie RGB (częste wymaganie procesora obrazów)
    if not img.mode == "RGB":
        img = img.convert("RGB")

    pipe = _get_pipe(model_id="AiBototicus/autotrain-colors-1-49130118878")

    # Uruchom klasyfikację (top_k: ile etykiet zwrócić)
    preds: List[Dict[str, Any]] = pipe(img, top_k=top_k)

    # Wynik pipeline może być listą list przy batchu – tutaj mamy pojedynczy obraz
    if isinstance(preds, list) and preds and isinstance(preds[0], dict):
        best = preds[0]
        return {
            "label": best["label"],
            "score": float(best["score"])
        }
    else:
        raise RuntimeError("Nieoczekiwany format wyjścia z pipeline'u")

def findAvgColor(
    img: np.ndarray) -> tuple[float, float, float]:
    return np.average(img.reshape(-1, 3), axis=0)

def findMostPopularColor(
        img: np.ndarray
) -> tuple[float, float, float]:
    unique, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    mostPopularColor = np.argmax(counts).astype(int)
    return unique[mostPopularColor]

os.environ["OMP_NUM_THREADS"] = "8"
clt = KMeans(
        n_clusters=3,
        max_iter=100,
        tolerance=1e-4,
        distance='euclidean',
        sub_sampling=None,
        max_neighbors=15,
    )

def findDominantColor(
    img
) -> tuple[int, int, int]:
    """
    Oblicza dominujący kolor na wycinku zdjęcia\
    :param img:
    :return: bgr color [int, int, int]
    """
    img = img.reshape(-1, 3)

    X = Helper(img)
    closest, centroids = clt.fit_predict(X)
    print(closest, centroids)

    # count_of_labels = {label: np.count_nonzero(clt.labels_ == label) for label in clt.labels_}
    # bestLabel = max(count_of_labels, key=count_of_labels.get)
    #
    # return clt.cluster_centers_[bestLabel]

def dominant_colors(
    img: Image.Image,
    k: int = 5,
    device: Optional[Literal["cuda","cpu"]] = None,
    max_side: int = 768,
    sample_pixels: int = 250_000,
    colorspace: Literal["rgb"] = "rgb",
    return_hex: bool = True,
) -> List[Tuple[Tuple[int,int,int], float, str]]:
    """
    Zwraca listę k dominujących kolorów wraz z udziałem procentowym.
    Każdy element: ((R,G,B), udział_0..1, "#RRGGBB").

    Parametry:
    - image_path: ścieżka do obrazu.
    - k: liczba klastrów/kolorów.
    - device: "cuda" lub "cpu" (domyślnie cuda jeśli dostępna).
    - max_side: maksymalny bok obrazu (dla przyspieszenia przeskalowujemy).
    - sample_pixels: ile pikseli maksymalnie losowo próbkujemy (dla bardzo dużych zdjęć).
    - colorspace: "rgb" (proste i szybkie).
    - return_hex: dołącz hex dla wygody.

    Wymaga: pillow, torch, kmeans-gpu
    """
    # urządzenie
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)

    # piksele -> [N, 3]
    arr = np.asarray(img, dtype=np.uint8).reshape(-1, 3)
    N = arr.shape[0]

    # losowe próbkowanie dla szybkości przy bardzo dużych N
    if N > sample_pixels:
        idx = np.random.choice(N, size=sample_pixels, replace=False)
        arr = arr[idx]
        N = arr.shape[0]

    # tensory na GPU
    pts = torch.from_numpy(arr.astype(np.float32)).to(device) / 255.0  # [N,3] w [0,1]
    pts = pts.unsqueeze(0)  # [1,N,3]

    # kmeans-gpu wymaga też "features" o kształcie [batch, feat_dim, num_pts];
    # dla prostego przypadku użyjemy po prostu transpozycji pikseli jako feature'ów.
    features = pts.permute(0, 2, 1).contiguous()  # [1,3,N]

    # klasteryzacja
    kmeans = KMeans(
        n_clusters=k,
        max_iter=50,
        tolerance=1e-4,
        distance="euclidean",
        sub_sampling=None,
        max_neighbors=15,
    )
    # centroids: [1,k,3]
    centroids, _ = kmeans(pts, features)
    C = centroids[0]  # [k,3]

    # przypisanie pikseli do najbliższych centroidów (na GPU)
    # dystanse: [N,k]
    dists = torch.cdist(pts[0], C)  # [N,k]
    labels = torch.argmin(dists, dim=1)  # [N]

    # częstości i sortowanie wg wielkości klastra
    counts = torch.bincount(labels, minlength=k).float()
    order = torch.argsort(counts, descending=True)
    counts = counts[order]
    C = C[order]

    # wyniki: RGB uint8, udział, hex
    totals = counts.sum().item() if counts.sum().item() > 0 else 1.0
    results = []
    for i in range(k):
        rgb01 = torch.clamp(C[i], 0, 1)
        rgb255 = (rgb01 * 255.0 + 0.5).to(torch.uint8).tolist()
        share = (counts[i].item() / totals)
        if return_hex:
            hexcol = "#{:02X}{:02X}{:02X}".format(*rgb255)
        else:
            hexcol = ""
        results.append(((rgb255[0], rgb255[1], rgb255[2]), share, hexcol))

    return results[0][0]
