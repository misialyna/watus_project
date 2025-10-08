from __future__ import annotations
import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import ultralytics.data.build as build
from dotenv import load_dotenv
load_dotenv()

from yolo_dataset import YOLOWeightedDataset
build.YOLODataset = YOLOWeightedDataset

# ——— Pomocnicze struktury ———

@dataclass
class YoloBox:
    cls: int
    x: float
    y: float
    w: float
    h: float
    img_w: int
    img_h: int
    image_path: Path

    @property
    def area_rel(self) -> float:
        # pole w ułamku powierzchni obrazu
        return max(self.w, 0.0) * max(self.h, 0.0)

    @property
    def aspect(self) -> float:
        # proporcja w/h
        return self.w / self.h if self.h > 0 else np.nan

    @property
    def width_px(self) -> float:
        return self.w * self.img_w

    @property
    def height_px(self) -> float:
        return self.h * self.img_h

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x, self.y)


# ——— Parsowanie YOLO ———

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def to_paths(x) -> List[Path]:
    # Entry w .yaml może być stringiem do folderu, listą plików .txt z listą ścieżek,
    # albo listą katalogów/plików; normalizujemy to do listy ścieżek obrazów.
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [Path(s) for s in x]
    return [Path(x)]

def expand_split_to_images(entry: Path) -> List[Path]:
    # Jeżeli podano plik .txt z listą obrazów — wczytaj.
    # Jeżeli katalog — zbierz typowe rozszerzenia.
    if entry.is_file() and entry.suffix.lower() == ".txt":
        with open(entry, "r", encoding="utf-8") as f:
            return [Path(line.strip()) for line in f if line.strip()]
    # Katalog z obrazami
    if entry.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return sorted([p for p in entry.rglob("*") if p.suffix.lower() in exts])
    # Gdy to bezpośrednia ścieżka do obrazu
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if entry.suffix.lower() in exts and entry.exists():
        return [entry]
    return []

def image_to_label_path(img_path: Path, data_root: Optional[Path]=None) -> Path:
    """
    Próbuje przemapować .../images/.../name.jpg -> .../labels/.../name.txt (standard YOLO),
    ale jeżeli dane mają nietypową strukturę, spróbuje labela w tym samym folderze.
    """
    # Standard: zamień 'images' -> 'labels' i rozszerzenie -> .txt
    parts = list(img_path.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        lbl = Path(*parts).with_suffix(".txt")
        if lbl.exists():
            return lbl
    except ValueError:
        pass

    # Alternatywa: /labels równoległy do /images
    if "images" in img_path.parts:
        i = parts.index("images")
        candidate = Path(*parts[:i], "labels", *parts[i+1:]).with_suffix(".txt")
        if candidate.exists():
            return candidate

    # Ostatecznie: ten sam folder, ta sama nazwa .txt
    same = img_path.with_suffix(".txt")
    if same.exists():
        return same

    # Szukaj po nazwie w całym drzewie (droższe, tylko jako fallback)
    if data_root and data_root.exists():
        name = img_path.stem + ".txt"
        for p in data_root.rglob(name):
            if p.suffix.lower() == ".txt":
                return p

    return img_path.with_suffix(".missing.txt")  # nieistniejący znacznik

def read_image_size(p: Path) -> Tuple[int, int]:
    with Image.open(p) as im:
        return im.size  # (w, h)

def parse_label_file(lbl_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Zwraca listę rekordów: (cls, x, y, w, h) w notacji YOLO (wartości znormalizowane).
    """
    items = []
    if not lbl_path.exists():
        return items
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                items.append((cls, x, y, w, h))
            except Exception:
                continue
    return items

def collect_images_from_yaml(data_yaml: Path, splits: List[str]) -> Tuple[List[Path], Dict[int, str], Optional[Path]]:
    data = load_yaml(data_yaml)
    names = {}
    if "names" in data:
        if isinstance(data["names"], dict):
            names = {int(k): str(v) for k, v in data["names"].items()}
        elif isinstance(data["names"], list):
            names = {i: str(n) for i, n in enumerate(data["names"])}
    # YOLO czasem podaje 'path' jako root dla względnych ścieżek

    root = Path(data.get("path", data_yaml)).resolve()

    split_keys = {"train": data.get("train"), "val": data.get("val"), "test": data.get("test")}
    selected = [k for k in splits if split_keys.get(k) is not None]

    all_images: List[Path] = []
    for k in selected:
        entries = to_paths(split_keys[k])
        for e in entries:
            e = (root / e).resolve() if not e.is_absolute() else e
            imgs = expand_split_to_images(e)
            all_images.extend(imgs)

    # deduplikacja
    all_images = sorted(list({p.resolve() for p in all_images if p.exists()}))
    return all_images, names, root

# ——— Analiza ———

@dataclass
class DatasetStats:
    boxes: List[YoloBox]
    classes_count: Counter
    imgs_obj_count: List[int]
    img_sizes: List[Tuple[int, int]]

def analyze_dataset(images: List[Path], names: Dict[int, str], data_root: Optional[Path], max_images: int = 0) -> DatasetStats:
    boxes: List[YoloBox] = []
    classes_count: Counter = Counter()
    imgs_obj_count: List[int] = []
    img_sizes: List[Tuple[int, int]] = []

    if max_images and max_images > 0:
        images = images[:max_images]

    for img in tqdm(images, desc="Przetwarzanie obrazów"):
        try:
            w, h = read_image_size(img)
            img_sizes.append((w, h))
        except Exception:
            # pomiń uszkodzone obrazy
            continue

        lbl = image_to_label_path(img, data_root)
        items = parse_label_file(lbl)
        imgs_obj_count.append(len(items))
        for cls, x, y, bw, bh in items:
            classes_count[cls] += 1
            boxes.append(YoloBox(cls=cls, x=x, y=y, w=bw, h=bh, img_w=w, img_h=h, image_path=img))

    return DatasetStats(boxes=boxes, classes_count=classes_count, imgs_obj_count=imgs_obj_count, img_sizes=img_sizes)

# ——— Wykresy ———

def ensure_out(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

def save_fig(path: Path, tight=True, dpi=150):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def plot_class_distribution(stats: DatasetStats, names: Dict[int, str], out: Path):
    if not stats.classes_count:
        return
    labels = sorted(stats.classes_count.keys())
    counts = [stats.classes_count[i] for i in labels]
    tick_labels = [names.get(i, str(i)) for i in labels]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), tick_labels, rotation=45, ha="right")
    plt.ylabel("Liczba obiektów")
    plt.title("Rozkład klas (obiekty na zbiorze)")
    save_fig(out / "classes_bar.png")

def plot_objects_per_image(stats: DatasetStats, out: Path):
    data = np.array(stats.imgs_obj_count, dtype=float)
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=min(50, max(5, int(np.sqrt(len(data))))))
    plt.xlabel("Obiektów na obraz")
    plt.ylabel("Liczba obrazów")
    plt.title("Histogram: obiekty na obraz")
    save_fig(out / "objects_per_image.png")

def plot_bbox_area(stats: DatasetStats, out: Path):
    if not stats.boxes:
        return
    areas = np.array([b.area_rel for b in stats.boxes], dtype=float)
    plt.figure(figsize=(8, 4))
    plt.hist(areas, bins=60)
    plt.xlabel("Pole bbox (ułamek powierzchni obrazu)")
    plt.ylabel("Liczba obiektów")
    plt.title("Histogram: pole bbox (znormalizowane)")
    save_fig(out / "bbox_area_rel.png")

def plot_bbox_aspect(stats: DatasetStats, out: Path):
    if not stats.boxes:
        return
    aspects = np.array([b.aspect for b in stats.boxes if not math.isnan(b.aspect)], dtype=float)
    plt.figure(figsize=(8, 4))
    plt.hist(aspects, bins=60)
    plt.xlabel("Proporcja bbox (w/h)")
    plt.ylabel("Liczba obiektów")
    plt.title("Histogram: proporcje bbox (w/h)")
    save_fig(out / "bbox_aspect.png")

def plot_bbox_w_vs_h(stats: DatasetStats, out: Path):
    if not stats.boxes:
        return
    ws = np.array([b.w for b in stats.boxes], dtype=float)
    hs = np.array([b.h for b in stats.boxes], dtype=float)
    plt.figure(figsize=(5, 5))
    plt.scatter(ws, hs, s=4, alpha=0.5)
    plt.xlabel("Szerokość bbox (rel.)")
    plt.ylabel("Wysokość bbox (rel.)")
    plt.title("Rozrzut: szerokość vs wysokość bbox (rel.)")
    save_fig(out / "bbox_w_vs_h.png")

def plot_centers_heatmap(stats: DatasetStats, out: Path, bins: int = 50):
    if not stats.boxes:
        return
    centers = np.array([b.center for b in stats.boxes], dtype=float)
    H, xedges, yedges = np.histogram2d(centers[:,0], centers[:,1], bins=bins, range=[[0,1],[0,1]])
    plt.figure(figsize=(5.5, 5))
    plt.imshow(H.T, origin="lower", extent=[0,1,0,1], aspect="auto")
    plt.colorbar(label="Gęstość")
    plt.xlabel("x (centrum)")
    plt.ylabel("y (centrum)")
    plt.title("Mapa gęstości centrów bbox")
    save_fig(out / "centers_heatmap.png")

def plot_image_resolutions(stats: DatasetStats, out: Path):
    if not stats.img_sizes:
        return
    sizes = np.array(stats.img_sizes, dtype=float)
    ws, hs = sizes[:,0], sizes[:,1]
    plt.figure(figsize=(6, 5))
    plt.scatter(ws, hs, s=6, alpha=0.6)
    plt.xlabel("Szerokość obrazu [px]")
    plt.ylabel("Wysokość obrazu [px]")
    plt.title("Rozdzielczości obrazów")
    save_fig(out / "image_resolutions.png")

# ——— Eksport statystyk ———

def export_summary(stats: DatasetStats, names: Dict[int, str], out: Path):
    summary = {
        "num_images": len(stats.img_sizes),
        "num_objects": int(sum(stats.classes_count.values())),
        "classes": [
            {"id": int(cid), "name": names.get(cid, str(cid)), "count": int(cnt)}
            for cid, cnt in sorted(stats.classes_count.items())
        ],
        "objects_per_image": {
            "min": int(min(stats.imgs_obj_count)) if stats.imgs_obj_count else 0,
            "max": int(max(stats.imgs_obj_count)) if stats.imgs_obj_count else 0,
            "mean": float(np.mean(stats.imgs_obj_count)) if stats.imgs_obj_count else 0.0,
            "median": float(np.median(stats.imgs_obj_count)) if stats.imgs_obj_count else 0.0,
        }
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

def export_objects_csv(stats: DatasetStats, names: Dict[int, str], out: Path, max_rows: int = 0):
    rows = []
    for b in stats.boxes:
        rows.append({
            "image": str(b.image_path),
            "class_id": b.cls,
            "class_name": names.get(b.cls, str(b.cls)),
            "cx_rel": b.x,
            "cy_rel": b.y,
            "w_rel": b.w,
            "h_rel": b.h,
            "img_w": b.img_w,
            "img_h": b.img_h,
            "area_rel": b.area_rel,
            "aspect": b.aspect
        })
    if max_rows and max_rows > 0:
        rows = rows[:max_rows]
    with open(out / "objects.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "image","class_id","class_name","cx_rel","cy_rel","w_rel","h_rel","img_w","img_h","area_rel","aspect"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ——— Główna ścieżka ———

def main():
    parser = argparse.ArgumentParser(description="Analiza zbioru YOLO (YOLOv12/YOLOv5/YOLOv8 zgodny format).")
    parser.add_argument("--data", type=str, required=True, help="Ścieżka do pliku dataset.yaml")
    parser.add_argument("--out", type=str, default="analysis_out", help="Katalog na wyniki (obrazy/JSON/CSV)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Które splity analizować (np. train val test)")
    parser.add_argument("--max-images", type=int, default=0, help="Limit liczby obrazów (0 = wszystkie)")
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()
    ensure_out(out_dir)

    print(f"[INFO] Wczytywanie {data_yaml}...")
    images, names, data_root = collect_images_from_yaml(data_yaml, args.splits)
    if args.max_images and args.max_images > 0:
        images = images[:args.max_images]
    print(f"[INFO] Obrazów do analizy: {len(images)}")
    if not images:
        print("[WARN] Nie znaleziono obrazów. Sprawdź ścieżki w YAML.")
        return

    stats = analyze_dataset(images, names, data_root, max_images=0)

    # Wykresy
    print("[INFO] Rysowanie wykresów...")
    plot_class_distribution(stats, names, out_dir)
    plot_objects_per_image(stats, out_dir)
    plot_bbox_area(stats, out_dir)
    plot_bbox_aspect(stats, out_dir)
    plot_bbox_w_vs_h(stats, out_dir)
    plot_centers_heatmap(stats, out_dir)
    plot_image_resolutions(stats, out_dir)

    # Eksport
    print("[INFO] Zapisywanie podsumowań...")
    export_summary(stats, names, out_dir)
    export_objects_csv(stats, names, out_dir, max_rows=0)

    print(f"[GOTOWE] Wyniki zapisane w: {out_dir}")
    print("Pliki: classes_bar.png, objects_per_image.png, bbox_area_rel.png, bbox_aspect.png, "
          "bbox_w_vs_h.png, centers_heatmap.png, image_resolutions.png, summary.json, objects.csv")

if __name__ == "__main__":
    main()
