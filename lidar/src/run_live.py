import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from hardware.lidar_driver import init_lidar, get_full_scan
from lidar.system import AiwataLidarSystem
from lidar.occupancy_grid import Pose2D, CellType, CellDanger
from lidar.preprocess import BeamCategory
from config import LIDAR_PORT

# liczba pakietów na pełny skan (jak w live_vis)
FULL_SCAN_PACKETS = 60


# ========= pomocnicze konwersje ==========

def beam_category_to_int(cat: BeamCategory) -> int:
    if cat == BeamCategory.NONE:
        return 0
    if cat == BeamCategory.HUMAN:
        return 1
    if cat == BeamCategory.OBSTACLE:
        return 2
    return -1


def cell_type_to_int(ct: CellType) -> int:
    return int(ct)


def cell_danger_to_int(cd: CellDanger) -> int:
    return int(cd)


# ========= ścieżki i folder sesji ==========

def create_session_dir() -> Path:
    # tworzy folder sesji: data/processed/lidar/session_YYYYMMDD_HHMMSS
    this_file = Path(__file__).resolve()
    src_dir = this_file.parent
    project_root = src_dir.parent

    base_dir = project_root / "data" / "processed" / "lidar"
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / f"session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"Folder sesji: {session_dir}")
    return session_dir


# ========= eksport jednego skanu ==========

def export_scan(
    session_dir: Path,
    scan_idx: int,
    t: float,
    pose: Pose2D,
    result,
) -> None:
    # zapisuje jeden skan do folderu scan_00001/{beams,segments,grid,tracks}.json
    scan_id_str = f"{scan_idx:05d}"
    scan_dir = session_dir / f"scan_{scan_id_str}"
    scan_dir.mkdir(parents=True, exist_ok=True)

    # 1) BEAMS
    beams_rows: List[List[float]] = []
    for b in result.beams:
        beams_rows.append([
            float(b.theta),
            float(b.r),
            float(beam_category_to_int(b.category)),
        ])

    beams_obj: Dict[str, Any] = {
        "scan_id": scan_idx,
        "timestamp": float(t),
        "pose": {
            "x": float(pose.x),
            "y": float(pose.y),
            "yaw": float(pose.yaw),
        },
        "beams": beams_rows,  # N x 3: [theta, r, category_int]
    }

    (scan_dir / "beams.json").write_text(
        json.dumps(beams_obj, ensure_ascii=False),
        encoding="utf-8",
    )

    # 2) SEGMENTS
    segments_rows: List[Dict[str, Any]] = []
    for seg in result.segments:
        segments_rows.append({
            "id": int(seg.id),
            "base_category": beam_category_to_int(seg.base_category),
            "center": [float(seg.center_x), float(seg.center_y)],
            "mean_r": float(seg.mean_r),
            "mean_theta": float(seg.mean_theta),
            "length": float(seg.length),
            "num_beams": len(seg.beams),
        })

    segments_obj: Dict[str, Any] = {
        "scan_id": scan_idx,
        "timestamp": float(t),
        "segments": segments_rows,
    }

    (scan_dir / "segments.json").write_text(
        json.dumps(segments_obj, ensure_ascii=False),
        encoding="utf-8",
    )

    # 3) GRID
    grid = result.grid

    grid_obj: Dict[str, Any] = {
        "scan_id": scan_idx,
        "timestamp": float(t),
        "map_width_m": float(grid.width_m),
        "map_height_m": float(grid.height_m),
        "cell_size_m": float(grid.cell_size),
        "x_min": float(grid.x_min),
        "y_min": float(grid.y_min),
        "cell_type": grid.cell_type.astype(int).tolist(),
        "cell_danger": grid.cell_danger.astype(int).tolist(),
    }

    (scan_dir / "grid.json").write_text(
        json.dumps(grid_obj, ensure_ascii=False),
        encoding="utf-8",
    )

    # 4) TRACKS
    tracks_rows: List[Dict[str, Any]] = []
    for tr in result.human_tracks:
        vx = float(getattr(tr, "vx", 0.0))
        vy = float(getattr(tr, "vy", 0.0))

        raw_speed = (vx ** 2 + vy ** 2) ** 0.5
        speed = float(getattr(tr, "speed", raw_speed))
        heading_rad = float(getattr(tr, "heading_rad", 0.0))

        predictions_list: List[List[float]] = []
        if hasattr(tr, "predict_future_points"):
            try:
                preds = tr.predict_future_points(
                    t_start=t,
                    horizon_s=0.8,
                    step_s=0.2,
                )
                predictions_list = [
                    [float(tp), float(px), float(py)]
                    for (tp, px, py) in preds
                ]
            except Exception:
                predictions_list = []

        tracks_rows.append({
            "id": int(tr.id),
            "type": tr.type,
            "last_position": [
                float(tr.last_position[0]),
                float(tr.last_position[1]),
            ],
            "last_update_time": float(tr.last_update_time),
            "missed_count": int(tr.missed_count),
            "history": [
                [float(ti), float(xi), float(yi)]
                for (ti, xi, yi) in tr.history
            ],
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "heading_rad": heading_rad,
            "predictions": predictions_list,  # [t_pred, x_pred, y_pred]
        })

    tracks_obj: Dict[str, Any] = {
        "scan_id": scan_idx,
        "timestamp": float(t),
        "tracks": tracks_rows,
    }

    (scan_dir / "tracks.json").write_text(
        json.dumps(tracks_obj, ensure_ascii=False),
        encoding="utf-8",
    )


# ========= main loop ==========

def main() -> None:
    # inicjalizacja lidara
    try:
        print(f"Próbuję otworzyć lidar na porcie {LIDAR_PORT}...")
        init_lidar(port=LIDAR_PORT)
        print("Połączono z lidarem.\n")
    except Exception as e:
        print(f"Nie udało się połączyć z lidarem na porcie {LIDAR_PORT}.")
        print("Powód:", e)
        print("Jeśli lidar nie jest podłączony, użyj trybu offline (np. offline_test.py).")
        return

    # folder sesji
    session_dir = create_session_dir()

    # system przetwarzania
    system = AiwataLidarSystem()

    # pozycja robota (na razie stała)
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)

    t0 = time.time()
    scan_idx = 1

    print("Start przetwarzania na żywo (eksport do JSON). Ctrl+C aby zatrzymać.\n")

    try:
        while True:
            # pełniejszy skan z wielu pakietów
            r, theta = get_full_scan(num_packets=FULL_SCAN_PACKETS)
            t = time.time() - t0

            # preprocess + segmentacja + mapa + tracking
            result = system.process_scan(r, theta, pose, t)

            # eksport skanu
            export_scan(session_dir, scan_idx, t, pose, result)
            print(f"Zapisano skan #{scan_idx} (wiązki={len(result.beams)})")

            scan_idx += 1
            # opcjonalny mały sleep, gdyby zapis nadmiernie obciążał dysk
            # time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nZatrzymano live_export (Ctrl+C).")


if __name__ == "__main__":
    main()
