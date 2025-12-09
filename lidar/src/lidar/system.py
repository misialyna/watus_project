from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from lidar.preprocess import preprocess_scan, BeamResult, BeamCategory
from lidar.segmentation import segment_scan, assign_segment_categories, Segment
from lidar.occupancy_grid import OccupancyGrid, Pose2D
from lidar.tracking import HumanTracker, HumanTrack

from config import (
    MAP_WIDTH_M,
    MAP_HEIGHT_M,
    CELL_SIZE_M,
    TRACK_MAX_MATCH_DISTANCE_M,
    TRACK_MAX_MISSED,
)


@dataclass
class ScanResult:
    # Wynik przetworzenia jednego skanu
    beams: List[BeamResult]         # wiązki z kategoriami
    segments: List[Segment]         # segmenty obiektów
    human_tracks: List[HumanTrack]  # tracki ludzi
    grid: OccupancyGrid             # zaktualizowana mapa


class AiwataLidarSystem:
    # Główna klasa spinająca preprocess, segmentację, mapę i tracking

    def __init__(
        self,
        map_width_m: float = MAP_WIDTH_M,
        map_height_m: float = MAP_HEIGHT_M,
        cell_size_m: float = CELL_SIZE_M,
        max_match_distance: float = TRACK_MAX_MATCH_DISTANCE_M,
        max_missed: int = TRACK_MAX_MISSED,
    ):
        # inicjalizacja mapy
        self.grid = OccupancyGrid(
            width_m=map_width_m,
            height_m=map_height_m,
            cell_size_m=cell_size_m,
        )

        # inicjalizacja trackera ludzi
        self.tracker = HumanTracker(
            max_match_distance=max_match_distance,
            max_missed=max_missed,
        )

    def process_scan(
        self,
        r: np.ndarray,
        theta: np.ndarray,
        pose: Pose2D,
        t: float,
    ) -> ScanResult:
        # pełny pipeline przetwarzania pojedynczego skanu

        # 1) preprocessing wiązek
        beams: List[BeamResult] = preprocess_scan(r, theta)

        # 2) segmentacja i klasyfikacja segmentów
        segments: List[Segment] = segment_scan(beams)
        assign_segment_categories(segments)

        # 3) aktualizacja mapy occupancy
        self.grid.update_from_scan(beams, pose)

        # 4) detections HUMAN – jedna detekcja na segment (środek obiektu)
        detections: List[Tuple[float, float]] = []
        for seg in segments:
            if seg.base_category == BeamCategory.HUMAN:
                # center_x, center_y są w układzie lokalnym robota
                xw, yw = self.grid.local_to_world_point(pose, seg.center_x, seg.center_y)
                detections.append((xw, yw))

        # 5) aktualizacja trackera ludzi
        human_tracks: List[HumanTrack] = self.tracker.update(detections, t)

        return ScanResult(
            beams=beams,
            segments=segments,
            human_tracks=human_tracks,
            grid=self.grid,
        )
