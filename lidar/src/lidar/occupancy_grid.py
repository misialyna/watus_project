from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple

import numpy as np

from config import MIN_HITS_FOR_STATIC, HIT_DECAY_FREE
from lidar.preprocess import BeamResult, BeamCategory


class CellType(IntEnum):
    UNKNOWN = 0
    FREE = 1
    STATIC_OBSTACLE = 2
    HUMAN = 3


class CellDanger(IntEnum):
    NO_DANGER = 0
    DANGER = 1


@dataclass
class Pose2D:
    x: float   # pozycja robota [m] w globalnym układzie
    y: float   # pozycja robota [m] w globalnym układzie
    yaw: float # orientacja robota [rad] (0 = oś X)


class OccupancyGrid:
    # 2D mapa komórkowa z typem komórki i flagą zagrożenia

    def __init__(self, width_m: float, height_m: float, cell_size_m: float):
        # inicjalizacja siatki i buforów pomocniczych
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.cell_size = float(cell_size_m)

        self.nx = int(np.round(self.width_m / self.cell_size))
        self.ny = int(np.round(self.height_m / self.cell_size))

        # zakres współrzędnych świata (0,0) w środku mapy
        self.x_min = -self.width_m / 2.0
        self.y_min = -self.height_m / 2.0

        self.cell_type = np.full((self.ny, self.nx), CellType.UNKNOWN, dtype=np.int8)
        self.cell_danger = np.full((self.ny, self.nx), CellDanger.NO_DANGER, dtype=np.int8)

        # liczba trafień przeszkody w komórce (do uznania za statyczną)
        self.obstacle_hits = np.zeros((self.ny, self.nx), dtype=np.uint8)

    # Konwersje współrzędnych
    def world_to_cell(self, x: float, y: float) -> Tuple[int, int] | None:
        # świat (metry) -> indeksy komórki (ix, iy) lub None gdy poza mapą
        ix = int(np.floor((x - self.x_min) / self.cell_size))
        iy = int(np.floor((y - self.y_min) / self.cell_size))

        if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny:
            return None

        return ix, iy

    # Aktualizacja komórek
    def _set_cell_type(self, ix: int, iy: int, new_type: CellType):
        # ustawia typ komórki z prostym priorytetem (HUMAN / STATIC / FREE)
        current = CellType(self.cell_type[iy, ix])

        if new_type == CellType.FREE:
            if current in (CellType.UNKNOWN, CellType.FREE):
                self.cell_type[iy, ix] = CellType.FREE
            return

        if new_type == CellType.STATIC_OBSTACLE:
            if current != CellType.HUMAN:
                self.cell_type[iy, ix] = CellType.STATIC_OBSTACLE
            return

        if new_type == CellType.HUMAN:
            if current != CellType.STATIC_OBSTACLE:
                self.cell_type[iy, ix] = CellType.HUMAN
            return

    def _mark_danger_around(self, ix: int, iy: int):
        # ustawia DANGER w (ix,iy) i sąsiadach 8-neighborhood
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                jx = ix + dx
                jy = iy + dy
                if 0 <= jx < self.nx and 0 <= jy < self.ny:
                    self.cell_danger[jy, jx] = CellDanger.DANGER

    def _beam_to_local_xy(self, beam: BeamResult) -> Tuple[float, float]:
        # BeamResult -> (x,y) w układzie robota
        x = beam.r * np.cos(beam.theta)
        y = beam.r * np.sin(beam.theta)
        return x, y

    def _local_to_world(self, pose: Pose2D, x_local: float, y_local: float) -> Tuple[float, float]:
        # lokalne (x,y) robota -> globalne (x,y)
        cos_yaw = np.cos(pose.yaw)
        sin_yaw = np.sin(pose.yaw)

        x_world = pose.x + cos_yaw * x_local - sin_yaw * y_local
        y_world = pose.y + sin_yaw * x_local + cos_yaw * y_local

        return x_world, y_world

    def _mark_ray_free(self, pose: Pose2D, x_hit: float, y_hit: float):
        # czyści komórki po drodze promienia i w razie potrzeby oznacza je jako FREE
        x0, y0 = pose.x, pose.y
        x1, y1 = x_hit, y_hit

        dist = np.hypot(x1 - x0, y1 - y0)
        if dist < 1e-6:
            return

        n_steps = int(dist / self.cell_size)
        if n_steps < 1:
            return

        for k in range(1, n_steps):  # nie zaznaczamy punktu końcowego
            t = k / n_steps
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            idx = self.world_to_cell(x, y)
            if idx is None:
                continue

            ix, iy = idx

            # wygaszanie licznika trafień, jeśli promień przeszedł przez komórkę
            if self.obstacle_hits[iy, ix] > 0:
                dec = HIT_DECAY_FREE
                if self.obstacle_hits[iy, ix] <= dec:
                    self.obstacle_hits[iy, ix] = 0
                else:
                    self.obstacle_hits[iy, ix] -= dec

            # jeśli licznik == 0 i nie ma tam człowieka, komórka jest FREE
            if CellType(self.cell_type[iy, ix]) != CellType.HUMAN and self.obstacle_hits[iy, ix] == 0:
                self.cell_type[iy, ix] = CellType.FREE

    # Główna aktualizacja z jednego skanu
    def update_from_scan(self, beams: List[BeamResult], pose: Pose2D):
        # aktualizuje mapę na podstawie pojedynczego skanu wiązek
        for b in beams:
            if b.category == BeamCategory.NONE:
                continue

            x_local, y_local = self._beam_to_local_xy(b)
            x_world, y_world = self._local_to_world(pose, x_local, y_local)

            hit_idx = self.world_to_cell(x_world, y_world)
            if hit_idx is None:
                continue
            ix_hit, iy_hit = hit_idx

            self._mark_ray_free(pose, x_world, y_world)

            if b.category == BeamCategory.OBSTACLE:
                if self.obstacle_hits[iy_hit, ix_hit] < 255:
                    self.obstacle_hits[iy_hit, ix_hit] += 1

                if self.obstacle_hits[iy_hit, ix_hit] >= MIN_HITS_FOR_STATIC:
                    self._set_cell_type(ix_hit, iy_hit, CellType.STATIC_OBSTACLE)

            elif b.category == BeamCategory.HUMAN:
                self._set_cell_type(ix_hit, iy_hit, CellType.HUMAN)

            self._mark_danger_around(ix_hit, iy_hit)

    # Statystyki / debug
    def count_cells_by_type(self) -> dict[CellType, int]:
        # zwraca liczbę komórek dla każdego CellType
        counts = {}
        for ct in CellType:
            counts[ct] = int(np.sum(self.cell_type == ct))
        return counts

    def count_danger_cells(self) -> int:
        # zwraca liczbę komórek z flagą DANGER
        return int(np.sum(self.cell_danger == CellDanger.DANGER))

    def beam_to_world_point(self, beam: BeamResult, pose: Pose2D) -> Tuple[float, float]:
        # BeamResult + Pose2D -> punkt (x,y) w świecie
        x_local, y_local = self._beam_to_local_xy(beam)
        x_world, y_world = self._local_to_world(pose, x_local, y_local)
        return x_world, y_world

    def local_to_world_point(self, pose: Pose2D, x_local: float, y_local: float) -> Tuple[float, float]:
        # lokalne (x,y) robota -> globalne (x,y)
        return self._local_to_world(pose, x_local, y_local)
