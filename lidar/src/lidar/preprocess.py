from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np

from config import R_MIN_M, R_MAX_M


class BeamCategory(Enum):
    # Klasy wiązek po przetwarzaniu
    NONE = "None"
    HUMAN = "Human"
    OBSTACLE = "Obstacle"


@dataclass
class BeamResult:
    theta: float          # kąt wiązki [rad]
    r: float              # odległość [m]
    category: BeamCategory  # wynik klasyfikacji


def classify_beam_by_range(r: float) -> BeamCategory:
    # Klasyfikacja po zasięgu (poza zakresem -> NONE)
    if r < R_MIN_M or r > R_MAX_M:
        return BeamCategory.NONE
    return BeamCategory.OBSTACLE


def preprocess_scan(r: np.ndarray, theta: np.ndarray) -> List[BeamResult]:
    # r, theta (N) -> lista BeamResult
    if r.shape != theta.shape:
        raise ValueError("Wektory r i theta muszą mieć ten sam rozmiar")

    results: List[BeamResult] = []
    for ri, ti in zip(r, theta):
        category = classify_beam_by_range(float(ri))
        beam = BeamResult(theta=float(ti), r=float(ri), category=category)
        results.append(beam)

    return results


def results_to_array(beams: List[BeamResult]) -> np.ndarray:
    # Lista BeamResult -> tablica N x 3: [theta, r, category_int]
    # category_int: 0=None, 1=Human, 2=Obstacle

    def cat_to_int(cat: BeamCategory) -> int:
        if cat == BeamCategory.NONE:
            return 0
        elif cat == BeamCategory.HUMAN:
            return 1
        elif cat == BeamCategory.OBSTACLE:
            return 2
        return 0

    N = len(beams)
    arr = np.zeros((N, 3), dtype=float)

    for i, b in enumerate(beams):
        arr[i, 0] = b.theta
        arr[i, 1] = b.r
        arr[i, 2] = cat_to_int(b.category)

    return arr
