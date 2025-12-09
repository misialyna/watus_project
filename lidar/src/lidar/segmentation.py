from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from lidar.preprocess import BeamResult, BeamCategory
from config import (
    SEG_MAX_DISTANCE_JUMP_M,
    SEG_MAX_ANGLE_JUMP_DEG,
    SEG_MIN_BEAMS,
)


@dataclass
class Segment:
    id: int
    beams: List[BeamResult]

    center_x: float
    center_y: float
    mean_r: float
    mean_theta: float
    length: float          # przybliżona długość obiektu [m]
    base_category: BeamCategory  # kategoria segmentu (np. OBSTACLE / HUMAN)


def beam_to_xy_local(beam: BeamResult) -> Tuple[float, float]:
    # BeamResult -> (x,y) w układzie lokalnym robota
    x = beam.r * np.cos(beam.theta)
    y = beam.r * np.sin(beam.theta)
    return x, y


def segment_scan(
    beams: List[BeamResult],
    max_distance_jump: float = SEG_MAX_DISTANCE_JUMP_M,
    max_angle_jump: float = np.deg2rad(SEG_MAX_ANGLE_JUMP_DEG),
    min_beams_in_segment: int = SEG_MIN_BEAMS,
) -> List[Segment]:
    # Dzieli skan (BeamResult) na segmenty na podstawie skoków odległości i kąta
    segments: List[Segment] = []
    current_beams: List[BeamResult] = []

    prev_beam: BeamResult | None = None
    prev_x: float | None = None
    prev_y: float | None = None

    def finalize_segment(seg_id: int, seg_beams: List[BeamResult]) -> Segment | None:
        # Tworzy Segment, liczy środek, długość itp.
        if len(seg_beams) < min_beams_in_segment:
            return None

        xs = []
        ys = []
        rs = []
        thetas = []

        for b in seg_beams:
            x, y = beam_to_xy_local(b)
            xs.append(x)
            ys.append(y)
            rs.append(b.r)
            thetas.append(b.theta)

        xs = np.array(xs)
        ys = np.array(ys)
        rs = np.array(rs)
        thetas = np.array(thetas)

        center_x = float(xs.mean())
        center_y = float(ys.mean())
        mean_r = float(rs.mean())
        mean_theta = float(thetas.mean())

        # długość = przekątna prostokąta otaczającego
        length = float(
            np.sqrt((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2)
        )

        base_cat = seg_beams[0].category

        return Segment(
            id=seg_id,
            beams=seg_beams,
            center_x=center_x,
            center_y=center_y,
            mean_r=mean_r,
            mean_theta=mean_theta,
            length=length,
            base_category=base_cat,
        )

    seg_id_counter = 1

    for b in beams:
        # pomijamy wiązki NONE
        if b.category == BeamCategory.NONE:
            if current_beams:
                seg = finalize_segment(seg_id_counter, current_beams)
                if seg is not None:
                    segments.append(seg)
                    seg_id_counter += 1
                current_beams = []
                prev_beam = None
                prev_x = None
                prev_y = None
            continue

        x, y = beam_to_xy_local(b)

        if prev_beam is None:
            # start nowego segmentu
            current_beams = [b]
            prev_beam = b
            prev_x, prev_y = x, y
            continue

        # skok w kącie i w przestrzeni między kolejnymi punktami
        angle_jump = abs(b.theta - prev_beam.theta)
        dist_jump = np.hypot(x - prev_x, y - prev_y)

        # jeśli skok za duży -> zamykamy segment, zaczynamy nowy
        if angle_jump > max_angle_jump or dist_jump > max_distance_jump:
            seg = finalize_segment(seg_id_counter, current_beams)
            if seg is not None:
                segments.append(seg)
                seg_id_counter += 1
            current_beams = [b]
        else:
            # kontynuujemy segment
            current_beams.append(b)

        prev_beam = b
        prev_x, prev_y = x, y

    # zamknięcie ostatniego segmentu
    if current_beams:
        seg = finalize_segment(seg_id_counter, current_beams)
        if seg is not None:
            segments.append(seg)

    return segments


def classify_segment_type(seg: Segment) -> BeamCategory:
    # zbyt blisko lub zbyt daleko -> nie człowiek
    if seg.mean_r < 0.8 or seg.mean_r > 4.0:
        return BeamCategory.OBSTACLE

    # zbyt mały albo zbyt duży segment -> nie człowiek
    if seg.length < 0.3 or seg.length > 0.7:
        return BeamCategory.OBSTACLE

    # bardzo mało wiązek -> raczej szum / krawędź
    if len(seg.beams) < 6:
        return BeamCategory.OBSTACLE

    return BeamCategory.HUMAN

def assign_segment_categories(segments: List[Segment]) -> None:
    # Ustawia kategorię segmentu i wszystkich jego wiązek
    for seg in segments:
        new_cat = classify_segment_type(seg)
        seg.base_category = new_cat
        for b in seg.beams:
            b.category = new_cat
