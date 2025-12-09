from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np

from config import (
    TRACK_MAX_MATCH_DISTANCE_M,
    TRACK_MAX_MISSED,
    TRACK_MIN_MOVING_SPEED_M_S,
)



@dataclass
class HumanTrack:
    # Track śledzonej osoby w układzie świata
    id: int
    type: str = "Human"

    history: List[Tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)
    last_position: Tuple[float, float] = (0.0, 0.0)
    last_update_time: float = 0.0

    missed_count: int = 0          # kolejne skany bez dopasowania
    vx: float = 0.0                # prędkość w osi X [m/s]
    vy: float = 0.0                # prędkość w osi Y [m/s]
    hits_count: int = 1            # ile razy track był zaktualizowany
    confirmed: bool = False        # czy track jest potwierdzony

    @property
    def x(self) -> float:
        return float(self.last_position[0])

    @property
    def y(self) -> float:
        return float(self.last_position[1])

    @property
    def speed(self) -> float:
        # moduł prędkości [m/s]
        return float(np.hypot(self.vx, self.vy))

    @property
    def heading_rad(self) -> float:
        # kierunek ruchu [rad], 0 = oś X, CCW
        if self.speed < 1e-6:
            return 0.0
        return float(np.arctan2(self.vy, self.vx))

    @property
    def age(self) -> int:
        # liczba aktualizacji (długość historii)
        return len(self.history)

    @property
    def is_active(self) -> bool:
        # aktywne są tylko tracki potwierdzone i nieprzegapione zbyt długo
        return self.confirmed and self.missed_count <= TRACK_MAX_MISSED

    @property
    def is_moving(self) -> bool:
        # obiekt uznajemy za ruchomy dopiero od pewnej prędkości
        return self.speed >= TRACK_MIN_MOVING_SPEED_M_S


    def predict_position(self, t: float) -> Tuple[float, float]:
        # predykcja pozycji przy stałej prędkości w chwili t
        t = float(t)
        dt = t - float(self.last_update_time)
        if dt <= 0.0:
            return self.last_position
        px = self.x + self.vx * dt
        py = self.y + self.vy * dt
        return float(px), float(py)


    def predict_future_points(
        self,
        t_start: float,
        horizon_s: float = 0.8,
        step_s: float = 0.2,
    ) -> List[Tuple[float, float, float]]:
        # krótka trajektoria w przyszłość (t, x, y) przy stałej prędkości
        t_start = float(t_start)
        horizon_s = max(0.0, float(horizon_s))
        step_s = max(1e-3, float(step_s))

        if not self.is_moving or horizon_s <= 0.0:
            return []

        points: List[Tuple[float, float, float]] = []
        n_steps = int(horizon_s / step_s) + 1

        for i in range(1, n_steps + 1):
            dt = step_s * i
            t_pred = t_start + dt
            x_pred = self.x + self.vx * dt
            y_pred = self.y + self.vy * dt
            points.append((float(t_pred), float(x_pred), float(y_pred)))

        return points

    def prediction_radius(
        self,
        dt: float,
        base: float = 0.1,
        growth: float = 0.25,
        max_speed_for_uncertainty: float = 3.0,
    ) -> float:
        # prosty promień niepewności predykcji [m]
        dt = max(0.0, float(dt))
        v = min(self.speed, max_speed_for_uncertainty)
        return float(base + growth * dt * (1.0 + v))


class HumanTracker:
    # Multi-target tracker z NN, predykcją, wygładzaniem i etapem potwierdzania

    def __init__(
        self,
        max_match_distance: float = TRACK_MAX_MATCH_DISTANCE_M,
        max_missed: float = TRACK_MAX_MISSED,
        pos_alpha: float = 0.6,
        vel_alpha: float = 0.5,
        min_confirm_hits: int = 3,
        max_missed_tentative: int = 2,
    ):
        # max_match_distance    - maks. dystans dopasowania [m]
        # max_missed            - ile skanów można "zgubić" potwierdzony track
        # pos_alpha, vel_alpha  - wagi wygładzania pozycji i prędkości
        # min_confirm_hits      - ile aktualizacji zanim track będzie potwierdzony
        # max_missed_tentative  - ile skanów może przeżyć niepotwierdzony track
        self.max_match_distance = float(max_match_distance)
        self.max_missed_confirmed = int(max_missed)
        self.max_missed_tentative = int(max_missed_tentative)
        self.pos_alpha = float(pos_alpha)
        self.vel_alpha = float(vel_alpha)
        self.min_confirm_hits = int(min_confirm_hits)

        self.tracks: List[HumanTrack] = []
        self._next_id: int = 1

    # --- pomocnicze ---

    def _create_track(self, t: float, x: float, y: float) -> HumanTrack:
        # tworzy nowy, jeszcze niepotwierdzony track
        t = float(t)
        xf = float(x)
        yf = float(y)

        track = HumanTrack(
            id=self._next_id,
            type="Human",
            history=[(t, xf, yf)],
            last_position=(xf, yf),
            last_update_time=t,
            missed_count=0,
            vx=0.0,
            vy=0.0,
            hits_count=1,
            confirmed=(self.min_confirm_hits <= 1),
        )
        self._next_id += 1
        self.tracks.append(track)
        return track

    def _find_best_match(self, x: float, y: float, t: float) -> Optional[HumanTrack]:
        # zwraca najbliższy track względem predykowanej pozycji
        xf = float(x)
        yf = float(y)
        t = float(t)

        best_track: Optional[HumanTrack] = None
        best_dist = self.max_match_distance

        for tr in self.tracks:
            # track może być zarówno potwierdzony, jak i nie
            px, py = tr.predict_position(t)
            dx = xf - px
            dy = yf - py
            d = float(np.hypot(dx, dy))

            if d < best_dist:
                best_dist = d
                best_track = tr

        return best_track

    def _update_track_state(
        self,
        track: HumanTrack,
        t: float,
        meas_x: float,
        meas_y: float,
    ) -> None:
        # aktualizacja pozycji i prędkości tracka
        t = float(t)
        mx = float(meas_x)
        my = float(meas_y)

        dt = t - float(track.last_update_time)
        if dt <= 1e-6:
            track.last_position = (mx, my)
            track.last_update_time = t
            track.history.append((t, mx, my))
            track.missed_count = 0
            track.hits_count += 1
            if not track.confirmed and track.hits_count >= self.min_confirm_hits:
                track.confirmed = True
            return

        raw_vx = (mx - track.x) / dt
        raw_vy = (my - track.y) / dt

        # minimalna prędkość, żeby tłumić drobny szum
        min_speed = 0.3  # m/s
        raw_speed = float(np.hypot(raw_vx, raw_vy))
        if raw_speed < min_speed:
            raw_vx = 0.0
            raw_vy = 0.0

        track.vx = (1.0 - self.vel_alpha) * track.vx + self.vel_alpha * raw_vx
        track.vy = (1.0 - self.vel_alpha) * track.vy + self.vel_alpha * raw_vy

        new_x = (1.0 - self.pos_alpha) * track.x + self.pos_alpha * mx
        new_y = (1.0 - self.pos_alpha) * track.y + self.pos_alpha * my

        track.last_position = (float(new_x), float(new_y))
        track.last_update_time = t
        track.history.append((t, float(new_x), float(new_y)))
        track.missed_count = 0
        track.hits_count += 1

        if not track.confirmed and track.hits_count >= self.min_confirm_hits:
            track.confirmed = True

    # --- API trackera ---

    def update(self, detections: List[Tuple[float, float]], t: float) -> List[HumanTrack]:
        # aktualizuje listę tracków na podstawie detekcji (x,y) w chwili t
        t = float(t)
        updated_tracks: set[int] = set()

        # dopasowanie detekcji do istniejących tracków
        for (x, y) in detections:
            xf = float(x)
            yf = float(y)

            track = self._find_best_match(xf, yf, t)
            if track is not None:
                self._update_track_state(track, t, xf, yf)
                updated_tracks.add(track.id)
            else:
                new_track = self._create_track(t, xf, yf)
                updated_tracks.add(new_track.id)

        # inkrementacja missed_count dla niedopasowanych tracków
        for tr in self.tracks:
            if tr.id not in updated_tracks:
                tr.missed_count += 1

        # usuwanie tracków niepotwierdzonych i potwierdzonych osobno
        remaining: List[HumanTrack] = []
        for tr in self.tracks:
            if tr.confirmed:
                limit = self.max_missed_confirmed
            else:
                limit = self.max_missed_tentative

            if tr.missed_count <= limit:
                remaining.append(tr)

        self.tracks = remaining

        # zwracamy tylko potwierdzone tracki
        return [tr for tr in self.tracks if tr.confirmed]

    def get_future_predictions(
        self,
        t_now: float,
        horizon_s: float = 0.8,
        step_s: float = 0.2,
    ) -> Dict[int, List[Tuple[float, float, float]]]:
        # zwraca przewidywane trajektorie (t,x,y) dla aktywnych tracków
        t_now = float(t_now)
        preds: Dict[int, List[Tuple[float, float, float]]] = {}
        for tr in self.tracks:
            if not tr.is_active:
                continue
            traj = tr.predict_future_points(
                t_start=t_now,
                horizon_s=horizon_s,
                step_s=step_s,
            )
            preds[tr.id] = traj
        return preds
