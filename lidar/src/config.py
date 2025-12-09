from __future__ import annotations

# =========================
# MAPA / OCCUPANCY GRID
# =========================

# Lidar STL-19P ma zasięg do ~12 m,
# więc mapa 24 x 24 m daje po 12 m w każdą stronę od pozycji startowej robota.
MAP_WIDTH_M: float = 24.0      # szerokość mapy w metrach (oś X)
MAP_HEIGHT_M: float = 24.0     # wysokość mapy w metrach (oś Y)

# Rozmiar komórki – 0.25 m = 25 cm (dokładniejsza mapa niż 0.5 m)
CELL_SIZE_M: float = 0.25

# Ile razy wiązka musi trafić w tę samą komórkę,
# żebyśmy uznali ją za statyczną przeszkodę na mapie
MIN_HITS_FOR_STATIC: int = 3

# O ile zmniejszamy licznik trafień na promieniu FREE
HIT_DECAY_FREE: int = 0


# =========================
# LIDAR / HARDWARE
# =========================

# Port szeregowy lidara – na Windows np. "COM3", "COM4" itd.
LIDAR_PORT: str = "COM3"

# Parametry UART STL-19P (dla nas: baudrate + timeout)
LIDAR_BAUDRATE: int = 230400
LIDAR_TIMEOUT: float = 1.0   # sekundy (można później zmniejszyć jak będzie stabilnie)
FULL_SCAN_PACKETS: int = 60

# =========================
# PREPROCESS (zasięg wiązek)
# =========================

# Fizyczny zakres STL-19P: ok. 0.03–12 m.
# My ignorujemy jeszcze najbliższe 5 cm jako martwą strefę.
R_MIN_M: float = 0.05        # poniżej 5 cm ignorujemy (martwa strefa)
R_MAX_M: float = 12.0        # powyżej 12 m ignorujemy (poza zasięgiem sensownym)


# =========================
# SEGMENTACJA
# =========================

# Maksymalna odległość (m) między sąsiednimi punktami w segmencie.
SEG_MAX_DISTANCE_JUMP_M: float = 1.0

# Maksymalna różnica kąta (w stopniach) między sąsiednimi wiązkami
# żeby nadal traktować je jako jeden segment.
SEG_MAX_ANGLE_JUMP_DEG: float = 10.0

# Minimalna liczba wiązek, żeby segment był uznany za obiekt.
SEG_MIN_BEAMS: int = 2


# =========================
# TRACKING LUDZI
# =========================

# Maksymalna odległość (m) między nową detekcją a istniejącym trackiem
# żeby uznać, że to ten sam człowiek.
TRACK_MAX_MATCH_DISTANCE_M: float = 1.0

# Ile kolejnych skanów możemy kogoś "nie widzieć", zanim skasujemy track.
TRACK_MAX_MISSED: int = 5

TRACK_MIN_MOVING_SPEED_M_S = 0.15  # prędkość od której uznajemy obiekt za ruchomy

