import math
from typing import Tuple

def calc_obj_angle(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    image_width: int,
    fov_deg: float = 60.0,
) -> float:
    """
    Szacuje poziomy kąt (w stopniach) między frontem kamery (oś optyczna)
    a środkiem obiektu. Pomija detekcję – zkłada, że p1 i p2 to dwa
    dowolne przeciwległe narożniki prostokąta obiektu w obrazie.

    Parametry:
      p1, p2      : (x, y) w pikselach
      image_size  : (width, height) obrazu w pikselach
      fov_deg     : poziomy kąt widzenia kamery w stopniach

    Zwraca:
      Kąt w stopniach (dodatni w prawo, ujemny w lewo).
    """
    (x1, y1), (x2, y2) = p1, p2
    W = image_width

    # Środek prostokąta obiektu
    x_c = 0.5 * (x1 + x2)

    # Środek obrazu (oś optyczna)
    c_x = W / 2.0

    # Ogniskowa w pikselach wyliczona z FOV
    fov_rad = math.radians(fov_deg)
    fx = (W / 2.0) / math.tan(fov_rad / 2.0)

    # Kąt w radianach -> stopnie
    theta_rad = math.atan((x_c - c_x) / fx)
    theta_deg = math.degrees(theta_rad)
    return theta_deg

# --- przykład użycia ---
if __name__ == "__main__":
    # prostokąt obiektu (dwa rogi), obraz 1920x1080, FOV=60°
    p1 = (100, 300)
    p2 = (200, 600)
    angle = calc_obj_angle(p1, p2, (1920, 1080), 60.0)
    print(f"Kąt względem osi kamery: {angle:.2f}°")
