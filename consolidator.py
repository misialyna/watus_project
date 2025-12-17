import json
import math

# Stała - połowa kąta widzenia kamery:
CAMERA_FOV_HALF = 51.0  # (102° / 2)

def compute_lidar_angle(lidar_track):
    """Oblicza kąt poziomy obiektu względem osi kamery na podstawie danych LIDAR."""
    x_offset = lidar_track["last_position"][0]   # odchylenie w lewo (+) / prawo (−)
    y_forward = lidar_track["last_position"][1]  # odległość do przodu
    # Kąt w stopniach (atan2 zwraca radiany, dodatni kąt dla x_offset > 0 (lewo), ujemny dla prawo)
    angle_deg = math.degrees(math.atan2(x_offset, y_forward))
    return angle_deg


# Pętla główna konsolidatora (np. wykonywana w interwałach ułamka sekundy)
while True:
    # 1. Odczytaj bieżące dane z plików JSON
    with open("lidar.jsonl", "r") as f:
        lidar_data = json.load(f)
    with open("camera.jsonl", "r") as f:
        camera_data = json.load(f)

    combined_tracks = []  # lista wynikowych ścieżek (osób)
    # 2. Przetwarzaj każdą ścieżkę z LIDAR (w założeniu jedna ścieżka w danej chwili)
    for track in lidar_data.get("tracks", []):
        combined_entry = {
            "id": track["id"],
            "type": track.get("type", "Unknown"),
            "history": track.get("history", []).copy(),       # pełna historia pozycji
            "last_position": track.get("last_position"),      # obecna pozycja [x, y]
            "last_update_time": track.get("last_update_time")
        }
        # 3. Jeśli kamera aktualnie widzi osobę, a jej kąt pokrywa się z tą ścieżką LIDAR:
        if camera_data.get("countOfPeople", 0) > 0 and camera_data["objects"]:
            cam_obj = camera_data["objects"][0]  # (założenie: jedna osoba w objects)
            # Sprawdź zgodność kąta (opcjonalnie, przy wielu osobach):
            angle_camera = cam_obj.get("angle")
            angle_lidar = compute_lidar_angle(track)
            if abs(angle_camera - angle_lidar) <= CAMERA_FOV_HALF:
                # Dodaj informacje z kamery do wpisu
                if cam_obj.get("isPerson"):
                    combined_entry["gender"] = cam_obj.get("type")   # np. "male" lub "female"
                else:
                    combined_entry["object_type"] = cam_obj.get("type")
                # Pozycja w obrazie z kamery (ramka obiektu):
                combined_entry["camera_bbox"] = {
                    "left": cam_obj.get("left"),
                    "top": cam_obj.get("top"),
                    "width": cam_obj.get("width"),
                    "height": cam_obj.get("height")
                }
                combined_entry["camera_angle"] = angle_camera
        # Dodaj scalony wpis do listy wynikowej
        combined_tracks.append(combined_entry)

    # 4. Zapisz połączone dane do pliku consolidator.json
    output_data = {"tracks": combined_tracks}
    with open("consolidator.json", "w") as f:
        json.dump(output_data, f, indent=2)

    # (Przerwa lub synchronizacja z nowymi danymi, np. sleep na ułamek sekundy)
    time.sleep(0.1)
