from hardware.lidar_driver import init_lidar, get_next_scan
from config import LIDAR_PORT


def main():
    print(f"Próbuję otworzyć port: {LIDAR_PORT}")
    try:
        init_lidar(port=LIDAR_PORT)
        print("Port otwarty poprawnie.")
    except Exception as e:
        print("Błąd przy otwieraniu portu:")
        print(repr(e))
        return

    print("Próbuję odczytać jeden mini-skan (12 punktów)...")
    try:
        r, theta = get_next_scan()
        print("Odczyt danych zakończony powodzeniem.")
        print("r (m):      ", r)
        print("theta (rad):", theta)
    except Exception as e:
        print("Port otwarty, ale wystąpił błąd przy czytaniu danych:")
        print(repr(e))


if __name__ == "__main__":
    main()
