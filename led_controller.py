
import time

# Aby uniknąć błędu na maszynach innych niż Raspberry Pi,
# używamy "fałszywej" biblioteki GPIO do testowania.
try:
    import RPi.GPIO as GPIO
    IS_RASPBERRY_PI = True
except (ImportError, RuntimeError):
    print("OSTRZEŻENIE: Biblioteka RPi.GPIO nie została znaleziona. Używam trybu symulacyjnego.")
    IS_RASPBERRY_PI = False
    # Definiujemy fałszywą klasę GPIO do celów deweloperskich na PC/Mac
    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        HIGH = 1
        LOW = 0
        def setmode(self, mode):
            print(f"[SIM] GPIO mode set to {mode}")
        def setup(self, pin, mode):
            print(f"[SIM] Pin {pin} set to mode {mode}")
        def output(self, pin, state):
            color = "UNKNOWN"
            if pin == 17: color = "GREEN"
            if pin == 27: color = "RED"
            status = "ON" if state == GPIO.HIGH else "OFF"
            print(f"[SIM] LED Pin {pin} ({color}) set to {status}")
        def cleanup(self, pins=None):
            print(f"[SIM] GPIO cleanup for pins {pins}")
        def setwarnings(self, value):
            pass
    GPIO = MockGPIO()


class LEDController:
    """
    Zarządza stanem diody LED (np. RGB) do sygnalizacji statusu aplikacji.
    """
    def __init__(self, green_pin=17, red_pin=27):
        self.GREEN_PIN = green_pin
        self.RED_PIN = red_pin
        self.is_pi = IS_RASPBERRY_PI
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.GREEN_PIN, GPIO.OUT)
        GPIO.setup(self.RED_PIN, GPIO.OUT)
        
        self.off()  # Domyślnie dioda jest wyłączona

    def listening(self):
        """Sygnalizuje nasłuchiwanie (zielony ON, czerwony OFF)."""
        GPIO.output(self.GREEN_PIN, GPIO.HIGH)
        GPIO.output(self.RED_PIN, GPIO.LOW)

    def processing_or_speaking(self):
        """Sygnalizuje przetwarzanie lub mówienie (czerwony ON, zielony OFF)."""
        GPIO.output(self.GREEN_PIN, GPIO.LOW)
        GPIO.output(self.RED_PIN, GPIO.HIGH)

    def off(self):
        """Całkowicie wyłącza diodę."""
        GPIO.output(self.GREEN_PIN, GPIO.LOW)
        GPIO.output(self.RED_PIN, GPIO.LOW)

    def cleanup(self):
        """Resetuje piny GPIO do stanu początkowego."""
        self.off()
        if self.is_pi:
            GPIO.cleanup([self.GREEN_PIN, self.RED_PIN])

# Przykład użycia i testowanie modułu
if __name__ == '__main__':
    print("Testowanie modułu LEDController...")
    led = LEDController()
    try:
        print("Stan: Nasłuchiwanie (Zielony)")
        led.listening()
        time.sleep(2)
        
        print("Stan: Przetwarzanie (Czerwony)")
        led.processing_or_speaking()
        time.sleep(2)
        
        print("Stan: Wyłączony")
        led.off()
        time.sleep(1)
    except KeyboardInterrupt:
        print("Test przerwany.")
    finally:
        print("Czyszczenie GPIO.")
        led.cleanup()
