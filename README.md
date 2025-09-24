# WATUS – Voice Frontend (Watus + Reporter)

Low-latency voice frontend z rozpoznawaniem lidera (ECAPA / SpeechBrain), transkrypcją (Whisper via Faster-Whisper),
kolejką ZMQ oraz TTS (Piper). Łączy się z lokalnym backendem LLM (`watus-ai`) przez HTTP.

<p align="center">
  <img src="docs/arch.png" alt="Architektura Watus + Reporter" width="820">
</p>

---

## Spis treści

1. [Opis](#opis)
2. [Wymagania](#wymagania)
3. [Szybki start (TL;DR)](#szybki-start-tldr)
4. [Instalacja — krok po kroku](#instalacja--krok-po-kroku)
   - [A. Biblioteki systemowe audio](#a-biblioteki-systemowe-audio)
   - [B. Wirtualne środowisko (venv)](#b-wirtualne-środowisko-venv)
   - [C. Python dependencies (CPU/GPU)](#c-python-dependencies-cpugpu)
   - [D. ECAPA / SpeechBrain (wymagane)](#d-ecapa--speechbrain-wymagane)
   - [E. Konfiguracja `.env`](#e-konfiguracja-env)
   - [F. Piper (binarka + model)](#f-piper-binarka--model)
5. [Uruchomienie](#uruchomienie)
6. [Konfiguracja i parametry](#konfiguracja-i-parametry)
7. [Troubleshooting](#troubleshooting)

---

## Opis

- **`watus.py`** – nasłuch audio (PortAudio), VAD (WebRTC), rozpoznawanie mówcy (ECAPA/SpeechBrain),
  transkrypcja (Faster-Whisper), zapis do `dialog.jsonl`, PUB `dialog.leader` **tylko dla lidera**, SUB `tts.speak` i odtwarzanie (Piper).
- **`reporter.py`** – SUB `dialog.leader`, buduje **meldunek** (opis + meta + czas systemowy + scenariusz + kamera),
  POST do backendu LLM (`/api1/process_question`), PUB `tts.speak` z odpowiedzią (dla Watusia). Ma retry dla 429/timeout i krótki fallback TTS.

Przepływ (skrótowo):  
**Mic → Watus (VAD + STT + ECAPA) → dialog.jsonl → (leader→ZMQ) → Reporter → HTTP→LLM → (odp) ZMQ→Watus → TTS (Piper)**

---

## Wymagania

- Python 3.11+
- Wolne porty: 7780, 7781, 8781, 8000 (LLM).
- **Piper** (binarka + model ONNX + config) i **piper-phonemize (biblioteka fonemizacji).**
  - **Głosy Piper (pl):**
    - **Darkman medium:**<br>
    https://huggingface.co/rhasspy/piper-voices/tree/main/pl/pl_PL/darkman/medium
    - **Piper releases (binarki):**<br>
    https://github.com/rhasspy/piper/releases/tag/2023.11.14-2
    - **piper-phonemize (biblioteki .dll/.so/.dylib):**<br>
    https://github.com/rhasspy/piper-phonemize/releases/tag/2023.11.14-4
- Biblioteki systemowe: PortAudio + libsndfile.
- **ECAPA (SpeechBrain) — wymagane**; opcjonalnie GPU:
  - **Windows/Linux (NVIDIA):** PyTorch z CUDA (instrukcje niżej),
  - **macOS (Intel/Apple Silicon):** PyTorch z MPS (akceleracja ECAPA); Faster-Whisper działa głównie na CPU na macOS.

---

## Szybki start (TL;DR)

```bash
git clone https://github.com/misialyna/watus_project.git
cd watus_project
````
## venv
### macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell):
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
## Deps + .env
```bash
pip install -U pip wheel
pip install -r requirements.txt
cp .env.example .env
# uzupełnij: PIPER_BIN, PIPER_MODEL, PIPER_CONFIG, urządzenia audio
```
## Backend LLM (repo `watus-ai`)
link do repozytorium:<br>
⁦https://github.com/pasjonatprogramowania/watus-ai⁩
```bash
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```
## Ten projekt:
(najlepiej w osobnych terminalach)
```bash
python3 reporter.py
```
```bash
python3 watus.py
```
# Instalacja — krok po kroku
## A. Biblioteki systemowe audio
### macOS
```bash
brew install portaudio libsndfile
```
### Ubuntu/Debian/Raspberry Pi
```bash
sudo apt update
sudo apt install -y libportaudio2 libsndfile1
```
### Windows
Z reguły nie trzeba nic doinstalowywać; w razie błędów z PortAudio – doinstaluj pakiet PortAudio z binarki/menedżera paczek.
# B. Wirtualne środowisko (venv)
### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
```
### Windows (PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
# C. Python dependencies (CPU/GPU)
### Plik requirements.txt (w repo) zawiera minimalny zestaw, z ECAPA:
numpy==1.26.4<br>
sounddevice==0.4.6<br>
soundfile==0.12.1<br>
webrtcvad==2.0.10<br>
faster-whisper==1.0.3<br>
pyzmq==25.1.2<br>
requests==2.32.3<br>
python-dotenv==1.0.1<br>
fastapi==0.112.2<br>
uvicorn==0.30.6<br>
PyYAML==6.0.1<br>
speechbrain==0.5.16<br>
## Instalacja (wszystkie systemy):
```bash
pip install -U pip wheel
pip install -r requirements.txt
```
**speechbrain** zwykle dociąga odpowiedni **PyTorch**.
Jeżeli pip nie zainstaluje Torcha — doinstaluj go wg sekcji D.
# D. ECAPA / SpeechBrain (wymagane)
## 1) Windows / Linux (NVIDIA GPU)
### GPU (CUDA 12.1):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
### CPU only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Upewnij się, że Twoje sterowniki NVIDIA wspierają wersję CUDA z koła PyTorch.
## 2) macOS (Intel / Apple Silicon)
### MPS (akceleracja ECAPA na GPU Apple/Intel):
```bash
pip install torch
#(opcjonalnie) pozwól spadać na CPU, gdy MPS nie wspiera op:
export PYTORCH_ENABLE_MPS_FALLBACK=1   # macOS z bash/zsh
```
Faster-Whisper na macOS zwykle działa na CPU (to jest OK – jest szybki).
## 3) Raspberry Pi (ARM)
```bash
pip install torch
```
Jeżeli brak gotowego koła dla tej architektury, rozważ:<br>
- Koła społeczności dla ARM,<br>
- Lub pozostanie przy CPU (ECAPA bywa ciężka na RPi — działa, ale wolniej).
# E. Konfiguracja .env
```bash
cp .env.example .env
```
## Uzupełnij najważniejsze:
### Piper (obowiązkowo)
```bash
    PIPER_BIN=/absolutna/ścieżka/do/venv/bin/piper      #Windows: C:\...\venv\Scripts\piper.exe
    PIPER_MODEL=/absolutna/ścieżka/do/models/pl_PL-*.onnx
    PIPER_CONFIG=/absolutna/ścieżka/do/models/pl_PL-*.onnx.json
```
### Whisper (CPU/GPU)
```bash
    WHISPER_MODEL=small
    WHISPER_DEVICE=cpu         # GPU: cuda   (Windows/Linux z NVIDIA)
    WHISPER_COMPUTE_TYPE=auto  # GPU: float16, CPU: int8/int8_float16/auto
```
### Audio
```bash
    WATUS_INPUT_DEVICE=MacBook      # albo indeks urządzenia (int)
    WATUS_OUTPUT_DEVICE=Speakers
```
### ZMQ / LLM (zostaw jak jest)
```bash
    ZMQ_PUB_ADDR=tcp://127.0.0.1:7780
    ZMQ_SUB_ADDR=tcp://127.0.0.1:7781
    LLM_HTTP_URL=http://127.0.0.1:8000/api1/process_question
```
# F. Piper (binarka + model)
1. Pobierz binarkę Piper dla swojego systemu (z oficjalnego repo wydania).
https://github.com/rhasspy/piper/releases/tag/2023.11.14-2<br>
Wypakuj np. do `models/piper/` i ustaw `PIPER_BIN` w `.env`.<br><br>
2. Pobierz polski model (np. `pl_PL-darkman-medium.onnx` + .`json`).<br>
Polski "darkman/medium":<br>
https://huggingface.co/rhasspy/piper-voices/tree/main/pl/pl_PL/darkman/medium<br>
Zapisz oba pliki w `models/piper/` i ustaw `PIPER_MODEL`, `PIPER_CONFIG` w `.env`.<br><br>
3. **Zainstaluj piper-phonemize** (biblioteki fonemizacji - wymagane przez Piper)<br>
   **Releases:** https://github.com/rhasspy/piper-phonemize/releases/tag/2023.11.14-4
- ### **macOS (Intel/Apple Silicon)**
Pobierz paczkę `piper-phonemize-...-macos-...` i **skopiuj biblioteki** do `models/piper/piper-phonemize/`.<br>
Jeśli Piper zgłosi błąd typu `dyld: Library not loaded: @rpath/libpiper_phonemize.1.dylib:`<br>
```bash
  # Zainstaluj onnxruntime i re2 (wymagane przez phonemizer)
  brew install onnxruntime re2

  # Skopiuj podstawowe dylib-y do katalogu z piperem (najprościej obok binarki)
  cp models/piper/piper-phonemize/lib/libpiper_phonemize.*.dylib models/piper/
  cp models/piper/piper-phonemize/lib/libespeak-ng.*.dylib     models/piper/
  cp models/piper/piper-phonemize/lib/libonnxruntime*.dylib    models/piper/
```
Alternatywnie możesz dodać ścieżki do `DYLD_LIBRARY_PATH`, ale skopiowanie obok binarki jest najprostsze.
- ### **Linux**
Pobierz paczkę `piper-phonemize-...-linux-...` i skopiuj `.so` do katalogu z Piperem **lub** do `/usr/local/lib` (wtedy `ldconfig`).<br>
W razie braków:<br>
```bash
    sudo apt install -y espeak-ng-data libespeak-ng1 libsndfile1
    # jeśli potrzeba: onnxruntime (z repo dystrybucji lub wheel przez pip)
```
- ### **Windows**
Pobierz paczkę `piper-phonemize-...-win-...` i skopiuj `piper_phonemize.dll` oraz zależności **do tego samego katalogu, co** `piper.exe`
(albo dodaj folder z DLL do `PATH`).<br>Najprościej: trzymaj wszystko w `models\piper\`.
# Uruchomienie
### 1. **Backend LLM** (repo `watus-ai`)<br>
link do tego samego repo, co wczesniej:<br>
⁦https://github.com/pasjonatprogramowania/watus-ai⁩
```bash
  uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. **Reporter** (to repo)
```bash
    python3 reporter.py
    #health: http://127.0.0.1:8781/health
````

### 3. **Watus** (to repo)
```bash
  python3 watus.py
```

* Watus loguje: **LISTENING / THINKING / SPEAKING / IDLE**.<br>
`unknown_*` jest zapisywany w `dialog.jsonl`, ale nie wysyłany do LLM.<br>
* `leader_*` jest zapisywany i wysyłany do LLM przez ZMQ.
# Konfiguracja i parametry
## Latencja / VAD
* `WATUS_BLOCKSIZE=320` → 20 ms @ 16 kHz (szybka reakcja)<br>
* `WATUS_VAD_MIN_MS=280`, `WATUS_SIL_MS_END=650` → krótkie segmenty<br>
* `WAIT_REPLY_S=2.0` → po wysyłce pytania czekamy na TTS, nie zbieramy śmieci<br>
## ECAPA (wymagana)
- `SPEAKER_REQUIRE_MATCH=1` → odpowiada tylko liderowi<br>
- progi: `SPEAKER_THRESHOLD`, `SPEAKER_BACK_THRESHOLD`, `SPEAKER_STICKY_S`
## LLM<br>
* `LLM_HTTP_URL=http://127.0.0.1:8000/api1/process_question`, `HTTP_TIMEOUT=20`<br>
* Reporter dodaje `SYS_TIME`/`SCENARIO`/`CAMERA_*` do meldunku (można rozszerzać)

# Troubleshooting
* **Address already in use (7780/7781)** – żyje stary proces. Zamknij stare terminale / zabij `python3 watus.py` / `python3 reporter.py`.<br><br>
* **HTTP 500, a w treści „429 RESOURCE_EXHAUSTED / retry in X”** – brak limit u dostawcy LLM (np. Gemini).<br> Reporter odczyta `retry in X` i spróbuje raz; w razie czego powie krótkie TTS i wróci do nasłuchu - w tym przypadku należy <u>**ponownie wygenerować klucz API Gemini**</u><br><br>
* **Brak TTS / błąd dylib/dll so** – upewnij się, że:
  - zainstalowano **piper-phonemize** (link wyżej),
  - na macOS doinstalowano `onnxruntime` + `re2` przez `brew`,
  - biblioteki `.dylib/.dll/.so` leżą obok `piper` **lub** są na ścieżce `DYLD_LIBRARY_PATH` / `PATH` / `LD_LIBRARY_PATH`.<br><br>
* **Mikrofon (macOS)** – `Settings → Privacy & Security → Microphone` (zezwól terminalowi/IDE).<br><br>
* **Wysoka latencja** – sprawdź `WATUS_BLOCKSIZE=320`, `SIL_MS_END≈600–700`, `WHISPER_COMPUTE_TYPE` (`int8` na CPU, `float16` na GPU).
# Uwaga o plikach w repo
Modele (Piper/Whisper/ECAPA) i binarki **nie są dołączone** – pobierz je linkami powyżej i ustaw ścieżki w `.env`. <br>
Logi (`dialog.jsonl`, `meldunki.jsonl`) możesz trzymać lokalnie; `.gitignore` domyślnie je wyklucza.