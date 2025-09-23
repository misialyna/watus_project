#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jednorazowy wybór scenariusza na start dnia.
Zapisuje do: ./scenarios_text/active.jsonl  (append – ostatnia linia = aktywny)
"""

import json, time, sys
from pathlib import Path

SCEN_DIR = Path("./scenarios_text")
ACTIVE_PATH = SCEN_DIR / "active.jsonl"

def list_ids() -> list[str]:
    return sorted(p.stem for p in SCEN_DIR.glob("*.md"))

def set_active(sid: str) -> None:
    SCEN_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": int(time.time()*1000), "id": sid}
    with ACTIVE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[set_scenario] Aktywny scenariusz → {sid} (dopisana linia w {ACTIVE_PATH})")

def main():
    ids = list_ids()
    if not ids:
        print("Brak plików scenariuszy w ./scenarios_text/*.md")
        sys.exit(1)
    if len(sys.argv) >= 2:
        sid = sys.argv[1]
        if sid not in ids:
            print("Nieznany scenariusz. Dostępne:", ", ".join(ids))
            sys.exit(2)
        set_active(sid)
        return
    print("Dostępne scenariusze:")
    for i, s in enumerate(ids, 1):
        print(f"  {i}. {s}")
    try:
        idx = int(input("Wybierz numer scenariusza: ").strip())
    except Exception:
        print("Niepoprawny numer.")
        sys.exit(3)
    if not (1 <= idx <= len(ids)):
        print("Poza zakresem.")
        sys.exit(4)
    set_active(ids[idx-1])

if __name__ == "__main__":
    main()
