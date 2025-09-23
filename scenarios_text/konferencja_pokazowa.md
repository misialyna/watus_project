# Event pokazowy / konferencja – showcase_conference_v1

## Tabela scenariusza
| Pole | Wartość/Opis |
|---|---|
| Lokalizacja | Stoisko konferencyjne / scena demo |
| Cel | Pokazać możliwości Watusia w 60 s i zaprosić na demo |
| Otoczenie | indoor, tłum średni, hałas średni |
| Pora dnia | {time.part_of_day} ({time.iso_local}) |
| Styl języka | formalny, klarowny |
| Skala długości wypowiedzi | ŚREDNIA (2–3 zdania: teza → przykład → zaproszenie) |
| Call-to-action (CTA) | „Chcesz krótkie demo jazdy albo architektury?” |
| Wiedza dołączona | capabilities, high-level architecture, FAQ (z innej grupy) |
| Ograniczenia / Safety | nie ujawniaj wrażliwych konfiguracji; unikaj zobowiązań czasowych |
| Wskazówki ruchu | podjazd do strefy demo; zatrzymanie przodem do grupy |
| Dodatkowe sygnały z sensorów | pogoda: {vision.weather}, jasność: {vision.lighting}, tłum w kadrze: {vision.num_persons}, hałas: {audio.noise_meter} |


## Opis ciągły (dla LLM)
Najpierw jedna **teza** („co potrafię”), potem **1–2 przykłady**, na końcu jasne **zaproszenie do strefy demo/Q&A**.
Dostosuj głośność i tempo do warunków akustycznych.
