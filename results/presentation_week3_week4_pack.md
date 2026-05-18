# Pakiet do rozszerzenia prezentacji (Week 3-4)

Ten plik jest przygotowany tak, aby mozna go bezposrednio wrzucic do Claude i wygenerowac rozszerzenie prezentacji.

## 1) Kontekst i cel rozszerzenia

### Co bylo wczesniej

- Week 1: EDA na OBD.
- Week 2: pierwsza implementacja Direct Method (DM).

### Co dokladamy teraz (Week 3-4)

- **Week 3:** reprodukcja benchmarku DM z przedzialami ufnosci.
- **Week 4:** diagnostyka wiarygodnosci DM (importance + OOD + risk map).

### Dlaczego to wazne

- Pokazujemy nie tylko wynik liczbowy, ale tez **jak bardzo mu ufamy**.
- Domykamy logike przejscia do kolejnych metod (IPS/DR): najpierw DM, potem jego ograniczenia.

## 2) Gotowy plan nowych slajdow (4-5 slajdow)

Ponizej jest dokladny szablon slajdow do wklejenia.

---

## Slajd A - Week 3: Benchmark DM (wynik glowny)

### Tytul slajdu

**Week 3 - Reprodukcja benchmarku Direct Method (DM)**

### Co ma byc na slajdzie

- Wstaw wykres: `figures/week3/week3_dm_vs_naive_ci.png`
- Obok wykresu tabela (2 wiersze):
  - random_baseline: `V_DM = 0.074759`, 95% CI `[0.073888, 0.075853]`
  - bts_logged_style: `V_DM = 0.082847`, 95% CI `[0.080344, 0.085181]`
- Pod spodem krótka notka: "Bootstrap CI: n=200"

### Tekst na slajdzie (gotowy)

- DM estymuje wyzsza wartosc dla polityki BTS niz dla random baseline.
- Przedzialy ufnosci sa rozdzielone, co wspiera stabilnosc roznicy.
- To zamyka milestone Week 3: wynik + niepewnosc estymacji.

### Narracja mowiona (30-45s)

"W tygodniu 3 zrobilismy reprodukcje benchmarku Direct Method na Open Bandit Dataset. Najwazniejszy wynik jest taki, ze oszacowana wartosc polityki BTS jest wyzsza niz dla polityki random. Co istotne, nie pokazujemy tylko jednej liczby - dodalismy bootstrapowe przedzialy ufnosci, dzieki czemu wynik jest bardziej wiarygodny i gotowy do raportowania."

---

## Slajd B - Week 3: Jak to policzylismy (reproducibility)

### Tytul slajdu

**Week 3 - Pipeline obliczen i replikowalnosc**

### Co ma byc na slajdzie

- Schemat 4 krokow:
  1. train/test split (`random_state=42`)
  2. trening `Q-hat` (XGBoost) na danych random
  3. estymacja `V_DM` dla 2 polityk
  4. bootstrap CI (`n=200`)
- Dopisek: notebook `notebooks/03_propensity_ips.ipynb`

### Tekst na slajdzie (gotowy)

- Trening reward modelu na random zmniejsza ryzyko dziedziczenia biasu selekcji.
- Ten sam pipeline uruchamiamy w pelni automatycznie (notebook wykonuje sie end-to-end).
- Wynik jest reprodukowalny i gotowy jako baseline przed IPS/DR.

### Narracja mowiona (30-45s)

"Kluczowe bylo zapewnienie replikowalnosci. Uzywamy stalego splitu i stalego ziarna losowosci. Reward model uczymy na polityce random, bo daje lepsze pokrycie przestrzeni akcji i kontekstu niz BTS. Potem liczymy wartosc DM dla obu polityk i dodajemy bootstrap, zeby oszacowac niepewnosc."

---

## Slajd C - Week 4: Interpretowalnosc modelu (importance)

### Tytul slajdu

**Week 4 - Co napedza predykcje DM?**

### Co ma byc na slajdzie

- Wstaw wykres: `figures/week4/week4_shap_top20.png`
- Dopisek techniczny:
  - "W tym srodowisku SHAP byl niekompatybilny z numpy"
  - "Automatyczny fallback: permutation importance"

### Tekst na slajdzie (gotowy)

- Najbardziej informatywne cechy to m.in. `ctx_9`, `act_18`, `act_49`.
- Ranking cech pozwala sprawdzic, czy model opiera sie na sensownym sygnale.
- To pierwszy krok diagnostyki: "czy model patrzy tam, gdzie powinien?".

### Narracja mowiona (30-45s)

"W tygodniu 4 zaczynamy od interpretowalnosci. Sprawdzamy, ktore cechy najmocniej wplywaja na predykcje modelu nagrody. W naszym srodowisku SHAP mial konflikt wersji z numpy, wiec automatycznie przeszlismy na permutation importance. Merytorycznie cel pozostaje ten sam: zweryfikowac, czy model bazuje na stabilnych i sensownych sygnalach."

---

## Slajd D - Week 4: OOD i ryzyko ekstrapolacji

### Tytul slajdu

**Week 4 - Gdzie DM moze sie mylic? (OOD + risk map)**

### Co ma byc na slajdzie

- Po lewej: `figures/week4/week4_ood_distance_hist.png`
- Po prawej: `figures/week4/week4_dm_risk_map.png`
- Podpis metryki: `OOD rate = 0.0016` (prog: 95 percentyl odleglosci NN)

### Tekst na slajdzie (gotowy)

- OOD mierzymy odlegloscia BTS-context od manifoldu treningowego (random-context).
- Punkty o duzej odleglosci OOD to obszary podwyzszonego ryzyka dla DM.
- Polaczenie "wysoka predykcja + duzy OOD" to sygnal mozliwej nadmiernej pewnosci modelu.

### Narracja mowiona (40-60s)

"Sama interpretowalnosc nie wystarczy, dlatego dodalismy diagnostyke OOD. Dla kazdego punktu z BTS mierzymy odleglosc do najblizszego punktu treningowego z random. To daje przyblizenie pytania: czy model predykuje blisko danych, ktore juz widzial, czy daleko poza nimi. Na risk mapie widzimy, gdzie model jest pewny, ale jednoczesnie daleko od danych treningowych - i to sa miejsca najwiekszego ryzyka ekstrapolacji."

---

## Slajd E - Wnioski i przejscie do kolejnego etapu

### Tytul slajdu

**Wnioski po Week 3-4 i krok nastepny**

### Co ma byc na slajdzie

- 3 sekcje:
  - "Co potwierdzilismy"
  - "Jakie sa ograniczenia DM"
  - "Co robimy w Week 5"

### Tekst na slajdzie (gotowy)

**Co potwierdzilismy:**
- DM daje spojny sygnal przewagi BTS nad random.
- Wynik ma oszacowana niepewnosc (bootstrap CI).

**Ograniczenia DM:**
- DM moze byc wrazliwy na ekstrapolacje poza obszar pokrycia danych.
- Potrzebna korekcja przez model propensity i metody wagowe.

**Kolejny krok (Week 5):**
- Trening `P(a|s)` (propensity model),
- kalibracja i overlap diagnostics,
- przejscie do IPS/SNIPS.

### Narracja mowiona (30-45s)

"Week 3-4 domknely nam baseline i diagnostyke Direct Method. Mamy wynik, mamy niepewnosc i mamy jasny obraz ograniczen. To idealny moment, aby wejsc w Week 5: propensity score i IPS, czyli etap, w ktorym zaczynamy aktywnie korygowac bias selekcji."

## 3) Material pomocniczy do sekcji 'Backup/Appendix' (opcjonalnie)

Mozna dodac 1 dodatkowy slajd zapasowy z ponizszymi punktami:

- Dataset:
  - random context shape: `(10000, 20)`
  - bts context shape: `(10000, 22)`
  - naive CTR random: `0.0038`
  - naive CTR bts: `0.0042`
- DM estimates:
  - random baseline: `0.074759` (CI: `0.073888-0.075853`)
  - bts logged style: `0.082847` (CI: `0.080344-0.085181`)
- OOD:
  - rate above 95th percentile: `0.0016`

## 4) Instrukcja dla Claude (gotowy prompt)

Skopiuj ponizszy prompt do Claude:

---

Na podstawie ponizszych informacji przygotuj rozszerzenie prezentacji (styl akademicki, czytelny, po polsku) o sekcje Week 3-4.

Wymagania:
1) Stworz 5 nowych slajdow:
- Week 3 benchmark DM wynik
- Week 3 pipeline i reprodukowalnosc
- Week 4 feature importance
- Week 4 OOD + risk map
- Wnioski i przejscie do Week 5
2) Dla kazdego slajdu podaj:
- tytul
- tresc punktowa (max 5 punktow)
- notatki prelegenta (40-80 slow)
3) Uzyj liczb:
- V_DM random_baseline = 0.074759 (95% CI: 0.073888-0.075853)
- V_DM bts_logged_style = 0.082847 (95% CI: 0.080344-0.085181)
- naive CTR random = 0.0038
- naive CTR bts = 0.0042
- OOD rate = 0.0016
4) Wskaz gdzie wstawic wykresy:
- figures/week3/week3_dm_vs_naive_ci.png
- figures/week4/week4_shap_top20.png
- figures/week4/week4_ood_distance_hist.png
- figures/week4/week4_dm_risk_map.png
5) Na koncu dodaj 1 slajd appendix z metodyka i ograniczeniami (fallback SHAP -> permutation importance).

Kontekst merytoryczny:
- Week 3: reprodukcja benchmarku Direct Method, bootstrap CI.
- Week 4: diagnostyka wiarygodnosci DM przez feature importance i OOD.
- Cel: przygotowanie przejscia do Week 5 (propensity + IPS/SNIPS).

---

## 5) Gdzie sa pliki

- Material opisowy: `results/presentation_week3_week4_pack.md`
- Opis techniczny Week 3-4: `results/week3_week4_plan_i_opis.md`
- Notebooki:
  - `notebooks/03_propensity_ips.ipynb`
  - `notebooks/04_doubly_robust.ipynb`
- Wykresy:
  - `figures/week3/week3_dm_vs_naive_ci.png`
  - `figures/week4/week4_shap_top20.png`
  - `figures/week4/week4_ood_distance_hist.png`
  - `figures/week4/week4_dm_risk_map.png`
