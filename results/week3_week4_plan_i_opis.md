# Week 3-4: Szczegolowy opis realizacji

Ten dokument domyka tygodnie 3 i 4 zgodnie z planem projektu:

- **Tydzien 3:** odtworzenie benchmarku Direct Method (DM),
- **Tydzien 4:** diagnoza slabosci DM (interpretowalnosc + out-of-distribution).

Pliki notebookow:

- `notebooks/03_propensity_ips.ipynb` (zawartosc: Week 3),
- `notebooks/04_doubly_robust.ipynb` (zawartosc: Week 4).

Wykresy:

- `figures/week3/week3_dm_vs_naive_ci.png`,
- `figures/week4/week4_shap_top20.png`,
- `figures/week4/week4_ood_distance_hist.png`,
- `figures/week4/week4_dm_risk_map.png`.

## Week 3 - Reprodukcja benchmarku DM

### Cel eksperymentu

W tygodniu 3 celem nie jest jeszcze IPS/DR, tylko mocne domkniecie DM:

1. uzyskac powtarzalny pipeline z kontrola losowosci,
2. policzyc wartosc polityki (`V_DM`) dla dwoch polityk referencyjnych,
3. oszacowac niepewnosc przez bootstrap confidence intervals,
4. przygotowac wykres gotowy do raportu/projektu badawczego.

### Co zostalo zaimplementowane

1. **Ladowanie Open Bandit Dataset**
   - dwie polityki historyczne: `random` i `bts`,
   - kampania: `all`.

2. **Powtarzalny split train/test**
   - `train_test_split(..., test_size=0.2, random_state=42, stratify=y)`,
   - model reward uczony na danych `random` (zmniejszenie selection bias).

3. **Model nagrody (`Q-hat`)**
   - `XGBClassifier`,
   - wejscie: `context + one-hot(action)`,
   - obsluga niezbalansowania przez `scale_pos_weight`.

4. **Predykcja oczekiwanej nagrody dla wszystkich akcji**
   - budowa macierzy `expected_reward` o ksztalcie `(n_rounds, n_actions)`,
   - konwersja do 3D dla OBP: `(n_rounds, n_actions, len_list)`.

5. **Estymacja `V_DM`**
   - polityka BTS-like (one-hot na zalogowanej akcji na zalogowanej pozycji),
   - polityka random baseline (rozkład jednostajny po akcjach),
   - obie estymowane przez `DirectMethod`.

6. **Przedzialy ufnosci**
   - bootstrap w `OffPolicyEvaluation.estimate_intervals`,
   - `n_bootstrap_samples = 200` (zgodnie z planem tyg. 3).

7. **Wykres raportowy**
   - porownanie `Naive CTR` vs `V_DM` dla dwoch polityk,
   - slupki + 95% CI dla `V_DM`,
   - zapis do `figures/week3/week3_dm_vs_naive_ci.png`.

### Jak interpretowac wynik Week 3

- Jesli `V_DM(bts_logged_style)` > `V_DM(random_baseline)`, model sugeruje przewage polityki BTS.
- Roznica miedzy `Naive CTR` i `V_DM` pokazuje efekt korekty przez model nagrody.
- Szerokosc CI mowi, na ile stabilna jest estymacja:
  - waskie CI -> stabilniejszy sygnal,
  - szerokie CI -> duza niepewnosc, ryzyko niestabilnosci.

### Ograniczenia Week 3

- To jest reprodukcja DM w wersji projektowej, nie jeszcze idealna reprodukcja 1:1 tabel benchmarkowych z papera.
- Dla porownania publikacyjnego nalezy:
  - odtworzyc identyczny protokol splitow i metryk jak w OBP benchmark,
  - uruchomic ewaluacje na tych samych konfiguracjach co w publikacji.

## Week 4 - Diagnoza slabosci DM

### Cel eksperymentu

Tydzien 4 odpowiada na pytanie:

**Kiedy DM jest wiarygodny, a kiedy moze sie mylic przez ekstrapolacje poza obszar danych treningowych?**

### Co zostalo zaimplementowane

1. **SHAP dla modelu nagrody**
   - `shap.TreeExplainer(model)`,
   - ranking cech po `mean(|SHAP|)`,
   - wykres `Top-20` cech:
     - `figures/week4/week4_shap_top20.png`.

2. **OOD proxy przez nearest-neighbor distance**
   - przestrzen porownania: tylko cechy kontekstowe,
   - treningowy manifold: probka kontekstow z `random`,
   - punkty oceniane: probka kontekstow z `bts`,
   - metryka: odleglosc do najblizszego punktu treningowego,
   - prog OOD: 95 percentyl odleglosci,
   - histogram odleglosci:
     - `figures/week4/week4_ood_distance_hist.png`.

3. **Risk map DM**
   - os X: odleglosc OOD,
   - os Y: przewidywana nagroda dla akcji zalogowanej,
   - mapa pokazuje, gdzie model daje wysokie predykcje mimo slabego pokrycia danych,
   - wykres:
     - `figures/week4/week4_dm_risk_map.png`.

### Jak interpretowac wynik Week 4

- **Wysoka predykcja + duza odleglosc OOD** -> sygnal ryzyka ekstrapolacji (mozliwy overconfidence modelu).
- **Niska odleglosc OOD + stabilne predykcje** -> obszar relatywnie bezpieczny dla DM.
- SHAP pozwala odroznic:
  - cechy faktycznie informatywne,
  - cechy potencjalnie artefaktowe/niestabilne.

## Powiazanie z planem semestralnym

Week 3-4 sa zamkniete na poziomie implementacji notebookowej:

- [x] Week 3: split + DM + CI + wykres porownawczy,
- [x] Week 4: SHAP + OOD + opis ograniczen DM.

Naturalny kolejny krok (Week 5):

1. model propensity score (`P(a|s)`),
2. kalibracja i diagnostyka overlap,
3. przejscie do IPS/SNIPS.

## Rekomendacje porzadkujace (przed Week 5)

1. Przeniesc funkcje pomocnicze z notebookow do `src/`:
   - estymacja expected rewards,
   - budowa action distributions,
   - helpery ewaluacyjne.
2. Dodac prosty skrypt uruchomieniowy (np. `scripts/run_week3.py`) dla replikowalnosci.
3. Ujednolicic nazwy notebookow (opcjonalnie) do:
   - `03_week3_dm_benchmark.ipynb`,
   - `04_week4_dm_diagnostics.ipynb`.
