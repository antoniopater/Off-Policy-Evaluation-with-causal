# Off-Policy Evaluation z korekcją kauzalną

> Ocena polityk decyzyjnych na danych obserwacyjnych — bez możliwości ich uruchomienia na żywo.

## Streszczenie

Projekt buduje i porównuje trzy estymatory OPE (Direct Method → IPS → Doubly Robust), które pozwalają ocenić czy inna polityka decyzyjna byłaby lepsza — z korekcją biasu wynikającego z tego, że dane historyczne nie były zbierane losowo.

Pipeline wchodzi bezpośrednio jako sekcja **Experiments** do projektu badawczego *"Causally Adjusted Valuation of Progressive Passes in Football"* — cel NeurIPS 2026 Workshop w Sydney.

---

## Problem

W danych obserwacyjnych decyzje nie są losowe. Bank odrzuca pewnych klientów z góry, piłkarz gra progresywnie tylko gdy sytuacja na to pozwala. Naiwny model ML uczy się historycznej polityki zamiast identyfikować optymalne decyzje — **myli jakość decyzji z jakością kontekstu, w którym była podjęta**. To jest bias selekcji.

Off-Policy Evaluation odpowiada na pytanie: *gdybyśmy przez ostatni rok stosowali politykę X zamiast Y — ile byśmy zyskali?* I robi to uczciwie, bez wdrażania nowej polityki na żywo.

---

## Metody

### Direct Method (DM)
Uczysz model ML, który przewiduje nagrodę po danej akcji. Pytasz go co by przewidział dla alternatywnej decyzji.

```
V_DM = E[ f(s, a_new) ]
```

- **Zaleta:** niska wariancja
- **Wada:** model może się mylić poza danymi treningowymi (extrapolation bias)

---

### Inverse Propensity Scoring (IPS)
Ważysz obserwacje odwrotnością prawdopodobieństwa, że stara polityka podjęła daną decyzję. Obserwacje zaskakujące dla starej polityki dostają dużą wagę.

```
V_IPS = E[ r × π_new(a|s) / π_old(a|s) ]
```

- **Zaleta:** nieobciążony przy dobrym modelu propensity score
- **Wada:** duża wariancja gdy polityki bardzo się różnią

---

### Doubly Robust (DR)
Łączy DM i IPS. Kluczowa własność: dobry wynik gdy **choć jedna** z metod działa poprawnie.

```
V_DR = V_IPS + DM_correction
```

- **Zaleta:** state of the art w OPE — obie metody muszą być złe jednocześnie, żeby wynik był zły
- **Wada:** wymaga wytrenowania dwóch modeli (reward model + propensity model)

---

## Dane

### Open Bandit Dataset (OBD) — główne dane
- Zozo Research (japońska firma e-commerce), paper NeurIPS 2021 (Saito et al.)
- 1,3 miliona prawdziwych obserwacji z systemu rekomendacji mody
- Dwie historyczne polityki: **Random** i **Bernoulli Thompson Sampling** — obie znane, dostępny ground truth
- Gotowa biblioteka `obp` z implementacjami DM, IPS, DR
- Paper benchmarkowy określa dokładnie jakich wyników się spodziewać

### StatsBomb Open Data — rozszerzenie (pilot sportowy)
- Prawdziwe dane meczowe z pozycjami zawodników (format 360°)
- **Treatment:** progressive pass (podanie >30 yardów do przodu)
- Sprawdzenie czy pipeline z OBD działa w domenie sportowej
- Bezpośredni pomost do projektu badawczego z doktorem

---

## Stos technologiczny

| Obszar | Biblioteki |
|---|---|
| OPE i kauzalne | `obp`, `econml` (Microsoft), `dowhy` |
| Modele ML | `xgboost`, `scikit-learn` |
| Interpretacja | `shap` |
| Dane sportowe | `statsbombpy` |
| Offline RL (opcjonalnie) | `d3rlpy`, `scope-rl` |

---

## Plan — 12 tygodni

### Faza 1 — Fundament (tygodnie 1–4) · gwarancja 3.0

**Tydzień 1 — Setup środowiska i dane OBD**
- [ ] `pip install obp xgboost shap`
- [ ] Pobranie Open Bandit Dataset — dwie polityki: Random i BernoulliTS
- [ ] EDA: rozkład nagród, rozkład akcji, brakujące wartości

**Tydzień 2 — Implementacja Direct Method**
- [ ] Lektura sekcji 3 papieru Saito et al. (NeurIPS 2021)
- [ ] Trening reward model (XGBoost)
- [ ] `DirectMethod` z biblioteki `obp` — pierwsza wartość policy value

**Tydzień 3 — Odtworzenie wyników z papieru**
- [ ] Identyczny split train/test jak w papierze
- [ ] Porównanie MSE z benchmarkiem z Table 2 / Figure 3
- [ ] Bootstrapowane przedziały ufności (n=200 resampli)

**Tydzień 4 — Analiza słabości DM**
- [ ] SHAP: które cechy są ważne dla reward model?
- [ ] Identyfikacja przykładów out-of-distribution
- [ ] Dokumentacja: założenia DM, kiedy zawodzi

> **Deliverable fazy 1:** działający pipeline DM + wyniki zgodne z benchmarkiem + README

---

### Faza 2 — Korekcja kauzalna (tygodnie 5–8) · cel 4.0

**Tydzień 5 — Propensity score model**
- [ ] Trening modelu PS: XGBoost multiclass — P(a | s)
- [ ] Kalibracja: reliability diagram
- [ ] Overlap assumption: histogram wag IPS = π_new / π_old

**Tydzień 6 — IPS Estimator z clippingiem**
- [ ] `InverseProbabilityWeighting` z `obp` + stabilizowane wagi (SNIPS)
- [ ] Eksperyment clipping: λ ∈ {1, 5, 10, 50} → bias vs wariancja
- [ ] Odtworzenie wyników IPS z papieru

**Tydzień 7 — Diagnoza kiedy IPS eksploduje**
- [ ] Symulacja naruszenia overlap — usunięcie obserwacji z P(a|s) < 0.05
- [ ] Effective Sample Size: ESS = (Σwi)² / Σwi²
- [ ] Instalacja i przegląd `econml` — dokumentacja DR

**Tydzień 8 — Porównanie DM vs IPS**
- [ ] Unified benchmark: DM vs IPS vs SNIPS
- [ ] Dekompozycja błędu: MSE = Bias² + Variance
- [ ] Wykres Bias-Variance trade-off

> **Deliverable fazy 2:** pełne porównanie DM vs IPS + dekompozycja błędu

---

### Faza 3 — Most do badań (tygodnie 9–12) · cel 5.0

**Tydzień 9 — Doubly Robust Estimator**
- [ ] `DoublyRobust` z `obp`
- [ ] Eksperyment: zły PS model → sprawdzenie czy DM ratuje DR
- [ ] Eksperyment: zły reward model → sprawdzenie czy IPS ratuje DR

**Tydzień 10 — Sensitivity analysis w DoWhy**
- [ ] Definicja modelu kauzalnego: treatment, outcome, confounders
- [ ] Random Common Cause test
- [ ] Placebo Treatment test
- [ ] Interpretacja: odporność estymatu DR na ukryte zmienne

**Tydzień 11 — Pilot na StatsBomb**
- [ ] `statsbombpy`: dane La Liga 2015/16 z formatem 360°
- [ ] Definicja treatment: progressive pass (>30 yardów do przodu)
- [ ] Features z danych 360°: pozycja, presja, zawodnicy w zasięgu
- [ ] DR pipeline na danych piłkarskich — pierwsze policy value dla progressive passes

**Tydzień 12 — Raport finalny**
- [ ] Tabela finalna: DM vs IPS vs DR — MSE, Bias², Variance, CI
- [ ] Sekcja "Experiments" w formacie NeurIPS
- [ ] Limitations: założenia i kiedy każda metoda zawodzi
- [ ] Czysty, reprodukowalny kod + README + tag `v1.0`

> **Deliverable fazy 3:** pełny pipeline OPE + raport + sekcja Experiments gotowa do projektu badawczego

---

## Struktura projektu

```
.
├── data/                   # dane OBD i StatsBomb (nie commitowane)
├── notebooks/
│   ├── 01_eda.ipynb         # eksploracja danych OBD
│   ├── 02_direct_method.ipynb
│   ├── 03_propensity_ips.ipynb
│   ├── 04_doubly_robust.ipynb
│   ├── 05_sensitivity.ipynb
│   └── 06_statsbomb_pilot.ipynb
├── src/
│   ├── estimators.py        # DM, IPS, DR wrappers
│   ├── propensity.py        # model propensity score
│   └── evaluation.py       # MSE, bias/variance, bootstrap CI
├── results/                 # wykresy, tabele
├── requirements.txt
└── README.md
```

---

## Instalacja

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt
```

**requirements.txt:**
```
obp
econml
dowhy
xgboost
shap
statsbombpy
scikit-learn
pandas
numpy
matplotlib
seaborn
```

---

## Literatura

- Saito, Y. et al. (2021). *Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.* NeurIPS 2021. [paper](https://arxiv.org/abs/2008.07146)
- Dudík, M., Langford, J., Li, L. (2011). *Doubly Robust Policy Evaluation and Learning.* ICML 2011.
- Athey, S., Imbens, G. (2016). *Recursive partitioning for heterogeneous causal effects.* PNAS.

---

## Status

| Faza | Status | Ocena |
|---|---|---|
| Faza 1 — Direct Method | 🔲 w toku |
| Faza 2 — IPS + propensity | 🔲 planowana | 
| Faza 3 — DR + StatsBomb | 🔲 planowana | 
