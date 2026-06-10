# Off-Policy Evaluation z korekcją kauzalną

> Ocena polityk decyzyjnych na danych obserwacyjnych — bez możliwości ich uruchomienia na żywo.

## Streszczenie

Projekt buduje i porównuje trzy estymatory OPE (Direct Method → IPS → Doubly Robust), które pozwalają ocenić czy inna polityka decyzyjna byłaby lepsza — z korekcją biasu wynikającego z tego, że dane historyczne nie były zbierane losowo.

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
- [x] `pip install obp xgboost shap`
- [x] Pobranie Open Bandit Dataset — dwie polityki: Random i BernoulliTS
- [x] EDA: rozkład nagród, rozkład akcji, brakujące wartości

**Tydzień 2 — Implementacja Direct Method**
- [x] Lektura sekcji 3 papieru Saito et al. (NeurIPS 2021)
- [x] Trening reward model (XGBoost)
- [x] `DirectMethod` z biblioteki `obp` — pierwsza wartość policy value

**Tydzień 3 — Odtworzenie wyników z papieru**
- [x] Identyczny split train/test jak w papierze
- [x] Porównanie MSE z benchmarkiem z Table 2 / Figure 3
- [x] Bootstrapowane przedziały ufności (n=200 resampli)

**Tydzień 4 — Analiza słabości DM**
- [x] SHAP: które cechy są ważne dla reward model?
- [x] Identyfikacja przykładów out-of-distribution
- [x] Dokumentacja: założenia DM, kiedy zawodzi

> **Deliverable fazy 1:** działający pipeline DM + wyniki zgodne z benchmarkiem + README

---

### Faza 2 — Korekcja kauzalna (tygodnie 5–8) · cel 4.0

**Tydzień 5 — Propensity score model**
- [x] Trening modelu PS: XGBoost multiclass — P(a | s)
- [x] Kalibracja: reliability diagram
- [x] Overlap assumption: histogram wag IPS = π_new / π_old

**Tydzień 6 — IPS Estimator z clippingiem**
- [x] `InverseProbabilityWeighting` z `obp` + stabilizowane wagi (SNIPS)
- [x] Eksperyment clipping: λ ∈ {1, 5, 10, 50} → bias vs wariancja
- [x] Odtworzenie wyników IPS z papieru

**Tydzień 7 — Diagnoza kiedy IPS eksploduje**
- [x] Symulacja naruszenia overlap — usunięcie obserwacji z P(a|s) < 0.05
- [x] Effective Sample Size: ESS = (Σwi)² / Σwi²
- [x] Instalacja i przegląd `econml` — dokumentacja DR

**Tydzień 8 — Porównanie DM vs IPS**
- [x] Unified benchmark: DM vs IPS vs SNIPS
- [x] Dekompozycja błędu: MSE = Bias² + Variance
- [x] Wykres Bias-Variance trade-off

> **Deliverable fazy 2:** pełne porównanie DM vs IPS + dekompozycja błędu

---

### Faza 3 — Most do badań (tygodnie 9–12) · cel 5.0

**Tydzień 9 — Doubly Robust Estimator**
- [x] `DoublyRobust` z `obp`
- [x] Eksperyment: zły PS model → sprawdzenie czy DM ratuje DR
- [x] Eksperyment: zły reward model → sprawdzenie czy IPS ratuje DR

**Tydzień 10 — Sensitivity analysis w DoWhy**
- [x] Definicja modelu kauzalnego: treatment, outcome, confounders
- [x] Random Common Cause test
- [x] Placebo Treatment test
- [x] Interpretacja: odporność estymatu DR na ukryte zmienne

**Tydzień 11 — Pilot na StatsBomb**
- [x] `statsbombpy`: dane La Liga 2015/16 z formatem 360°
- [x] Definicja treatment: progressive pass (>30 yardów do przodu)
- [x] Features z danych 360°: pozycja, presja, zawodnicy w zasięgu
- [x] DR pipeline na danych piłkarskich — pierwsze policy value dla progressive passes

**Tydzień 12 — Raport finalny**
- [x] Tabela finalna: DM vs IPS vs DR — MSE, Bias², Variance, CI
- [x] Sekcja "Experiments" w formacie NeurIPS
- [x] Limitations: założenia i kiedy każda metoda zawodzi
- [x] Czysty, reprodukowalny kod + README + tag `v1.0`

> **Deliverable fazy 3:** pełny pipeline OPE + raport + sekcja Experiments gotowa do projektu badawczego

---

## Struktura projektu

```
.
├── data/                        # dane OBD i StatsBomb (nie commitowane)
├── notebooks/
│   ├── 01_eda.ipynb              # T1 — eksploracja danych OBD
│   ├── 02_direct_method.ipynb   # T2 — Direct Method, reward model XGBoost
│   ├── 03_propensity_ips.ipynb  # T3 — reprodukcja benchmarku DM
│   ├── 04_doubly_robust.ipynb   # T4 — SHAP + OOD diagnostika
│   ├── 07_propensity_scores.ipynb  # T5 — propensity score model P(a|s)
│   ├── 08_ips_snips.ipynb       # T6 — IPS + SNIPS + clipping experiment
│   ├── 09_overlap_ess.ipynb     # T7 — overlap violation + ESS
│   ├── 10_bias_variance.ipynb   # T8 — unified benchmark + MSE decomposition
│   ├── 11_doubly_robust.ipynb   # T9 — Doubly Robust + robustness experiments
│   ├── 12_sensitivity_dowhy.ipynb  # T10 — sensitivity analysis (DoWhy)
│   └── 13_statsbomb_pilot.ipynb # T11 — StatsBomb La Liga OPE pilot
├── figures/                     # wykresy (week2–week11)
├── src/
│   ├── estimators.py            # DM, IPS, DR wrappers
│   ├── propensity.py            # model propensity score
│   ├── evaluation.py            # MSE, bias/variance, bootstrap CI
│   └── dataset.py               # patch OpenBanditDataset dla pandas 2.x
├── results/                     # tabele CSV, FINAL_REPORT.md
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Instalacja

### Opcja 1: uv (rekomendowane)

```bash
git clone <repo-url>
cd <repo>
uv sync
```

Uruchamianie notebookow:

```bash
uv run jupyter lab
```

### Opcja 2: pip

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
| Faza 1 — Direct Method (T1–T4) | ✅ ukończona | 3.0+ |
| Faza 2 — IPS + propensity (T5–T8) | ✅ ukończona | 4.0+ |
| Faza 3 — DR + StatsBomb (T9–T12) | ✅ ukończona | 5.0 |

Raport finalny: [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md)
