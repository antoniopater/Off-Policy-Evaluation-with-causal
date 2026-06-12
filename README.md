# Off-Policy Evaluation z korekcją kauzalną

> Ocena polityk decyzyjnych na danych obserwacyjnych — bez możliwości ich uruchomienia na żywo.

## Streszczenie

Projekt buduje i porównuje trzy estymatory OPE (Direct Method → IPS → Doubly Robust), które pozwalają ocenić czy inna polityka decyzyjna byłaby lepsza — z korekcją biasu wynikającego z tego, że dane historyczne nie były zbierane losowo.

## Wyniki projektu

Krótkie podsumowanie tego, co wyszło z notebooków (szczegóły: [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md)):

**Open Bandit Dataset (OBD)** — reprodukcja pipeline’u z papieru Saito et al., estymatory z biblioteki `obp` + XGBoost (reward i propensity):

| Estymator | V̂ | vs V* ≈ 0.0038 | Główny wniosek |
|---|---|---|---|
| DM | 0.003515 | −0.000285 | najniższe MSE (dominuje niska wariancja) |
| IPS / SNIPS | 0.0044 / 0.0042 | bliżej V* | wyższa wariancja, mniejszy bias |
| DR | 0.004223 | balans | łączy oba podejścia |

- **Double robustness** (nb. 11): przy losowym PS modelu DR ≈ V*; przy zepsutym reward modelu (stała 0.5) DR zawodzi — IPS ratuje wynik.
- **Diagnoza DM** (nb. 04): reward model słaby (AUC-PR ≈ 0.015), SHAP + OOD; ekstrapolacja poza danymi treningowymi.
- **IPS / overlap** (nb. 08–09): na OBD wagi ≈ 1 (logging ≈ eval); symulacja naruszenia overlap obniża ESS z 0.99 do 0.11 i eksploduje wariancję IPS.
- **DoWhy** (nb. 12): ATE akcji 0 vs reszta = −0.0039; testy refutacji (Random Common Cause, Placebo, Subset) stabilne.

**StatsBomb La Liga 2015/16** (nb. 13) — pilot sportowy: treatment = progressive pass, eval policy = 50% progressive; DR = **1.79%** vs naive CTR **1.93%**, ESS = 0.686, szerokie CI (20 meczów).

**Walidacja syntetyczna** (nb. 14) — symulacja ze znanym V*: przy ESS ≈ 0.7 (jak StatsBomb) estymatory wykrywają duży efekt, ale nie rozróżniają Δ = 0 od małego Δ — wynik DR ≈ naive na piłce jest spójny z metodologią, nie świadczy o awarii estymatora.

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
Łączy DM i IPS. Poprawny wynik, gdy choć jeden z modeli (nagrody lub propensity) jest wystarczająco dobry.

```
V_DR = V_IPS + DM_correction
```

- **Zaleta:** odporność na błąd jednego z modeli
- **Wada:** wymaga dwóch modeli (reward + propensity)

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

### Faza 1 — Fundament (tygodnie 1–4)

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

---

### Faza 2 — Korekcja kauzalna (tygodnie 5–8)

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

---

### Faza 3 — Most do badań (tygodnie 9–12)

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
│   ├── 13_statsbomb_pilot.ipynb    # T11 — StatsBomb La Liga OPE pilot
│   └── 14_synthetic_validation.ipynb  # T12 — walidacja syntetyczna
├── figures/                     # wykresy (week2–week12)
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

Projekt opiera się na trzech głównych źródłach teoretycznych oraz bibliotekach implementujących te metody:

| Źródło | Rola w projekcie |
|---|---|
| **Saito et al. (2021)** | Główny benchmark: Open Bandit Dataset, split train/test, definicje DM/IPS/DR, biblioteka `obp`. Notebooki 01–03, 07–11 odtwarzają pipeline i porównują MSE z Table 2 / Figure 3 z papieru. |
| **Dudík et al. (2011)** | Teoria estymatora Doubly Robust i własność double robustness. Notebook 11: eksperymenty ze złym PS modelem vs złym reward modelem. |
| **Athey & Imbens (2016)** | Kontekst kauzalnego ML i heterogenicznych efektów leczenia (CATE). Notebook 09: przegląd `econml` (`DRLearner`, `LinearDML`) jako alternatywnego API do DR. |

**Bibliografia:**

- Saito, Y. et al. (2021). *Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.* NeurIPS 2021. [arxiv:2008.07146](https://arxiv.org/abs/2008.07146)
- Dudík, M., Langford, J., Li, L. (2011). *Doubly Robust Policy Evaluation and Learning.* ICML 2011.
- Athey, S., Imbens, G. (2016). *Recursive partitioning for heterogeneous causal effects.* PNAS 113(27), 7353–7360.

**Narzędzia (poza powyższą literaturą):** `dowhy` — testy refutacji modelu kauzalnego (nb. 12); `statsbombpy` — dane piłkarskie (nb. 13).

---

## Status

| Faza | Zakres |
|---|---|
| Faza 1 | Direct Method, EDA, SHAP/OOD (T1–T4) |
| Faza 2 | Propensity, IPS/SNIPS, bias–variance (T5–T8) |
| Faza 3 | DR, DoWhy, StatsBomb, walidacja syntetyczna (T9–T12) |

Raport: [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md)
