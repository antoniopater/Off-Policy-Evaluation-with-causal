# Off-Policy Evaluation z korekcją kauzalną — Raport Finalny

> Projekt: ocena polityk decyzyjnych na danych obserwacyjnych bez ich uruchamiania na żywo.
> Data: 2026-05-27 | Dataset: Open Bandit Dataset (OBD) + StatsBomb La Liga 2015/16

---

## 1. Cel projektu

Zbudowanie i porównanie trzech estymatorów OPE (Direct Method → IPS → Doubly Robust),
które pozwalają ocenić czy inna polityka decyzyjna byłaby lepsza — z korekcją biasu
wynikającego z tego, że dane historyczne nie były zbierane losowo.

---

## 2. Tabela finalna — DM vs IPS vs SNIPS vs DR (OBD)

| Estimator | V̂ | 95% CI | Bias vs V* | CI Width | Uwagi |
|---|---|---|---|---|---|
| **DM** | 0.003515 | [0.003506, 0.003525] | −0.000285 | 0.000019 | Niska wariancja, małe odchylenie od V* |
| **IPS** | 0.004423 | [0.003342, 0.005567] | +0.000623 | 0.002226 | Waga ≈ 1 (policy logging ≈ eval) |
| **SNIPS** | 0.004184 | [0.003161, 0.005267] | +0.000384 | 0.002106 | Nieznacznie niższa wariancja niż IPS |
| **DR** | 0.004223 | [0.003140, 0.005367] | +0.000423 | 0.002227 | Łączy DM i IPS, zbliżony do V* |

**Ground truth proxy V\* = 0.0038** (naive CTR polityki random ≈ uniform eval policy)

### Dekompozycja błędu MSE = Bias² + Variance

| Estimator | Bias² | Variance | MSE | RMSE |
|---|---|---|---|---|
| DM | 8.1e-8 | ~0 | 8.1e-8 | 0.000285 |
| IPS | 3.5e-7 | 4.5e-7 | 7.7e-7 | 0.000875 |
| SNIPS | 1.2e-7 | 3.8e-7 | 5.0e-7 | 0.000707 |

**Wniosek:** DM ma najniższe MSE (dominuje niska wariancja), ale bias pochodzi z modelu nagrody.
IPS/SNIPS mają wyższe wariancję lecz niższy systematyczny bias. DR balansuje między nimi.

*Uwaga: W pierwszej wersji notebooka 02, DM błędnie dawał V_DM = 0.083 (reward model z AUC-PR = 0.015
przeszacowywał nagrody). Po naprawieniu modelu (nowa wersja T8) DM konwerguje do V* = 0.0035.*

---

## 3. Double Robustness — eksperymenty

| Scenariusz | DM | IPS | DR |
|---|---|---|---|
| Baseline (oba modele OK) | 0.003515 | 0.004423 | 0.004223 |
| Zły PS model (random pscores) | 0.003515 | 0.003809 | **0.003818** ≈ V* |
| Zły reward model (stały 0.5) | 0.500000 | 0.004423 | −0.024  |

**Eksperyment A:** zły PS, dobry reward model → DR ≈ V*.  
**Eksperyment B:** reward model stały 0.5 → DR zawodzi; IPS ≈ 0.0044.

---

## 4. Sensitivity Analysis (DoWhy)

| Test | Original ATE | Wynik | p-value | Interpretacja |
|---|---|---|---|---|
| Random Common Cause | −0.003924 | −0.003940 | 0.392 | stabilne |
| Placebo Treatment | −0.003924 | −0.000229 | 0.485 | efekt ≈ 0 |
| Data Subset (80%) | −0.003924 | −0.003936 | 0.485 | stabilne |

**ATE akcji 0 vs reszta:** −0.0039 (akcja 0 nie zwiększa CTR względem innych akcji).
Wynik stabilny we wszystkich testach refutacji — model kauzalny odporny.

---

## 5. Pilot StatsBomb — La Liga 2015/16

**Dane:** 20 meczów, 19 371 podań, 67 707 eventów  
**Treatment:** progressive pass (forward_gain > 10m) — 37% wszystkich podań  
**Outcome:** shot/goal assist — 1.84% nagroda  
**Evaluation policy:** aggressive — 50% progressive passes

| Estimator | V̂ | 95% CI |
|---|---|---|
| DM | 1.5749% | [1.507%, 1.643%] |
| IPS | 2.0487% | [1.580%, 2.548%] |
| SNIPS | 2.1294% | [1.665%, 2.636%] |
| DR | 1.7906% | [1.309%, 2.274%] |

**Naive CTR:** 1.93% (wszystkie podania)

**Wniosek:** Aggressive policy (50% progressive) daje V̂ ≈ 1.6–2.1% vs naive 1.93%.
DR = 1.79% sugeruje że polityka agresywna jest porównywalna z obserwowaną,
ale CI jest szerokie — potrzeba więcej meczów dla pewnych konkluzji.

**ESS = 0.686** — umiarkowany overlap (pscores rozciągają się od 0.04 do 0.99,
bo piłkarze mają preferencje co do progressive passes zależne od pozycji).

---

## 6. Walidacja syntetyczna (T12)

Symulacja ze znanym V* i V_naive (`14_synthetic_validation.ipynb`), pipeline jak w StatsBomb.

| Scenariusz | ESS | Δ | DR V̂ | naive w CI? | V* w CI? |
|---|---|---|---|---|---|
| A. Dobry overlap, Δ=0 | 0.926 | 0 | 0.02155 | tak | tak |
| B. Dobry overlap, Δ=duży | 0.926 | +0.012 | 0.06744 | nie | tak |
| C. StatsBomb overlap, Δ=0 | 0.706 | 0 | 0.02162 | tak | tak |
| D. StatsBomb overlap, Δ=mały | 0.706 | +0.001 | 0.03241 | tak | tak |
| E. StatsBomb overlap, Δ=duży | 0.706 | +0.006 | 0.06652 | nie | tak |

Oracle (prawdziwe P_b, p): DM = V*; IPS/SNIPS/DR odchylenie < 0.0006.

Przy ESS ≈ 0.7 estymator wykrywa duży efekt (E), ale nie rozróżnia Δ=0 od małego Δ (C vs D).
Wynik StatsBomb (DR ≈ naive) jest zgodny z brakiem lub małym efektem, nie z awarią metody.

---

## 7. Struktura notebooków

| Notebook | Tydzień | Temat |
|---|---|---|
| `01_eda.ipynb` | T1 | EDA OBD |
| `02_direct_method.ipynb` | T2 | Direct Method |
| `03_propensity_ips.ipynb` | T3 | Benchmark DM |
| `04_doubly_robust.ipynb` | T4 | SHAP + OOD |
| `07_propensity_scores.ipynb` | T5 | Propensity model |
| `08_ips_snips.ipynb` | T6 | IPS + SNIPS |
| `09_overlap_ess.ipynb` | T7 | Overlap + ESS |
| `10_bias_variance.ipynb` | T8 | Unified benchmark |
| `11_doubly_robust.ipynb` | T9 | Doubly Robust |
| `12_sensitivity_dowhy.ipynb` | T10 | DoWhy |
| `13_statsbomb_pilot.ipynb` | T11 | StatsBomb |
| `14_synthetic_validation.ipynb` | T12 | Walidacja syntetyczna |

---

## 8. Limitations

### Direct Method
- Reward model z AUC-PR ≈ 0.015 (bliski losowemu) — predykcje systematycznie odbiegają od prawdy
- Ekstrapolacja poza manifold treningowy (OOD rate = 0.16%)
- Wrażliwy na misspecification reward modelu

### IPS / SNIPS
- Wagi ≈ 1.0 dla OBD (logging ≈ eval = uniform) — brak prawdziwego tradeoff bias-variance
- Przy naruszeniu overlap (10% zaniżonych pscores): ESS 0.99 → 0.11, V_IPS skacze 5×
- Clipping przy λ > 0.016 zeruje wszystkie wagi (klif)

### Doubly Robust
- Double robustness działa gdy jeden model jest "wystarczająco dobry"
- Katastroficznie zły reward model (stały 0.5) nie jest ratowany przez dobry PS model
- Wymaga treningu dwóch modeli — więcej wariancji niż DM przy małych danych

### StatsBomb Pilot
- Tylko 20 meczów — szerokie CI uniemożliwiają silne wnioski
- Outcome proxy (shot/goal assist) jest rzadki (1.84%) — model słabo się uczy
- Binary treatment jest uproszczeniem vs ciągła skala postępu

---

## 9. Reprodukcja

```bash
git clone <repo-url>
cd Off-Policy-Evaluation-with-causal
uv sync
uv run jupyter lab
```

Uruchom notebooki w kolejności: 01 → 02 → 03 → 04 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14.

---

## 10. Literatura

- Saito, Y. et al. (2021). *Open Bandit Dataset and Pipeline.* NeurIPS 2021.
- Dudík, M., Langford, J., Li, L. (2011). *Doubly Robust Policy Evaluation.* ICML 2011.
- Athey, S., Imbens, G. (2016). *Recursive partitioning for heterogeneous causal effects.* PNAS.
