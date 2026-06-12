# Przewodnik prezentera — Off-Policy Evaluation z korekcją kauzalną

> Czas prezentacji: 15–20 minut + pytania  
> Slajdy: 15 | Wykresy: `presentation/figures/`  
> Format każdego slajdu: **TYTUŁ → WYKRES → CO MÓWIĆ → LICZBY DO ZACYTOWANIA**

---

## BLOK 1 — PROBLEM (slajdy 1–3)

---

### Slajd 1 — Tytuł

**Tytuł slajdu:** Off-Policy Evaluation z korekcją kauzalną

**Wykres:** brak (slajd tytułowy)

**Co mówić:**
Projekt odpowiada na jedno konkretne pytanie: *czy inna polityka decyzyjna byłaby lepsza — i jak to zmierzyć bez wdrażania jej na żywo?* To jest problem Off-Policy Evaluation. Przez 12 tygodni zbudowałem trzy estymatory OPE rosnące w złożoności — Direct Method, IPS, Doubly Robust — na prawdziwych danych z e-commerce i na danych piłkarskich z La Liga.

---

### Slajd 2 — Motywacja: selection bias

**Tytuł slajdu:** Dlaczego naiwny model ML zawodzi na danych obserwacyjnych

**Wykres:** `01_selection_bias.png`

**Co mówić:**
Wyobraź sobie bank, który przez rok odrzucał wnioski kredytowe klientów z niską oceną. Mamy dane historyczne — ale *tylko o klientach, których bank obsłużył*. Teraz pytasz: czy inna polityka kredytowa byłaby lepsza? Naiwny model ML uczy się historycznej polityki banku — myli jakość decyzji z jakością kontekstu, w którym decyzja była podjęta. To jest bias selekcji.

To samo w piłce nożnej: piłkarz gra progresywnie tylko w określonych sytuacjach. Model nie może po prostu sprawdzić "co by było gdyby zagrał inaczej" — bo te obserwacje nie istnieją w danych.

**Liczby do zacytowania:**
- Dane OBD mają **dwie polityki**: Random (losowa) i BernoulliTS (rekomendacyjna) — obie zbierane historycznie, obie zawierają bias selekcji względem nowej polityki, którą chcemy ocenić.

---

### Slajd 3 — Co to jest OPE

**Tytuł slajdu:** Off-Policy Evaluation — pytanie counterfactual

**Wykres:** brak (formuły na slajdzie)

**Co mówić:**
OPE odpowiada na pytanie: *gdybyśmy przez ostatni rok stosowali politykę X zamiast Y — ile byśmy zyskali?* I robi to uczciwie, bez uruchamiania nowej polityki na żywo.

Kluczowe założenie: dane są obserwacyjne, więc decyzje nie były losowe. Żeby porównywać polityki, musimy korygować bias selekcji. Buduję trzy estymatory, każdy z innym podejściem do tej korekcji.

**Formuły (pokazać na slajdzie):**
```
V* = E[r | policy π_new]   ← to chcemy wiedzieć

V_DM  = E[ f(s, a_new) ]                           ← model
V_IPS = E[ r × π_new(a|s) / π_old(a|s) ]          ← ważenie
V_DR  = V_IPS + korekcja z DM                      ← kombinacja
```

---

## BLOK 2 — DANE (slajd 4)

---

### Slajd 4 — Open Bandit Dataset

**Tytuł slajdu:** Dane: Open Bandit Dataset (Zozo Research, NeurIPS 2021)

**Wykresy:** `02_reward_distribution.png`, `03_action_distribution.png`

**Co mówić:**
Używam Open Bandit Dataset — prawdziwych danych z japońskiej platformy e-commerce Zozo. To rzadkość w OPE: mamy **ground truth**, bo obie polityki (Random i BernoulliTS) były faktycznie wdrożone i znamy ich prawdziwe wartości.

- Random — polityka losowa, CTR ≈ 0.38% → **V\* = 0.0038**
- BernoulliTS — polityka rekomendacyjna, wyższy CTR

Na wykresach po prawej stronie widać rozkład nagród (rzadkie kliknięcia — 0.38%) i rozkład 80 akcji (produkty modowe). Model musi działać dobrze mimo bardzo rzadkiej nagrody.

**Liczby do zacytowania:**
- **1.3 miliona obserwacji** (pełny dataset), używamy small set ~10K do trenowania
- **80 akcji** (produkty)
- **CTR ≈ 0.38%** — nagroda jest bardzo rzadka (klasa imbalanced 260:1)
- Ground truth V\* = **0.0038**

---

## BLOK 3 — METODA 1: DIRECT METHOD (slajdy 5–6)

---

### Slajd 5 — Direct Method

**Tytuł slajdu:** Direct Method — reward model jako estymator polityki

**Wykresy:** `04_dm_vs_naive_ci.png`, `05_pr_curve.png`

**Co mówić:**
Direct Method to najprostsze podejście: trenujemy model ML (XGBoost), który przewiduje nagrodę po danej akcji. Pytamy go: "co by przewidziałeś gdybyśmy stosowali nową politykę?" Szacujemy V_DM = średnia przewidywana nagroda.

Wykres po lewej pokazuje: V_DM = **0.0035**, bardzo blisko ground truth V\* = 0.0038. Przedziały ufności są wąskie — DM ma niską wariancję.

Ale jest problem: krzywa precision-recall (wykres po prawej) — **AUC-PR = 0.0056**, czyli praktycznie na poziomie baseline (CTR ≈ 0.0037). Model w zasadzie nie odróżnia klasy "kliknięcie" od "brak kliknięcia" — przewiduje blisko średniej dla każdej akcji. To nie jest błąd implementacji — to fundamentalna trudność: nagroda jest zbyt rzadka (0.38%) żeby model nauczył się precyzyjnie przewidywać per-akcja.

**Liczby do zacytowania:**
- V_DM = **0.003515**
- AUC-PR = **0.0056** (≈ losowy — baseline CTR walidacyjny ≈ 0.0037)
- 95% CI = [0.003506, 0.003525] — **bardzo wąskie**

**Ważna uwaga interpretacyjna:** to, że V_DM ≈ V*, jest w dużej mierze efektem *kalibracji do średniej* (model przewiduje wartości bliskie globalnemu CTR dla każdej akcji), a nie dowodem na trafne różnicowanie nagrody między akcjami. Niska wariancja DM nie oznacza niskiego ryzyka błędu, jeśli różne akcje faktycznie mają różne wartości — model tego po prostu nie widzi.

---

### Slajd 6 — Słabość DM: OOD i SHAP

**Tytuł slajdu:** Słabość DM — ekstrapolacja i bias modelu nagrody

**Wykresy:** `06_shap_top20.png`, `07_ood_risk_map.png`

**Co mówić:**
SHAP pokazuje które cechy napędzają predykcje reward modelu. Widać, że model opiera się głównie na cechach kontekstu użytkownika — ale nowa polityka może rekomendować produkty w rejonach przestrzeni cech, których model nigdy nie widział w treningu. To jest **ekstrapolacja poza manifold** (out-of-distribution).

Mapa ryzyka OOD (wykres po prawej) pokazuje obserwacje, które leżą daleko od danych treningowych. OOD rate = 0.16% — mały, ale te obserwacje mogą dominować estymację.

**Wniosek z tego slajdu:** DM jest dobry gdy reward model jest dobry. Gdy model zawodzi poza danymi treningowymi — DM ma systematyczny bias, który nie znika przy większej próbie.

**Liczby do zacytowania:**
- OOD rate = **0.16%** obserwacji poza manifoldem treningowym
- Bias DM = **−0.000285** (model systematycznie zaniża nagrodę)

---

## BLOK 4 — METODA 2: IPS (slajdy 7–8)

---

### Slajd 7 — Propensity scores i IPS

**Tytuł slajdu:** IPS — ważenie przez propensity scores

**Wykresy:** `08_calibration_curve.png`, `09_ips_weights_hist.png`

**Co mówić:**
Inverse Propensity Scoring to inne podejście: nie modelujemy nagrody, tylko **prawdopodobieństwo podjęcia decyzji**. Trenujemy model P(a|s) — "ile wynosi szansa, że stara polityka wybrała właśnie tę akcję?". Obserwacje zaskakujące dla starej polityki (niska szansa) dostają **dużą wagę** — bo gdyby nowa polityka je wybierała, byłyby niedoreprezentowane.

```
V_IPS = E[ r × π_new(a|s) / π_old(a|s) ]
```

Krzywa kalibracji (wykres po lewej) sprawdza czy nasz model propensity dobrze się kalibruje — czy "60% przewidywane" naprawdę oznacza 60% w rzeczywistości. Histogram wag (wykres po prawej) pokazuje rozkład π_new/π_old — dla OBD wagi są skupione wokół 1 (obie polityki podobne).

**Liczby do zacytowania:**
- Model P(a|s): **80-klasowy XGBoost**, multiclass logloss ≈ 4.37
- Top-1 accuracy: **1.75%** (losowy baseline = 1.25%) — uczymy się czegoś, ale mało
- ESS (Effective Sample Size) = **9877 / 10000 = 98.77%** — wagi prawie równe (bo polityki są podobne)
- V_IPS = **0.004423**

---

### Slajd 8 — Kiedy IPS eksploduje

**Tytuł slajdu:** Słabość IPS — overlap violation i eksplozja wariancji

**Wykresy:** `10_bias_variance_clipping.png`, `11_overlap_violation.png`

**Co mówić:**
IPS ma fundamentalny problem: gdy nowa polityka chce oceniać akcje, które stara polityka wybierała bardzo rzadko, wagi π_new/π_old mogą urosnąć do nieskończoności. To jest naruszenie **założenia overlap** — wspólnego supportu.

Lewy wykres pokazuje eksperyment z clippingiem: obcinamy wagi do wartości maksymalnej λ. Przy małych λ — wzrasta bias (ignorujemy ważne obserwacje). Przy dużych λ — wariancja wybucha. Jest klif: przy λ > 0.016 wszystkie wagi OBD zerują się.

Prawy wykres (histogramy wag IPS) symuluje naruszenie overlap: zaniżamy o rząd wielkości pscore dla 10% obserwacji. Po lewej — oryginalne wagi (max=1.32, ESS ratio=0.988). Po prawej — wagi po naruszeniu (max=124.4, ESS ratio=0.113, skala logarytmiczna). ESS spada z 0.99 do **0.11** — efektywna próba kurczy się 9-krotnie, a V_IPS skacze 5×.

**Liczby do zacytowania:**
- Przy naruszeniu overlap: ESS **0.99 → 0.11**
- V_IPS skacze **5×** przy usunięciu 10% małych pscore
- Clipping klif przy λ > **0.016** (zeruje wszystkie wagi w OBD)

---

## BLOK 5 — METODA 3: DOUBLY ROBUST (slajd 9)

---

### Slajd 9 — Doubly Robust Estimator

**Tytuł slajdu:** Doubly Robust — Doubly Robust

**Wykres:** `14_dr_robustness.png`

**Co mówić:**
Doubly Robust łączy DM i IPS. Kluczowa własność: wynik jest poprawny gdy **choć jedna** z metod działa prawidłowo. Obie muszą zawieść jednocześnie, żeby DR dał zły wynik.

```
V_DR = V_IPS + E[ (r - f(s,a)) × π_new/π_old ]
```

Wykres pokazuje trzy scenariusze:
- **Baseline** (oba modele OK): DR ≈ 0.004223 ≈ V\*
- **Zły PS model** (losowe pscores): DR = **0.003818 ≈ V\*** — DR ratuje się dzięki dobremu reward modelowi
- **Zły reward model** (stały 0.5): DR = −0.024 nie — katastrofalnie zły model nie jest ratowany

**Wniosek:** Double robustness działa gdy jeden model jest "wystarczająco dobry", nie gdy jest ekstremalnie błędny. W praktyce to mocna gwarancja — rzadko oba modele zawodzą jednocześnie.

**Liczby do zacytowania:**
- Scenariusz zły PS: DR = **0.003818** ≈ V\* = 0.0038
- Scenariusz zły reward (stały 0.5 vs prawdziwe 0.38%): DR = **−0.024** nie — 100× za duże
- V_DR baseline = **0.004223**

---

## BLOK 6 — WYNIKI GŁÓWNE (slajdy 10–11)

---

### Slajd 10 — Unified benchmark

**Tytuł slajdu:** Wyniki — DM vs IPS vs SNIPS vs DR na Open Bandit Dataset

**Wykres:** `12_unified_benchmark_ci.png`

**Co mówić:**
Teraz porównuję wszystkie estymatory na tych samych danych, z bootstrapowanymi przedziałami ufności (n=200 resampli). Ground truth V\* = 0.0038.

Wykres pokazuje: DM ma najwęższy CI — prawie punkt. IPS i SNIPS mają szerokie CI. DR leży pomiędzy. Wszystkie estymatory trafiają V\* w swoim przedziale ufności.

SNIPS (Self-Normalized IPS) jest wersją znormalizowaną IPS — nieznacznie zawęża CI względem IPS, bo normalizuje wagi do sumy 1.

| Estymator | V̂ | 95% CI | Uwagi |
|---|---|---|---|
| DM | 0.003515 | [0.003506, 0.003525] | Bardzo wąskie CI |
| IPS | 0.004423 | [0.003342, 0.005567] | Szerokie CI |
| SNIPS | 0.004184 | [0.003161, 0.005267] | Nieznacznie węższe |
| DR | 0.004223 | [0.003140, 0.005367] | Balans |

**Liczby do zacytowania:**
- Ground truth V\* = **0.0038**
- CI width DM: **0.000019** vs CI width IPS: **0.002226** — IPS 117× szersze
- Wszystkie estymatory zawierają V\* w swoich CI

---

### Slajd 11 — MSE decomposition (GŁÓWNY WYKRES)

**Tytuł slajdu:** MSE = Bias² + Variance — kluczowy tradeoff w OPE

**Wykres:** `13_bias_variance_decomposition.png`

**Co mówić:**
To jest centralny wynik projektu. Każdy estymator ma swój profil błędu:

- **DM**: MSE dominuje Bias². Wariancja praktycznie zero — każde uruchomienie daje ten sam wynik. Ale bias pochodzi z modelu nagrody i nie zniknie przy większej próbie.
- **IPS**: MSE dominuje Variance. Prawie nieobciążony (bias² mały), ale wysoka wariancja — wyniki skaczą między uruchomieniami.
- **SNIPS**: Kompromis — niższy bias niż IPS, niższa wariancja niż IPS, ale w OBD nie jest dramatycznie lepszy bo wagi są bliskie 1.

Wniosek: nie ma jednego "najlepszego" estymatora. Wybór zależy od tego czy ufamy modelowi nagrody (DM) czy modelowi propensity (IPS).

| Estymator | Bias² | Variance | MSE | RMSE |
|---|---|---|---|---|
| DM | 8.1e-8 | ~0 | **8.1e-8** | 0.000285 |
| IPS | 3.5e-7 | 4.5e-7 | 7.7e-7 | 0.000875 |
| SNIPS | 1.2e-7 | 3.8e-7 | 5.0e-7 | 0.000707 |

**Liczby do zacytowania:**
- DM MSE = **8.1e-8** — 10× niższe niż IPS
- IPS Variance = **4.5e-7** — dominuje w IPS
- SNIPS redukuje wariancję IPS o **15%** w tym datasecie

---

## BLOK 7 — WALIDACJA KAUZALNA (slajd 12)

---

### Slajd 12 — Sensitivity analysis (DoWhy)

**Tytuł slajdu:** Sensitivity Analysis — czy wynik jest kauzalny czy przypadkowy?

**Wykres:** `15_sensitivity_refutations.png`

**Co mówić:**
Do tej pory sprawdzałem dokładność estymatorów — ale skąd wiemy, że odkryty efekt jest kauzalny, a nie artefaktem danych? Używam biblioteki DoWhy, która implementuje trzy **testy refutacji**:

1. **Random Common Cause** — dodaję losową zmienną do modelu. Jeśli model jest stabilny, ATE się nie zmieni.
2. **Placebo Treatment** — zastępuję prawdziwy treatment losowym. Estymowany efekt powinien zniknąć.
3. **Data Subset** — uruchamiam analizę na losowym 80% danych. Wynik powinien być zbliżony.

Wykres pokazuje: wszystkie testy przeszły. ATE = −0.0039 akcji 0 jest stabilne.

**Liczby do zacytowania:**
- Original ATE = **−0.003924** (akcja 0 nie zwiększa CTR)
- Random Common Cause: ATE = −0.003940, p = **0.392** (stabilne, p > 0.05)
- Placebo Treatment: ATE = −0.000229 (efekt znika dla losowego treatment tak)
- Data Subset (80%): ATE = −0.003936, p = **0.485** (stabilne tak)

---

## BLOK 8 — PILOT PIŁKARSKI (slajdy 13–14)

---

### Slajd 13 — StatsBomb La Liga

**Tytuł slajdu:** Pilot sportowy — OPE na danych piłkarskich (La Liga 2015/16)

**Wykres:** `16_statsbomb_eda.png`

**Co mówić:**
Ostatni krok to sprawdzenie czy pipeline OPE działa poza e-commerce — na zupełnie innej domenie. Używam StatsBomb Open Data: dane meczowe z La Liga 2015/16 z formatem 360° — pozycje wszystkich zawodników.

Definiuję **treatment** jako *progressive pass* — podanie o więcej niż 10 metrów do przodu. **Outcome** to czy akcja zakończyła się szansą strzelecką lub asystą.

Wykres EDA pokazuje: 37% podań to progressive passes, rozkład różni się zależnie od pozycji zawodnika (skrzydłowi i napastnicy częściej grają progresywnie).

**Liczby do zacytowania:**
- **20 meczów**, **19,371 podań**, **67,707 eventów**
- Treatment (progressive pass): **37%** wszystkich podań
- Outcome (shot/goal assist): **1.84%** nagród
- Evaluation policy: aggressive — **50%** progressive passes

---

### Slajd 14 — Wyniki OPE na piłce

**Tytuł slajdu:** Wyniki — Czy aggressive policy byłaby lepsza dla La Liga?

**Wykres:** `17_statsbomb_ope_results.png`

**Co mówić:**
Pytanie: gdybyśmy przez całą La Liga wymuszali strategię "aggressive" (50% progressive passes zamiast obserwowanych 37%) — czy byłoby więcej szans strzeleckich?

Wyniki OPE:
- DM: 1.57% — konserwatywna estymata
- IPS: 2.05%, SNIPS: 2.13% — wyższe, ale szerokie CI
- DR: 1.79% — balans między DM i IPS

Naive CTR = 1.93% (wszystkie podania bez korekty). DR sugeruje, że aggressive policy daje porównywalny wynik — nie jednoznacznie lepsza.

ESS = 0.686 — umiarkowany overlap. Piłkarze mają silne preferencje zależne od pozycji, co utrudnia IPS.

**Ważny wniosek:** Z 20 meczów CI jest za szerokie na mocne konkluzje. Ale pipeline działa — te same metody co na OBD działają na danych sportowych. To jest bezpośredni most do badań z doktorem.

**Liczby do zacytowania:**
- V_DR = **1.79%** [1.31%, 2.27%] vs naive CTR = **1.93%**
- V_SNIPS = **2.13%** — najwyższa estymata
- ESS = **0.686** (vs 0.989 w OBD — piłka ma gorszy overlap)
- Potrzeba co najmniej **100 meczów** dla wąskich CI (docelowy dataset do badań)

---

## BLOK 9 — WNIOSKI (slajd 15)

---

### Slajd 15 — Wnioski i dalsze kroki

**Tytuł slajdu:** Wnioski — kiedy używać DM, IPS, DR?

**Wykres:** brak (tabela na slajdzie)

**Co mówić:**
Projekt pokazał, że OPE to nie jeden algorytm, ale rodzina estymatorów z różnymi profilami błędów. Kluczowe wnioski:

**Kiedy używać DM:**
Gdy ufasz modelowi nagrody. Jeśli masz dużo danych i dobry model ML — DM daje najniższe MSE i wąskie CI. Ostrzeżenie: jeśli model ekstrapoluje poza trening, bias DM jest systematyczny.

**Kiedy używać IPS/SNIPS:**
Gdy masz dobry model propensity i overlap assumption jest spełnione. Nieobciążony estymator, ale wymaga kontroli ESS. Clipping pomaga, ale zawsze zwiększa bias.

**Kiedy używać DR:**
W produkcji — DR to domyślny wybór. Wymaga treningu dwóch modeli, ale gwarancja double robustness jest silna. Zawiedzie tylko gdy oba modele są złe jednocześnie.

**Ograniczenia:**
- OBD w tym projekcie: wagi IPS ≈ 1 (polityki podobne) — nie widzimy pełnego tradeoff bias-variance
- StatsBomb: 20 meczów to za mało — CI zbyt szerokie
- Binary treatment jest uproszczeniem (vs ciągła miara postępu)

**Dalsze kroki:**
- Rozszerzenie na pełny dataset StatsBomb (38 kolejek, 5 sezonów)
- Continuous treatment (stopień postępu podania)
- Integracja z modelem wartości akcji (Q-function) dla RL
- Projekt badawczy z doktorem — OPE jako framework dla taktyki piłkarskiej

---

## Appendix — szybkie liczby do pytań

| Pytanie | Odpowiedź |
|---|---|
| Ile danych? | OBD: 1.3M obs (small set: 10K) |
| Ground truth V\*? | 0.0038 (naive CTR polityki random) |
| Najlepszy estymator MSE? | DM: MSE = 8.1e-8 |
| Najniższa wariancja? | DM: Variance ≈ 0 |
| ESS w OBD? | 98.77% (wagi ≈ 1) |
| ESS w StatsBomb? | 68.6% |
| DR kiedy działa? | Gdy choć jeden model (PS lub reward) jest dobry |
| DR kiedy nie działa? | Gdy reward model stały 0.5 → DR = −0.024 |
| Notebooki do przejrzenia? | 10_bias_variance, 11_doubly_robust, 13_statsbomb_pilot |
| Tag wersji? | v1.0 |
| Reprodukcja? | `uv sync && uv run jupyter lab` |

---

## Mapa wykresów

| Plik | Slajd | Co pokazuje |
|---|---|---|
| `01_selection_bias.png` | S2 | Bias selekcji — motywacja |
| `02_reward_distribution.png` | S4 | CTR ≈ 0.38%, klasa imbalanced |
| `03_action_distribution.png` | S4 | Rozkład 80 akcji |
| `04_dm_vs_naive_ci.png` | S5 | V_DM = 0.0035 z CI |
| `05_pr_curve.png` | S5 | AUC-PR = 0.0056 reward modelu (T8, max_depth=4) |
| `06_shap_top20.png` | S6 | Top 20 feature importance |
| `07_ood_risk_map.png` | S6 | OOD — ekstrapolacja poza manifold |
| `08_calibration_curve.png` | S7 | Kalibracja modelu P(a\|s) |
| `09_ips_weights_hist.png` | S7 | Histogram wag IPS |
| `10_bias_variance_clipping.png` | S8 | Tradeoff bias-variance clipping |
| `11_overlap_violation.png` | S8 | ESS 0.99→0.11 przy naruszeniu overlap |
| `12_unified_benchmark_ci.png` | S10 | DM vs IPS vs SNIPS vs DR z CI |
| `13_bias_variance_decomposition.png` | S11 | MSE = Bias² + Variance (GŁÓWNY) |
| `14_dr_robustness.png` | S9 | Double robustness experiment |
| `15_sensitivity_refutations.png` | S12 | DoWhy — 3 testy refutacji |
| `16_statsbomb_eda.png` | S13 | La Liga — rozkład progressive passes |
| `17_statsbomb_ope_results.png` | S14 | Wyniki OPE na danych piłkarskich |
