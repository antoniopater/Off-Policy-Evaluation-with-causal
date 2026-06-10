# Scenariusz prezentacji — `OPE_Prezentacja_v2.pptx` (18 slajdów)

> Kompletny przewodnik slajd po slajdzie: co jest na slajdzie, co mówić,
> co dany slajd tłumaczy i jaki jest jego cel w narracji całej prezentacji.
> Czas: ok. 18–22 minuty + pytania.
>
> Ten scenariusz jest dopasowany 1:1 do `OPE_Prezentacja_v2.pptx` (18 slajdów).
> Jest rozszerzeniem `PRESENTATION_GUIDE.md` (wersja 15-slajdowa) — slajdy
> 5, 7 i 8 z tamtej wersji zostały tu rozbite na po dwa slajdy (5+6, 8+9, 10+11),
> żeby każdy wykres miał własny slajd i więcej miejsca na komentarz.
>
> **Spójność liczb (stan po korekcie):** wszystkie wartości V_DM/V_IPS/V_SNIPS/V_DR,
> CI i AUC-PR pochodzą z `notebooks/10_bias_variance.ipynb` (T8, model `max_depth=4`)
> i `results/FINAL_REPORT.md`. AUC-PR reward modelu = **0.0056** (nie 0.015 — to była
> wartość ze starszego modelu z notebooków 02/03, `max_depth=6`, który dawał
> V_DM≈0.075–0.083, czyli ~20x za dużo. Patrz adnotacje w `02_direct_method.ipynb`
> i `03_propensity_ips.ipynb`).

---

## Mapa slajdów

| # | Tytuł slajdu | Blok | Wykres/tabela |
|---|---|---|---|
| 1 | Off-Policy Evaluation z korekcją kauzalną | — | tytułowy |
| 2 | Dlaczego naiwny model ML zawodzi na danych obserwacyjnych | BLOK 1 | `01_selection_bias.png` |
| 3 | Off-Policy Evaluation — pytanie counterfactual | BLOK 1 | formuły |
| 4 | Dane: Open Bandit Dataset | BLOK 2 | `02_reward_distribution.png`, `03_action_distribution.png` |
| 5 | Direct Method — V_DM ≈ V* | BLOK 3 | `04_dm_vs_naive_ci.png` |
| 6 | Reward model: AUC-PR = 0.0056 | BLOK 3 | `05_pr_curve.png` (T8) |
| 7 | Słabość DM — OOD i SHAP | BLOK 3 | `06_shap_top20.png`, `07_ood_risk_map.png` |
| 8 | IPS — model propensity P(a\|s) i kalibracja | BLOK 4 | `08_calibration_curve.png` |
| 9 | IPS — ważenie przez propensity scores | BLOK 4 | `09_ips_weights_hist.png` + tabela |
| 10 | Clipping — tradeoff bias-variance | BLOK 4 | `10_bias_variance_clipping.png` |
| 11 | Overlap violation — ESS 0.99 → 0.11 | BLOK 4 | `11_overlap_violation.png` |
| 12 | Doubly Robust — state of the art w OPE | BLOK 5 | `14_dr_robustness.png` + tabela |
| 13 | Wyniki — DM vs IPS vs SNIPS vs DR | BLOK 6 | `12_unified_benchmark_ci.png` + tabela |
| 14 | MSE = Bias² + Variance (GŁÓWNY) | BLOK 6 | `13_bias_variance_decomposition.png` + tabela |
| 15 | Sensitivity Analysis (DoWhy) | BLOK 7 | `15_sensitivity_refutations.png` + tabela |
| 16 | Pilot sportowy — StatsBomb La Liga | BLOK 8 | `16_statsbomb_eda.png` |
| 17 | Wyniki — aggressive policy w La Liga | BLOK 8 | `17_statsbomb_ope_results.png` + tabela |
| 18 | Wnioski — kiedy używać DM, IPS, DR? | — | tabela porównawcza |

---

## Slajd 1 — Off-Policy Evaluation z korekcją kauzalną

**Co jest na slajdzie:** Tytuł, podtytuł "Direct Method · IPS · Doubly Robust",
źródła danych (Open Bandit Dataset NeurIPS 2021, StatsBomb La Liga 2015/16),
"12 tygodni · 3 estymatory · 1 pytanie counterfactual".

**Co mówić:**
"Projekt odpowiada na jedno konkretne pytanie: czy inna polityka decyzyjna
byłaby lepsza — i jak to zmierzyć bez wdrażania jej na żywo? To jest problem
Off-Policy Evaluation (OPE). Przez 12 tygodni zbudowałem trzy estymatory OPE,
rosnące w złożoności — Direct Method, IPS i Doubly Robust — i przetestowałem
je na dwóch zbiorach danych: Open Bandit Dataset (prawdziwe dane e-commerce
z japońskiej platformy Zozo, NeurIPS 2021) oraz na danych piłkarskich
StatsBomb z La Liga 2015/16."

**Czego to uczy / Cel slajdu:**
Ustawia ramy całej prezentacji — jedno pytanie (counterfactual), trzy metody,
dwie domeny danych. Słuchacz od razu wie, czego się spodziewać i dlaczego
warto słuchać dalej (prawdziwe dane, znana ground truth, most do badań
piłkarskich).

---

## Slajd 2 — Dlaczego naiwny model ML zawodzi na danych obserwacyjnych

**Co jest na slajdzie:** Wykres `01_selection_bias.png` ilustrujący bias
selekcji.

**Co mówić:**
"Wyobraź sobie bank, który przez rok odrzucał wnioski kredytowe klientów
z niską oceną. Mamy dane historyczne — ale tylko o klientach, których bank
faktycznie obsłużył. Teraz pytamy: czy inna polityka kredytowa byłaby lepsza?
Naiwny model ML uczy się historycznej polityki banku — myli jakość decyzji
z jakością kontekstu, w którym ta decyzja została podjęta. To jest **bias
selekcji**.

To samo dzieje się w piłce nożnej: piłkarz gra progresywnie tylko w określonych
sytuacjach — gdy ma przestrzeń, gdy wynik tego wymaga. Model nie może po prostu
sprawdzić 'co by było, gdyby zagrał inaczej', bo takie obserwacje po prostu nie
istnieją w danych."

**Czego to uczy / Cel slajdu:**
To jest **motywacja całego projektu** — dlaczego standardowe ML (predykcja)
nie wystarcza do oceny polityk decyzyjnych i dlaczego potrzebujemy aparatu
przyczynowego (causal inference / OPE). Bez tego slajdu reszta prezentacji
brzmi jak "kolejny model ML" — z nim widać, że to inny problem.

**Kluczowe liczby:**
Dane OBD mają dwie polityki logujące: Random (losowa) i BernoulliTS
(rekomendacyjna) — obie zbierane historycznie i obie obciążone biasem
selekcji względem nowej polityki, którą chcemy ocenić.

---

## Slajd 3 — Off-Policy Evaluation — pytanie counterfactual

**Co jest na slajdzie:** Definicja pytania counterfactual + trzy formuły
(V*, V_DM, V_IPS, V_DR) + trzy karty: Direct Method / IPS / Doubly Robust
z jednozdaniowym opisem każdej.

**Co mówić:**
"OPE odpowiada na pytanie: *gdybyśmy przez ostatni rok stosowali politykę X
zamiast Y — ile byśmy zyskali?* I robi to bez uruchamiania nowej polityki na
żywo — czysto na danych historycznych.

Kluczowe założenie: dane są obserwacyjne, więc decyzje nie były losowe.
Żeby uczciwie porównać polityki, musimy skorygować ten bias selekcji.
Buduję trzy estymatory, każdy z innym podejściem do tej korekcji:

- **Direct Method** — trenuje model nagrody f(s,a) i pyta go wprost: 'co by
  przewidziałeś dla nowej polityki?'
- **IPS** — nie modeluje nagrody, tylko przeważa obserwowane nagrody przez
  stosunek π_new/π_old — czyli 'jak bardzo nowa polityka różni się od starej
  w tym konkretnym przypadku'.
- **Doubly Robust** — łączy obie metody. Działa poprawnie, gdy choć jedna
  z nich jest dobra."

```
V* = E[r | policy π_new]                    ← to chcemy poznać

V_DM  = E[ f(s, a_new) ]                     ← model nagrody
V_IPS = E[ r × π_new(a|s) / π_old(a|s) ]    ← ważenie obserwacji
V_DR  = V_IPS + korekcja z DM                ← kombinacja
```

**Czego to uczy / Cel slajdu:**
To jest "słownik" prezentacji — wszystkie symbole (V*, V_DM, V_IPS, V_DR),
których słuchacz będzie potrzebował na kolejnych 15 slajdach, są wprowadzone
tutaj w jednym miejscu. Warto zostawić ten slajd "w pamięci" — można do niego
wrócić gestem ("przypomnijmy formułę z wcześniej...") przy slajdach 5, 9, 12.

---

## Slajd 4 — Dane: Open Bandit Dataset (Zozo Research, NeurIPS 2021)

**Co jest na slajdzie:** Dwa wykresy (`02_reward_distribution.png`,
`03_action_distribution.png`) + 5 kafelków z liczbami: 1.3M obserwacji, ~10K
small set, 80 akcji, 0.38% CTR, 260:1 imbalance ratio, V* = 0.0038.

**Co mówić:**
"Używam Open Bandit Dataset — prawdziwych danych z japońskiej platformy
e-commerce Zozo. To rzadkość w świecie OPE: mamy **ground truth**, bo obie
polityki logujące — Random i BernoulliTS — były faktycznie wdrożone na
produkcji i znamy ich rzeczywiste wartości CTR.

Polityka Random (losowa, każda z 80 akcji z prawdopodobieństwem 1/80) ma
naive CTR ≈ 0.38%. To jest nasz **ground truth proxy V\* = 0.0038** — bo
polityka ewaluowana w tym projekcie (uniform, π_eval = 1/80) jest praktycznie
tożsama z polityką Random.

Na wykresach po prawej widać dwa wyzwania: rozkład nagród jest skrajnie
niezbalansowany (klikamy w 0.38% przypadków, czyli stosunek 260:1), a 80 akcji
to produkty modowe o bardzo różnej popularności. Każdy model — czy to reward
model dla DM, czy propensity model dla IPS — musi sobie poradzić z tą rzadką
nagrodą."

**Czego to uczy / Cel slajdu:**
Wprowadza dataset i — co najważniejsze — **ground truth V\* = 0.0038**, do
którego będziemy porównywać każdy estymator na kolejnych slajdach. Pokazuje
też od razu, dlaczego to zadanie jest trudne (rzadka nagroda, imbalance) —
to przygotowuje grunt pod problem AUC-PR na slajdzie 6.

**Kluczowe liczby:**
- 1.3 mln obserwacji (pełny dataset), small set ~10K do treningu
- 80 akcji, CTR ≈ 0.38% (imbalance 260:1)
- Ground truth **V\* = 0.0038**

---

## Slajd 5 — Direct Method — V_DM = 0.003515 ≈ V* = 0.0038

**Co jest na slajdzie:** Wykres `04_dm_vs_naive_ci.png` (słupki: V* = 0.0038
vs V_DM = 0.003515 z 95% CI [0.003506, 0.003525]) + lista "Wyniki i wnioski"
po prawej: V_DM ≈ V*, CI bardzo wąskie, najniższa wariancja, AUC-PR = 0.0056
(model ≈ losowy), bias z ekstrapolacji OOD.

**Co mówić:**
"Direct Method to najprostsze podejście: trenujemy model ML — tu XGBoost
z `max_depth=4` i early stopping — który przewiduje prawdopodobieństwo
kliknięcia f(s,a) dla danego kontekstu i akcji. Pytamy go: 'co byś przewidział,
gdybyśmy stosowali nową, jednostajną politykę (każda z 80 akcji z
prawdopodobieństwem 1/80)?' Uśredniamy te predykcje po wszystkich akcjach —
to jest V_DM.

Wynik: **V_DM = 0.003515**, bardzo blisko ground truth **V\* = 0.0038**.
Przedział ufności (bootstrap, n=200) jest wąski: [0.003506, 0.003525] —
DM ma najniższą wariancję ze wszystkich trzech estymatorów.

Ale — i to jest ważne — na następnym slajdzie pokażę, że ten model ma
**AUC-PR = 0.0056**, czyli praktycznie na poziomie losowego zgadywania.
To rodzi pytanie: jak model, który prawie nie odróżnia 'kliknięcia' od
'braku kliknięcia', może dać tak dokładny wynik V_DM?"

**Czego to uczy / Cel slajdu:**
Pokazuje **pierwszy, pozytywny wynik** projektu — DM trafia w ground truth
z bardzo małą wariancją. Ale celowo zostawia "haczyk" (AUC-PR = 0.0056), który
slajd 6 rozwinie. To jest moment, w którym warto **zbudować napięcie**: "wynik
wygląda świetnie — ale czy naprawdę rozumiemy, dlaczego?"

**Kluczowe liczby:**
- V_DM = **0.003515** ≈ V* = **0.0038**
- 95% CI = **[0.003506, 0.003525]** (szerokość 0.000019 — bardzo wąskie)
- Bias = −0.000285 (DM lekko zaniża V*)

---

## Slajd 6 — Reward model: AUC-PR = 0.0056 — problem z rzadką nagrodą

**Co jest na slajdzie:** Krzywa precision-recall reward modelu z T8
(`05_pr_curve.png`), AUC-PR = 0.0056, linia bazowa (baseline CTR ≈ 0.0037),
krzywa praktycznie pokrywająca się z baseline na całej długości.

**Co mówić:**
"To jest najważniejszy slajd dla zrozumienia *dlaczego* DM działa tak, jak
działa. Krzywa precision-recall reward modelu daje **AUC-PR = 0.0056** —
to jest *poniżej* poziomu losowego klasyfikatora dla tego zbioru
(baseline CTR walidacyjny ≈ 0.0037, czyli losowy klasyfikator dawałby
AUC-PR ≈ 0.0037). Krzywa praktycznie pokrywa się z linią bazową na całej
długości — model **nie ma żadnej zdolności rozróżniającej** które
(kontekst, akcja) skończą się kliknięciem.

To rozwiązuje zagadkę ze slajdu poprzedniego: **V_DM ≈ V\* nie jest dowodem,
że model trafnie przewiduje nagrodę dla każdej pary (s,a)**. To raczej efekt
**kalibracji do średniej** — model, niezdolny rozróżnić akcje, 'uczy się'
po prostu przewidywać wartość bliską globalnemu CTR (~0.37–0.38%) dla każdej
obserwacji. Skoro V_DM to średnia z tych predykcji po wszystkich akcjach,
a wszystkie predykcje ≈ globalna średnia — to V_DM z definicji wyjdzie blisko
V\*, niezależnie od tego, czy model 'rozumie' różnice między akcjami.

Dlaczego model nie może się nauczyć więcej? Bo nagroda jest zbyt rzadka
(0.38% CTR, imbalance 260:1) — sygnał jest zalany szumem. To nie jest błąd
implementacji, tylko fundamentalne ograniczenie danych."

**Czego to uczy / Cel slajdu:**
To jest **kluczowy moment intelektualny** prezentacji — uczy słuchacza
**krytycznego czytania wyników OPE**: sama bliskość V̂ ≈ V\* nie wystarcza,
trzeba też sprawdzić, *czy* model bazowy ma w ogóle moc dyskryminacyjną.
Jeśli nie ma — niska wariancja DM jest złudna: jeśli różne akcje faktycznie
mają różne wartości (a tak jest w realnym przypadku BTS), model tego po
prostu nie zobaczy, a DM da ten sam "spłaszczony" wynik bez ostrzeżenia.
To bezpośrednio uzasadnia, dlaczego w ogóle potrzebujemy IPS i DR (slajdy 8–12)
— metod, które nie polegają (wyłącznie) na trafności reward modelu.

**Kluczowe liczby:**
- AUC-PR = **0.0056** (baseline/losowy ≈ 0.0037)
- AUC-ROC ≈ 0.51 (≈ losowy, dla porównania)
- CTR (validation) ≈ 0.37%

---

## Slajd 7 — Słabość DM — ekstrapolacja i bias modelu nagrody

**Co jest na slajdzie:** Dwa wykresy — `06_shap_top20.png` (top 20 cech wg
SHAP) i `07_ood_risk_map.png` (mapa ryzyka OOD) + tekst: "OOD rate = 0.16%
obserwacji poza manifoldem", "Bias DM = −0.000285 (model zaniża nagrodę
systematycznie)".

**Co mówić:**
"Mamy już dwa obrazy reward modelu: dokładny w sensie V_DM≈V* (slajd 5), ale
praktycznie bez mocy dyskryminacyjnej (slajd 6). Ten slajd pokazuje trzecią
perspektywę — *na czym* model się opiera i *gdzie* zawodzi.

SHAP (po lewej) pokazuje, które cechy najbardziej wpływają na predykcje —
głównie cechy kontekstu użytkownika, mniej cechy samej akcji (co tłumaczy
brak dyskryminacji między akcjami ze slajdu 6).

Mapa OOD (po prawej) pokazuje obserwacje leżące daleko od danych treningowych
— **ekstrapolacja poza manifold**. OOD rate = 0.16% — niewielki odsetek, ale
te obserwacje mogą dominować błąd, bo model 'zgaduje' poza obszarem, którego
się nauczył.

Razem to tłumaczy, skąd bierze się **bias DM = −0.000285** — systematyczne,
niewielkie niedoszacowanie V* widoczne już na slajdzie 5."

**Czego to uczy / Cel slajdu:**
Domyka "rozdział o DM" trzecią diagnostyką (po wyniku i po AUC-PR):
interpretowalność (SHAP) i ryzyko ekstrapolacji (OOD). Pokazuje, że DM nie
jest "czarną skrzynką" — da się zdiagnozować *jak* i *gdzie* model może się
mylić, nawet jeśli (jak na slajdzie 6) jego ogólna moc dyskryminacyjna jest
niska.

**Kluczowe liczby:**
- OOD rate = **0.16%**
- Bias DM = **−0.000285**

---

## Slajd 8 — IPS — model propensity P(a|s) i kalibracja

**Co jest na slajdzie:** Krzywa kalibracji `08_calibration_curve.png` dla
modelu propensity (P(a|s)).

**Co mówić:**
"Przechodzimy do drugiej rodziny estymatorów — IPS (Inverse Propensity
Scoring). Tu *nie* modelujemy nagrody. Modelujemy coś innego: **jak bardzo
prawdopodobne było, że stara polityka wybrała akcję, którą faktycznie
zaobserwowaliśmy w danych** — czyli P(a|s), propensity score.

Trenujemy 80-klasowy XGBoost (jeden z 80 produktów = jedna klasa). Zanim
użyjemy go do ważenia obserwacji, sprawdzamy **kalibrację**: czy gdy model
mówi 'jestem pewny na 60%', to rzeczywiście w 60% przypadków ma rację? Krzywa
kalibracji na tym slajdzie pokazuje, jak blisko modelu jest do przekątnej
(idealna kalibracja)."

**Czego to uczy / Cel slajdu:**
Ustanawia **drugi filar** projektu (po reward modelu z DM) — model
propensity. Kalibracja jest tu kluczowa, bo IPS dzieli przez P(a|s) —
źle skalibrowany model propensity może dawać systematycznie zniekształcone
wagi, nawet jeśli "ranking" akcji jest w miarę poprawny.

---

## Slajd 9 — IPS — ważenie przez propensity scores

**Co jest na slajdzie:** Formuła V_IPS = E[r × π_new(a|s)/π_old(a|s)],
histogram wag IPS (`09_ips_weights_hist.png`) oraz tabela: model P(a|s) =
80-klasowy XGBoost, multiclass logloss ≈ 4.37, top-1 accuracy = 1.75%
(baseline 1.25%), ESS = 9877/10000 = 98.77%, V_IPS = 0.004423.

**Co mówić:**
"IPS przeważa **obserwowaną** nagrodę przez stosunek π_new(a|s)/π_old(a|s).
Intuicja: jeśli stara polityka rzadko wybierała akcję A w danym kontekście
(niskie π_old), a nowa polityka wybierałaby ją częściej (π_new), to ta
pojedyncza obserwacja dostaje **dużą wagę** — bo 'reprezentuje' wiele
hipotetycznych przyszłych obserwacji nowej polityki.

Dla naszej polityki ewaluowanej (uniform, π_new = 1/80 dla każdej akcji)
i polityki logującej (też w przybliżeniu uniform — Random) — wagi wychodzą
bliskie 1. Histogram po prawej to potwierdza: rozkład wag jest skupiony
wokół wartości 1.

ESS (Effective Sample Size) = 9877/10000 = **98.77%** — prawie wszystkie 10
tysięcy obserwacji 'liczą się' efektywnie, bo wagi są wyrównane. To daje
**V_IPS = 0.004423** — też blisko V* = 0.0038, ale z dużo szerszym CI niż
DM (zobaczymy to na slajdzie 13)."

**Czego to uczy / Cel slajdu:**
Wprowadza IPS jako **alternatywne, niezależne od reward modelu** podejście
— i pokazuje mechanikę ważenia w praktyce (histogram wag ≈ 1). ESS = 98.77%
to ważny punkt odniesienia: na slajdzie 11 zobaczymy, co się dzieje, gdy ESS
spada do 11%.

**Kluczowe liczby:**
- Model P(a|s): 80-klasowy XGBoost, multiclass logloss ≈ **4.37**
- Top-1 accuracy = **1.75%** (losowy baseline = 1.25%)
- ESS = **9877/10000 = 98.77%**
- V_IPS = **0.004423**

---

## Slajd 10 — Clipping — tradeoff bias-variance

**Co jest na slajdzie:** Wykres `10_bias_variance_clipping.png` — wpływ
progu obcięcia wag (clipping threshold λ) na bias i wariancję V_IPS.

**Co mówić:**
"Wagi IPS mogą w teorii urosnąć bardzo wysoko (gdy π_old → 0, a π_new > 0).
Pojedyncza obserwacja z ekstremalną wagą może zdominować całą estymatę i
wywindować wariancję. Standardowym remedium jest **clipping** — obcinanie
wag do maksymalnej wartości λ.

Wykres pokazuje klasyczny tradeoff: przy małych λ obcinamy dużo wag — rośnie
bias (tracimy informację o 'rzadkich' obserwacjach), ale spada wariancja.
Przy dużych λ — odwrotnie. Dla tego datasetu jest dodatkowo **klif**: przy
λ > 0.016 *wszystkie* wagi w OBD zostają wyzerowane (bo λ jest tu progiem
na pscore w mianowniku, a minimalny pscore w danych jest blisko tej wartości)
— estymator się zapada."

**Czego to uczy / Cel slajdu:**
Wprowadza **clipping jako narzędzie** kontroli wariancji IPS — i od razu
pokazuje jego ograniczenie (klif). To przygotowuje grunt pod slajd 11, gdzie
zobaczymy *dlaczego* kontrola wariancji jest aż tak istotna — gdy overlap
się psuje, wariancja eksploduje bez ostrzeżenia.

**Kluczowe liczby:**
- Klif clippingu przy λ > **0.016** (zeruje wszystkie wagi w OBD)

---

## Slajd 11 — Overlap violation — ESS 0.99 → 0.11, V_IPS skacze 5×

**Co jest na slajdzie:** Dwa histogramy wag IPS obok siebie
(`11_overlap_violation.png`): po lewej "Oryginalne wagi IPS" (max=1.32,
ESS ratio=0.988), po prawej "Wagi IPS po naruszeniu overlap (10% zaniżone)"
(max=124.4, ESS ratio=0.113, skala logarytmiczna).

**Co mówić:**
"Slajd 9 pokazał ESS = 98.77% — prawie idealny overlap, bo polityki Random
i nasza polityka ewaluowana są w praktyce takie same. Ale co by się stało,
gdyby polityka logująca miała 'martwe strefy' — kontekst-akcje, które
prawie nigdy nie były wybierane (niskie π_old)?

Symuluję to: dla 10% obserwacji sztucznie zaniżam pscore o rząd wielkości
(naruszenie założenia overlap). Histogram po lewej to baseline — wagi
skupione wokół 1, max = 1.32, ESS ratio = 0.988. Histogram po prawej (uwaga:
skala logarytmiczna na osi Y!) to po naruszeniu — pojawia się długi ogon wag
sięgających **124.4**, a ESS ratio spada do **0.113**.

Konsekwencja: efektywna próba kurczy się 9-krotnie (z 98.77% do 11.3% z 10 000
obserwacji), a **V_IPS skacze z 0.004423 do 0.021314 — czyli niemal 5×**,
mimo że prawdziwa wartość polityki się nie zmieniła. To pokazuje, jak
kruchy jest IPS, gdy złamane jest założenie overlap."

**Czego to uczy / Cel slajdu:**
To jest **najważniejszy slajd o słabości IPS** — pokazuje namacalnie, że
"nieobciążoność" IPS jest okupiona kruchością: mała zmiana w ogonie
rozkładu propensity (10% obserwacji) może wywrócić wynik 5-krotnie. To
bezpośrednio motywuje **Doubly Robust** (slajd 12) jako metodę, która ma
dodatkową "siatkę bezpieczeństwa" w postaci reward modelu.

**Kluczowe liczby:**
- ESS ratio: **0.988 → 0.113** (przy zaniżeniu pscore dla 10% obserwacji)
- max waga: **1.32 → 124.4**
- V_IPS: **0.004423 → 0.021314** (≈ **5×**)

---

## Slajd 12 — Doubly Robust — state of the art w OPE

**Co jest na slajdzie:** Wykres `14_dr_robustness.png` + formuła
V_DR = V_IPS + E[(r−f(s,a))×π_new/π_old] + tabela trzech scenariuszy:
Baseline (oba modele OK) → 0.004223 ≈ V*; Zły PS model → 0.003818 ≈ V* ✅;
Zły reward model (stały 0.5) → −0.024 ❌.

**Co mówić:**
"Doubly Robust łączy DM i IPS: bierze IPS jako bazę i dodaje korektę z DM
dla reszty (różnicy między obserwowaną a przewidywaną nagrodą). Kluczowa
własność, która daje nazwę metodzie: **wynik jest poprawny, gdy choć JEDEN**
z dwóch modeli (propensity LUB reward) jest poprawny. Oba muszą zawieść
jednocześnie, żeby DR dał zły wynik.

Pokazuję trzy scenariusze:
- **Baseline** (oba modele takie, jak na poprzednich slajdach): V_DR =
  0.004223 ≈ V\*.
- **Zły model propensity** (zastępuję go losowymi pscores): V_DR =
  **0.003818 ≈ V\*** ✅ — DR 'ratuje się' dzięki temu, że reward model
  (mimo niskiego AUC-PR, jak widzieliśmy na slajdzie 6) wciąż wnosi
  użyteczną korektę średniej.
- **Katastrofalnie zły reward model** (stała predykcja 0.5, czyli 50% CTR
  zamiast prawdziwych ~0.4%): V_DR = **−0.024** ❌ — DR *nie* jest w stanie
  tego naprawić. To 100× za duża wartość bezwzględna i wynik wychodzi
  ujemny (co samo w sobie jest niemożliwym wynikiem dla CTR — czytelny
  sygnał alarmowy)."

**Czego to uczy / Cel slajdu:**
Pokazuje **granice gwarancji** DR — to nie jest "magiczna" metoda, która
zawsze działa, tylko metoda z konkretnym, sprawdzalnym warunkiem
("przynajmniej jeden model wystarczająco dobry"). Eksperyment B (reward
model = stała 0.5) jest też ważnym kontrastem do slajdu 6: tam reward model
miał AUC-PR≈0.0056 i *mimo to* DR działał (scenariusz "zły PS"). Różnica
między "słabym, ale niezdegenerowanym" modelem (slajd 6) a "katastrofalnie
złym, stałym" modelem (tu) jest jakościowa — DR toleruje pierwsze, nie
toleruje drugiego.

**Kluczowe liczby:**
- Baseline: V_DR = **0.004223** ≈ V*
- Zły PS model: V_DR = **0.003818** ≈ V* ✅
- Zły reward model (stały 0.5): V_DR = **−0.024** ❌

---

## Slajd 13 — Wyniki — DM vs IPS vs SNIPS vs DR na Open Bandit Dataset

**Co jest na slajdzie:** Wykres `12_unified_benchmark_ci.png` (słupki V̂ z CI
dla DM/IPS/SNIPS/DR + linia V* = 0.0038) + tabela: DM=0.003515
[0.003506,0.003525] (width 0.000019), IPS=0.004423 [0.003342,0.005567]
(width 0.002226), SNIPS=0.004184 [0.003161,0.005267] (width 0.002106),
DR=0.004223 [0.003140,0.005367] (width 0.002227). Adnotacja: "V*=0.0038 ·
Wszystkie estymatory zawierają V* w CI · IPS CI 117× szersze niż DM".

**Co mówić:**
"To jest zbiorcze porównanie wszystkich czterech estymatorów na tym samym
zbiorze danych, z bootstrapowanymi 95% CI (n=200 resampli). Dochodzi tu
**SNIPS** (Self-Normalized IPS) — wariant IPS, który normalizuje wagi tak,
by sumowały się do 1 zamiast do n; to lekko zmniejsza wariancję kosztem
niewielkiego biasu.

Wszystkie cztery estymatory **zawierają V\* = 0.0038 w swoim 95% CI** — więc
żaden z nich nie jest 'błędny' w sensie statystycznym. Różnica jest w
**szerokości CI**: DM ma CI szerokości 0.000019, IPS — 0.002226, czyli
**117× szersze**. SNIPS i DR są pomiędzy, bliżej IPS.

Pamiętając slajdy 5–6: DM jest 'punktowo' najdokładniejszy, ale ta dokładność
opiera się na reward modelu o AUC-PR=0.0056. IPS/SNIPS/DR nie mają tego
założenia — płacą za to szerszym CI."

**Czego to uczy / Cel slajdu:**
To jest **podsumowanie liczbowe** — zestawia wszystko, co było pokazane
osobno na slajdach 5, 9, 12, w jednej tabeli/wykresie. Przygotowuje grunt
pod slajd 14, który wyjaśni *skąd* bierze się ta różnica w szerokości CI
(dekompozycja Bias²/Variance).

**Kluczowe liczby:**
- V* = **0.0038**
- CI width: DM = **0.000019**, IPS = **0.002226** (117× szersze niż DM)
- Wszystkie 4 estymatory: V* ∈ 95% CI

---

## Slajd 14 — MSE = Bias² + Variance — kluczowy tradeoff w OPE (GŁÓWNY)

**Co jest na slajdzie:** Wykres `13_bias_variance_decomposition.png` +
tabela: DM (Bias²=8.1e-8, Var≈0, MSE=8.1e-8 ✓), IPS (Bias²=3.5e-7,
Var=4.5e-7, MSE=7.7e-7), SNIPS (Bias²=1.2e-7, Var=3.8e-7, MSE=5.0e-7).

**Co mówić:**
"To jest **centralny wynik całego projektu** — dekompozycja błędu MSE na
dwie składowe: Bias² i Variance, MSE = Bias² + Variance.

- **DM**: MSE zdominowane przez Bias² (8.1e-8), wariancja praktycznie zero.
  Każde uruchomienie bootstrapu daje niemal ten sam wynik — ale ten wynik
  ma wbudowany systematyczny błąd z reward modelu (slajdy 5–7), który
  *nie zmaleje* przy większej próbie.
- **IPS**: niemal nieobciążony (Bias² mały — 3.5e-7), ale Variance = 4.5e-7
  dominuje MSE. Wyniki 'skaczą' między uruchomieniami (i — jak slajd 11
  pokazał — mogą eksplodować przy naruszeniu overlap).
- **SNIPS**: kompromis — niższy bias *i* niższa wariancja niż IPS (redukcja
  wariancji o ok. 15% w tym datasecie), ale w OBD różnica nie jest
  dramatyczna, bo wagi i tak są bliskie 1.

Wniosek: **nie ma jednego najlepszego estymatora w sensie absolutnym** —
wybór zależy od tego, czy bardziej ufamy modelowi nagrody (→ DM), czy
modelowi propensity i mamy dobry overlap (→ IPS/SNIPS), czy chcemy
'ubezpieczenia' obu (→ DR, slajd 12)."

**Czego to uczy / Cel slajdu:**
Spaja w jedną ramę pojęciową wszystko, co było pokazane na slajdach 5–13:
niska wariancja DM (slajd 5) + niska moc dyskryminacyjna reward modelu
(slajd 6) + bias z OOD (slajd 7) = **wysoki, systematyczny Bias²**. Wysoka
ESS i wagi ≈1 dla IPS (slajd 9) + kruchość przy naruszeniu overlap (slajd 11)
= **wysoka Variance**. To jest slajd, na którym słuchacz powinien poczuć
"aha — to wszystko się spina".

**Kluczowe liczby:**
- DM: MSE = **8.1e-8** (10× niższe niż IPS) — ale to Bias², nie "dokładność"
- IPS: Variance = **4.5e-7** — dominująca składowa
- SNIPS redukuje wariancję IPS o ok. **15%**

---

## Slajd 15 — Sensitivity Analysis — czy wynik jest kauzalny czy przypadkowy?

**Co jest na slajdzie:** Wykres `15_sensitivity_refutations.png` + tabela
3 testów refutacji DoWhy: Random Common Cause (ATE=−0.003940, p=0.392 ✅),
Placebo Treatment (ATE=−0.000229 ✅), Data Subset 80% (ATE=−0.003936,
p=0.485 ✅). Original ATE = −0.003924.

**Co mówić:**
"Do tej pory mierzyliśmy *dokładność* estymatorów względem znanego V*. Ale
w realnym przypadku (StatsBomb, slajdy 16–17) nie mamy ground truth — musimy
więc zadać inne pytanie: **czy odkryty efekt jest w ogóle przyczynowy, czy
to artefakt danych/modelu?**

Tu wchodzi DoWhy i trzy standardowe testy refutacji, na przykładzie ATE
(average treatment effect) akcji 0 vs reszta = **−0.003924** (akcja 0 nie
zwiększa CTR względem innych akcji):

1. **Random Common Cause** — dodaję do modelu losową, niezwiązaną zmienną.
   Jeśli wynik jest prawdziwie przyczynowy, ATE powinno pozostać stabilne.
   Wynik: ATE = −0.003940, p = 0.392 ✅ — stabilne.
2. **Placebo Treatment** — zastępuję prawdziwy treatment losowym szumem.
   Jeśli efekt był 'prawdziwy', powinien zniknąć dla placebo. Wynik:
   ATE = −0.000229 (≈ 0) ✅ — efekt rzeczywiście znika.
3. **Data Subset (80%)** — uruchamiam całą analizę na losowym 80% danych.
   Wynik powinien być zbliżony do oryginalnego. ATE = −0.003936, p = 0.485 ✅
   — stabilne.

Wszystkie trzy testy przechodzą — model kauzalny jest odporny."

**Czego to uczy / Cel slajdu:**
Wprowadza **drugą warstwę walidacji**, niezależną od porównania z V*
(które jest dostępne tylko dzięki specyfice OBD). Te same trzy testy
refutacji są stosowane później w pilocie StatsBomb (slajdy 16–17), gdzie
*nie ma* ground truth — więc to jest most łączący część "z ground truth"
(OBD) z częścią "bez ground truth" (piłka nożna).

**Kluczowe liczby:**
- Original ATE = **−0.003924**
- Random Common Cause: p = **0.392** (>0.05, stabilne)
- Placebo Treatment: ATE → **−0.000229** (≈ 0, efekt znika)
- Data Subset (80%): p = **0.485** (stabilne)

---

## Slajd 16 — Pilot sportowy — OPE na danych piłkarskich (La Liga 2015/16)

**Co jest na slajdzie:** Wykres EDA `16_statsbomb_eda.png` (StatsBomb
La Liga 2015/16, dane 360°).

**Co mówić:**
"Ostatni krok to sprawdzenie, czy cały pipeline OPE — DM, IPS, SNIPS, DR,
testy refutacji — działa poza e-commerce, na zupełnie innej domenie.
Używam StatsBomb Open Data: 20 meczów La Liga 2015/16, z danymi 360°
(pozycje wszystkich zawodników na boisku w momencie podania).

Definiuję **treatment** jako *progressive pass* — podanie, które przesuwa
piłkę o więcej niż 10 metrów w kierunku bramki przeciwnika. **Outcome** to
binarna flaga: czy ta sekwencja zakończyła się strzałem lub asystą.

Wykres EDA pokazuje: 37% wszystkich podań to progressive passes, a ich
częstość mocno zależy od pozycji zawodnika — skrzydłowi i napastnicy grają
progresywnie znacznie częściej niż obrońcy. To jest właśnie **bias selekcji**
ze slajdu 2, tym razem w danych sportowych: pozycja zawodnika determinuje
zarówno prawdopodobieństwo 'treatmentu', jak i prawdopodobieństwo 'outcome'."

**Czego to uczy / Cel slajdu:**
Domyka pętlę ze slajdem 2 ("to samo dzieje się w piłce nożnej") — pokazuje
konkretną definicję treatment/outcome dla domeny sportowej i przygotowuje
dane wejściowe do wyników OPE na slajdzie 17.

**Kluczowe liczby:**
- 20 meczów, **19 371 podań**, 67 707 eventów
- Treatment (progressive pass): **37%** wszystkich podań
- Outcome (shot/goal assist): **1.84%** nagród
- Evaluation policy: aggressive — **50%** progressive passes

---

## Slajd 17 — Wyniki — Czy aggressive policy byłaby lepsza dla La Liga?

**Co jest na slajdzie:** Wykres `17_statsbomb_ope_results.png` + tabela
estymat (eval: 50% progressive): DM=1.57%, IPS=2.05%, SNIPS=2.13%,
DR=1.79% [1.31%, 2.27%]. Adnotacja: "Naive CTR = 1.93% · ESS = 0.686 ·
Aggressive policy nie jest jednoznacznie lepsza. Potrzeba ≥100 meczów dla
wąskich CI."

**Co mówić:**
"Pytanie brzmi: gdyby przez całą ligę narzucić strategię 'aggressive' —
50% progressive passes zamiast obserwowanych 37% — czy powstałoby więcej
sytuacji strzeleckich?

Stosuję dokładnie te same cztery estymatory co dla OBD:
- DM = 1.57% — najbardziej konserwatywna estymata,
- IPS = 2.05%, SNIPS = 2.13% — wyższe, ale z szerokimi CI,
- DR = 1.79% [1.31%, 2.27%] — pomiędzy, z przedziałem ufności.

Naive CTR (bez żadnej korekty) = 1.93%. DR sugeruje wynik porównywalny do
naive — czyli polityka 'aggressive' **nie jest jednoznacznie lepsza** od
obserwowanej.

ESS = 0.686 — zauważalnie niższe niż 0.989 w OBD (slajd 9). To dlatego, że
piłkarze mają silne, pozycyjnie uwarunkowane preferencje co do progressive
passes — gorszy overlap niż przy (w przybliżeniu) losowej polityce w OBD."

**Czego to uczy / Cel slajdu:**
Pokazuje **przeniesienie metodologii** z domeny o znanym ground truth (OBD)
do domeny bez ground truth (piłka) — i uczciwie komunikuje, że na 20 meczach
wyniki nie pozwalają na mocne wnioski (szerokie CI). To jest dowód na to,
że pipeline *działa technicznie* (wszystkie 4 estymatory + ESS + CI liczą
się poprawnie na nowych danych), ale wnioski merytoryczne wymagają więcej
danych — co prowadzi do "dalszych kroków" na ostatnim slajdzie.

**Kluczowe liczby:**
- V_DR = **1.79%** [1.31%, 2.27%] vs naive CTR = **1.93%**
- V_SNIPS = **2.13%** (najwyższa estymata)
- ESS = **0.686** (vs 0.989 w OBD)
- Potrzeba ≥ **100 meczów** dla wąskich CI

---

## Slajd 18 — Wnioski — kiedy używać DM, IPS, DR?

**Co jest na slajdzie:** Tabela porównawcza trzech metod (PRO/CON):
Direct Method (PRO: najniższe MSE, wąskie CI; CON: bias systematyczny przy
OOD), IPS/SNIPS (PRO: nieobciążony estymator; CON: wysoka wariancja, wymaga
ESS > 0.3), Doubly Robust (PRO: działa gdy choć jeden model OK; CON: zawiedzie
tylko gdy oba modele złe). Plus "Dalsze kroki": pełny StatsBomb (38 kolejek
× 5 sezonów) · continuous treatment · Q-function dla RL · v1.0.

**Co mówić:**
"Podsumowując — OPE to nie jeden algorytm, tylko **rodzina estymatorów
o różnych profilach błędu**, i wybór zależy od tego, czemu ufamy:

- **Direct Method** — gdy ufasz modelowi nagrody i masz dużo danych. Daje
  najniższe MSE i najwęższe CI (slajdy 5, 13, 14). Ale uwaga — jak pokazały
  slajdy 6–7, niskie MSE może współistnieć z bardzo niską mocą
  dyskryminacyjną modelu (AUC-PR=0.0056) i systematycznym biasem przy
  ekstrapolacji (OOD).
- **IPS / SNIPS** — gdy masz dobry model propensity *i* spełnione jest
  założenie overlap. Nieobciążony, ale wymaga monitorowania ESS — slajd 11
  pokazał, że przy ESS spadającym do 0.11, wynik może skoczyć 5×. Reguła
  kciuka: ESS > 0.3.
- **Doubly Robust** — domyślny wybór w produkcji. Wymaga wytrenowania dwóch
  modeli (więcej wariancji przy małych danych), ale gwarancja "wystarczy,
  by jeden model był dobry" (slajd 12) jest silna w praktyce — oba modele
  rzadko zawodzą jednocześnie.

Ograniczenia tego projektu: w OBD wagi IPS ≈ 1 (polityki podobne), więc nie
widzieliśmy 'naturalnego' tradeoffu bias-variance — dlatego pokazałem go
syntetycznie (slajd 11). StatsBomb to na razie 20 meczów — za mało na mocne
wnioski merytoryczne, ale wystarczająco, by zwalidować cały pipeline.

Dalsze kroki: pełny dataset StatsBomb (38 kolejek × 5 sezonów), continuous
treatment (stopień 'progresywności' podania zamiast progu binarnego),
integracja z Q-function dla RL, i — w dłuższej perspektywie — projekt
badawczy z wykorzystaniem tego frameworku do oceny taktyki piłkarskiej."

**Czego to uczy / Cel slajdu:**
Zamyka prezentację praktyczną "ściągawką decyzyjną" — slajd, do którego
słuchacz może wrócić, gdy będzie musiał wybrać estymator do własnego
problemu. Świadomie odsyła do konkretnych wcześniejszych slajdów (5–7, 11,
12), żeby decyzja była oparta na *zaobserwowanych* w tej prezentacji
zjawiskach, a nie na ogólnikach z literatury.

---

## Appendix — szybkie odpowiedzi na pytania

| Pytanie | Odpowiedź |
|---|---|
| Ile danych? | OBD: 1.3M obs (small set: 10K), StatsBomb: 20 meczów / 19 371 podań |
| Ground truth V*? | 0.0038 (naive CTR polityki random ≈ uniform eval policy) |
| AUC-PR reward modelu? | **0.0056** (≈ baseline 0.0037, brak mocy dyskryminacyjnej) |
| Dlaczego V_DM ≈ V* mimo niskiego AUC-PR? | Model kalibruje się do średniej CTR — V_DM to średnia predykcji, więc zbiega do globalnego CTR niezależnie od trafności per-akcja |
| Najlepszy estymator wg MSE? | DM: MSE = 8.1e-8 (ale to głównie Bias², nie precyzja per-akcja) |
| ESS w OBD (baseline)? | 98.77% |
| ESS po naruszeniu overlap (10% pscore)? | 11.3% (V_IPS skacze ~5×: 0.004423 → 0.021314) |
| ESS w StatsBomb? | 68.6% |
| DR kiedy działa? | Gdy choć jeden model (PS lub reward) jest "wystarczająco dobry" |
| DR kiedy nie działa? | Gdy reward model jest stały (0.5) → DR = −0.024 |
| Notebooki kluczowe? | `09_overlap_ess`, `10_bias_variance` (T8 — finalny model), `11_doubly_robust`, `13_statsbomb_pilot` |
| Reprodukcja? | `uv sync && uv run jupyter lab` |
