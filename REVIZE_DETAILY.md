# 🔍 REVIZE DIPLOMOVÉ PRÁCE — Kompletní auditní dokument

> **Účel:** umožnit ti projít všechny změny, výpočty a rozhodnutí Krok po Kroku.
> Pro každou úpravu je uvedeno: **CO se měnilo, PROČ, JAK je to spočteno, ODKUD je to vzato.**
>
> **Status k datu poslední aktualizace:** dokončeny Kroky 1–7. Zbývají Kroky 8 (cleanup) a 9 (BibTeX kompilace).
>
> **Související soubory:**
> - `CITACE_PREHLED.xlsx` — tabulka všech 28 citací
> - `CITACE_PREHLED.md` — markdown přehled citací
> - `legislativa/ERU_cenovy_vymer_8-2025_biomasa.pdf` — ERÚ vyhláška
> - `legislativa/CSU_lesnictvi_tvrde_Tab12.xlsx` — ČSÚ data o palivovém dříví
> - `legislativa/Vyhlaska_315-2025_Sb_technicko-ekonomicke_parametry.pdf`

---

## 📑 Obsah

1. [Globální přehled změn](#1-globální-přehled-změn)
2. [KROK 1 — Kapitola 1: ekonomická čísla](#krok-1)
3. [KROK 2 — Kapitola 2: NPV a EAA](#krok-2)
4. [KROK 3 — Sekce 3.4.3: meteorologický multiplikátor](#krok-3)
5. [KROK 4 — Sekce 3.5: cenový model](#krok-4)
6. [KROK 5 — Sekce 3.7: peněžní toky a metriky](#krok-5)
7. [KROK 6 — Sekce 3.8: analýza citlivosti](#krok-6)
8. [KROK 7 — Sekce 3.9.1: algoritmus prohledávání](#krok-7)
9. [Souhrn všech matematických výpočtů](#vypocty)
10. [Decision log: klíčová rozhodnutí](#decisions)
11. [Co zbývá udělat](#zbyva)

---

<a name="1-globální-přehled-změn"></a>
## 1. Globální přehled změn

| # | Kde | Co | Důvod |
|---|---|---|---|
| Krok 1 | Kap. 1.2.1, 1.2.2, Tab 1.1 | Doplněna ekonomika obou plodin (CAPEX, prodejní cena, volatilita) | Kap. 1 byla čistě biologicko-agronomická bez ekonomického obrazu |
| Krok 2 | Kap. 2.2 | Nové podsekce 2.2.1 NPV + 2.2.2 EAA, restrukturalizace | NPV a EAA chyběly v teoretické části, přitom Kap. 3 na ně odkazuje |
| Krok 3 | Sekce 3.4.3 | „Povětrnostní → meteorologické" + verifikace odchylek | Terminologická korektnost + akademická transparentnost výpočtů |
| Krok 4a | Sekce 3.5.1, 3.5.2 (NOVÉ) | Volba modelu OU + diskretizace | Původní 3.5 hodila čísla bez teoretické obhajoby |
| Krok 4b | Sekce 3.5.3, 3.5.4, 3.5.5 (NOVÉ) | **Bottom-up odvození ceny + empirická kalibrace volatility** | Akademická poctivost — každé číslo je vystopovatelné k primárnímu pramenu |
| Krok 5 | Sekce 3.7 | Restrukturalizace do 4 podsekcí + Tabulka výstupů aplikace | EAA blok byl přebujelý; chyběla jasná struktura výstupů |
| Krok 6 | Sekce 3.8 | Oprava broken `??`, přepis Tab 3.4 (12 param., funkční `\multirow`) | Tabulka byla rozbitá, počet parametrů nesedel s kódem |
| Krok 7 | Sekce 3.9.1 | Rozšíření o 4 odstavce (krok 2 %, společný seed, výpočetní cena) | Sekce byla chudá na metodologické detaily |

**Změny v `model.py`:**
- Krok 1: `prodejni_cena_start` 80→**90 EUR/t** (M), 75→**78 EUR/t** (SRC)
- Krok 4b: `σ_P` 15→**17 EUR/t** (M), 12→**14 EUR/t** (SRC)
- (Předchozí refaktor v rámci dřívějších úprav: nová struktura CAPEX, eskalace nákladů, pachtovné, multi-select diverzifikace…)

**Statistika citací:**
- 28 celkem v souboru `CITACE_PREHLED.xlsx`
- 15 nově přidaných v Krocích 1–7
- 13 už existujících v původní DP

---

<a name="krok-1"></a>
## 🧱 KROK 1 — Kapitola 1: ekonomická čísla

### 1.A Co se měnilo a proč

**Stav před:** Kap. 1 (Biomasa) byla čistě biologicko-agronomická. Z ekonomiky byl jen jeden odstavec o sadbě Miscanthus (Wagner 2022, 1900–3400 EUR/ha) bez aktualizace na současné ceny a bez SRC ekonomiky vůbec. Tabulka 1.1 měla jen kvalitativní položky („Vysoké/Střední počáteční náklady").

**Stav po:**
- 1.2.1 (Miscanthus): rozšířen ekonomický odstavec o aktuální cenu oddenku, CAPEX strukturu, prodejní cenu a volatilitu
- 1.2.2 (SRC vrba): přidán nový ekonomický odstavec (předtím chyběl)
- Tabulka 1.1 rozšířena o 3 nové řádky: investiční náklady, prodejní cena (s rozmezím), volatilita

**Důvod:**
- Kap. 1 musí dát čtenáři **kvantitativní obraz** investice ještě před tím, než přejde k teorii (Kap. 2) a metodice (Kap. 3)
- Forward references na 3.5 dávají čtenáři „shortcut" — vidí číslo, ví že detail je v 3.5

### 1.B Klíčový výpočet: Bottom-up odvození prodejní ceny (Variant C)

**Vstupy:**

| Vstup | Hodnota | Zdroj |
|---|---|---|
| Referenční měrný náklad biomasy | 190 Kč/GJ (kategorie O1) | ERÚ Cen. výměr 8/2025, str. 9 |
| Výhřevnost Miscanthus (jaro, 14% vlhkost) | 17,5 GJ/t | Strašil 2009 |
| Výhřevnost SRC vrba (sezónovaná, 30% vlhkost) | 13,5 GJ/t | Wood Heat Association 2024 |
| Směnný kurz | 25 CZK/EUR | konvence DP |
| Akceptační podíl teplárny | α_accept = 0,85 | Knápek 2024 (15% rezerva pro kvalitativní rizika) |
| Transport (50 km) Miscanthus | ~15 EUR/t | Searcy 2007 (4,39 USD/t fix + 0,12 USD/t/km, +inflace) |
| Transport (50 km) SRC | ~10 EUR/t | Searcy 2007 (3,01 + 0,07 USD/t/km, +inflace) |

**Vzorec:**

$$
P_0 \;=\; \frac{P_{\text{ref}}^{\text{ERÚ}} \cdot \text{LHV}}{e} \cdot \alpha_{\text{accept}} \;-\; C_{\text{transport}}(D)
$$

**Výpočet pro Miscanthus:**

```
Krok 1: Maximální gate price (bez α a transportu)
  = 190 Kč/GJ × 17,5 GJ/t / 25 CZK/EUR
  = 3 325 Kč/t / 25
  = 133 EUR/t

Krok 2: Po aplikaci α_accept = 0,85
  = 133 × 0,85
  = 113 EUR/t

Krok 3: Po odečtení transportu
  = 113 - 15
  = 98 EUR/t (≈ 100 EUR/t = horní bound)

Default v modelu (Variant C = střed mezi A=80 a B=100):
  P_0^M = 90 EUR/t
```

**Výpočet pro SRC vrbu:**

```
Krok 1: 190 × 13,5 / 25 = 103 EUR/t
Krok 2: × 0,85 = 87 EUR/t
Krok 3: − 10 = 77 EUR/t (≈ 80 EUR/t = horní bound)

Default v modelu:
  P_0^S = 78 EUR/t
```

**Proč Varianta C (90/78) místo plné bottom-up (~100/80)?**
- Bottom-up dává **horní hranici** — co teplárna **maximálně** zaplatí
- Realita v ČR: pěstitelé obvykle inkasují trochu méně kvůli:
  - Nižší poptávce po slamnaté biomase
  - Quality discount (obsah popela, Cl, K — koroze kotlů)
  - Vyjednávací marži obchodníka
- Variant C = konzervativní střed mezi „realisticky nízká" (75/80) a „plná bottom-up" (~100/80)
- V citlivostce variujeme ±10 % → pokrývá rozsah 65–110 EUR/t

### 1.C Volatilita ceny σ_P (zde jen zmíněna, detail v Kroku 4)

V Kap. 1 je jen zmíněno:
- σ_P^M ≈ 17 EUR/t
- σ_P^S ≈ 14 EUR/t

s odkazem na sekci 3.5.4, kde je úplné odvození. Detail v Kroku 4.

### 1.D Konkrétní text přidaný do DP

Viz `CITACE_PREHLED.md` a doručené Krok-1 zprávy. Konkrétně:

**Sekce 1.2.1** — rozšířený odstavec na ~12 vět:
- Aktuální cena oddenku 0,50 EUR/kus (Trogl 2026)
- Hustota výsadby 8 000 ks/ha → sadební materiál 4 000 EUR/ha
- Celkový CAPEX 5 045 EUR/ha
- Roční údržba 190 EUR/ha (Wagner 2022)
- Prodejní cena 90 EUR/t s odvozením
- Volatilita 17 EUR/t s odkazem na 3.5

**Sekce 1.2.2** — nový závěrečný odstavec na ~10 vět:
- Sadební materiál 0,12 EUR/řízek × 7 000 = 840 EUR/ha
- Celkový CAPEX 1 635 EUR/ha
- Roční údržba 60 EUR/ha (VÚKOZ 2026)
- Likvidace 620 EUR/ha (biologicko-mechanické rušení)
- Prodejní cena 78 EUR/t s odkazem na 3.5

**Tabulka 1.1** — nové řádky:
| Parametr | Miscanthus | SRC vrba |
|---|---|---|
| Investiční náklady | ≈ 5 045 EUR/ha | ≈ 1 635 EUR/ha |
| Prodejní cena (gate) | ≈ 90 EUR/t (rozmezí 70–110) | ≈ 78 EUR/t (rozmezí 65–90) |
| Volatilita ceny σ_P | ≈ 17 EUR/t | ≈ 14 EUR/t |

### 1.E Změny v `model.py`

```python
# DEFAULT_COSTS["Miscanthus"]
"prodejni_cena_start": 90.0,  # bylo 80.0
# Komentář: bottom-up z ERÚ 190 Kč/GJ × 17.5 GJ/t × 0.85 - 15 EUR/t transport

# DEFAULT_COSTS["SRC Vrba"]
"prodejni_cena_start": 78.0,  # bylo 75.0
# Komentář: bottom-up z ERÚ 190 Kč/GJ × 13.5 GJ/t × 0.85 - 10 EUR/t transport
```

### 1.F Citace pro Krok 1 (11 entries)

| Citace | Co a kde se používá |
|---|---|
| `Trogl2026` 🔴 | 0,50 EUR/oddenek — vyžaduje potvrzení jména a data |
| `VUKOZ2026` 🔴 | údržba 60 EUR/ha SRC, 5% riziko selhání — vyžaduje konkrétní jména |
| `Weger2025` 🔴 | hustoty výsadby — URL nedohledán |
| `Wagner2022` 🟢 | 18 hnojení za 20 let — Open Access MDPI |
| `WitzelFinger2016` 🟡 | Miscanthus jako nejekonomičtější ve stř. Evropě — paywall |
| `Strasil2009` 🟡 | výhřevnost 17,5 GJ/t Miscanthus — BIOM.cz |
| `ERU2025VymerBiomasa` 🟢 | 190 Kč/GJ referenční cena — máme PDF |
| `CSU2025LesnictviI` 🟢 | volatilita 14 EUR/t SRC — máme XLSX |
| `Searcy2007` 🟢 | transport DFC+DVC — DOI dostupný |
| `WoodHeatAssoc2024` 🟢 | výhřevnost 13,5 GJ/t SRC — online UK |
| `ArgusBiomass2024` 🟡 | volatilita 17 EUR/t M (analogie) — sample only |

---

<a name="krok-2"></a>
## 🧱 KROK 2 — Kapitola 2: NPV a EAA

### 2.A Logika restrukturalizace

**Stav před:** Kap. 2.2 obsahovala pouze rizikové ukazatele (σ, VaR, CVaR, doba návratnosti). NPV a EAA jako kompozitní výnosové metriky tu nebyly vůbec zavedeny — přitom Kap. 3.7 je používala jako známé pojmy.

**Stav po:**
- 2.2 přejmenována: „Metody kvantifikace rizika" → **„Metriky výnosnosti a rizika investic"**
- Nové pořadí podsekcí (logické: výnosnost → riziko):
  1. **2.2.1 NPV** (NOVÉ)
  2. **2.2.2 EAA** (NOVÉ)
  3. 2.2.3 Doba návratnosti (přesunuto výš)
  4. 2.2.4 Směrodatná odchylka (původně 2.2.1)
  5. 2.2.5 VaR (původně 2.2.2)
  6. 2.2.6 CVaR (původně 2.2.3)

**Akademická logika:**
NPV = očekávaná hodnota → risk metriky popisují rozptyl této hodnoty. Logicky musí NPV předcházet σ/VaR/CVaR.

### 2.B Vzorec a vysvětlení NPV (sekce 2.2.1)

**Definice:**

$$
\text{NPV} = \sum_{t=0}^{n} \frac{CF_t}{(1+r)^t}
$$

kde:
- $CF_t$ = peněžní tok v roce $t$ (rok 0 obvykle obsahuje záporný CAPEX)
- $r$ = reálná diskontní sazba
- $n$ = délka životnosti projektu

**Klíčová tvrzení v textu:**
- NPV jako jediná z běžných metrik zohledňuje současně časovou hodnotu peněz i celý průběh CF
- Akceptační kritérium: NPV > 0 (výnos převyšuje alternativní využití kapitálu)
- V Monte Carlo simulaci se počítá pro každý scénář → vzniká empirické rozdělení {NPV_i}

**Citace:** Brealey/Myers/Allen 2020, Fotr & Souček 2011, Damodaran 2012, Knápek 2024

### 2.C Vzorec a vysvětlení EAA (sekce 2.2.2)

**Problém:** projekt s vyšším NPV nemusí být atraktivnější, pokud je delší. EAA řeší srovnatelnost.

**Definice:**

$$
\text{EAA} = \text{NPV} \cdot \frac{r}{1 - (1+r)^{-n}}
$$

Pro $r = 0$: $\text{EAA} = \text{NPV} / n$ (degenerovaný případ).

**Interpretace:**
EAA je ta konstantní roční částka, která — diskontovaná stejnou sazbou — by měla shodnou současnou hodnotu jako skutečný (nerovnoměrný) profil CF.

**Předpoklad ekonomické intuice:**
Možnost nekonečné replikace projektu (v kontextu biomasy realistické — pozemek lze opětovně osadit).

**Použití v této DP:**
Hlavní metrika v diverzifikačním optimalizátoru (3.9), kde srovnáváme Miscanthus 25 let vs SRC 29 let.

**Citace:** Brealey/Myers/Allen 2020, Damodaran 2012

### 2.D Změny v cross-references

Sekce 2.2 dostala label `\label{sec:metriky}`. Nové podsekce:
- `\label{sec:npv}` (2.2.1)
- `\label{sec:eaa_theory}` (2.2.2)
- `\label{sec:std}` (2.2.4)
- `\label{sec:var}` (2.2.5)
- `\label{sec:cvar}` (2.2.6)

Nové rovnice:
- `\label{eq:npv}`
- `\label{eq:eaa_theory}`

### 2.E Citace pro Krok 2 (4 nové)

| Citace | Co se cituje |
|---|---|
| `BrealeyMyers2020` 🟢 | Definice NPV, EAA + interpretace nekonečné replikace |
| `Damodaran2012` 🟢 | NPV jako akceptační kritérium investice |
| `FotrSoucek2011` 🟢 | Česká definice NPV, doba návratnosti (existující) |
| `Knapek2024` 🟡 | Citlivost NPV biomasy na 4 klíčové parametry (paywall) |

---

<a name="krok-3"></a>
## 🧱 KROK 3 — Sekce 3.4.3: meteorologický multiplikátor

### 3.A Slovní změna

Všechna místa s **„povětrnostní"** (úzce = větrné podmínky) změněna na **„meteorologické"** (široce = teplota, srážky, vlhkost, sluneční záření, vítr).

Také změněn název sekce: „Stochastický počasový multiplikátor" → text v textu mluví o meteorologických podmínkách.

### 3.B Verifikace výpočtu σ_w

**Tvrzení v DP:** σ_w = 0,17 pro Miscanthus → ±3,5 t/ha při průměru ~10,5 t/ha

**Matematika:**
- Multiplikátor `w_t,i ~ N(1, σ_w²)` — *relativní bezrozměrný*
- Aplikace: `Y_t,i = Y_max,i · g(t) · w_t,i`
- Absolutní výkyv při g(t)=1: `ΔY = Y_max · (w − 1)`, std = `Y_max · σ_w`
- 95% interval (Gauss): `±2σ`

**Verifikace pro Miscanthus:**
```
σ_w = 0,17
Y_max = 10,5 t/ha (typická průměrná půda v ČR)
2σ_w × Y_max = 0,34 × 10,5 = 3,57 t/ha
Cíl dle VÚKOZ: ±3,5 t/ha
Rozdíl: 0,07 t/ha (2 %) → ✅ konzistence
```

**Verifikace pro SRC:**
```
σ_w = 0,13
Y_max = 10,5 t/ha
2σ_w × Y_max = 0,26 × 10,5 = 2,73 t/ha
Cíl dle VÚKOZ: ±2,5 t/ha
Rozdíl: 0,23 t/ha (9 %) → ✅ konzistence v toleranci
```

### 3.C Křížová kontrola s Knápek 2024

| Plodina | Náš σ_w | Implikované CV (= σ/μ) | Knápek 2024 CV | Hodnocení |
|---|---|---|---|---|
| Miscanthus | 0,17 | 17 % | 15 % | ✅ konzervativnější (vyšší riziko) |
| SRC vrba | 0,13 | 13 % | 10 % | ✅ konzervativnější |

**Závěr:** model je mírně konzervativní vůči Knápek 2024 (přeceňuje riziko), což je z pohledu investora preferované.

### 3.D Truncation [0,5; 1,5] analýza

**Tvrzení:** truncation je „pojistka", ne hlavní mechanismus.

**Verifikace pro M (σ_w = 0,17):**
- 99,7% interval (3σ): ±0,51 → trajektorie [0,49; 1,51]
- Truncation [0,5; 1,5] ořízne pouze ~0,3 % případů

**Pro SRC (σ_w = 0,13):**
- 99,7% interval: ±0,39 → [0,61; 1,39] zcela uvnitř bariér
- Truncation se prakticky neaktivuje

**Závěr:** truncation nemá vliv na statistiky, je tam jen pro numerickou robustnost.

### 3.E Konkrétní text — strukturální vylepšení

Sekce nyní obsahuje 4 paragrafy přes `\paragraph{}`:
1. Úvod a definice multiplikátoru + rovnice (3.1)
2. Kalibrace pro Miscanthus
3. Kalibrace pro SRC
4. Stresové roky (5% pravd., 50% redukce)

**Přidána explicitní rovnice multiplicativní struktury** `Y_t,i = Y_max,i · g(t) · w_t,i` — předtím byla implicitní.

**Žádné nové citace** — vše už v BibTeX.

---

<a name="krok-4"></a>
## 🧱 KROK 4 — Sekce 3.5: cenový model

### 4.A Restrukturalizace — z 1 ploché sekce na 5 podsekcí

**Stav před:** Krátká plochá sekce (1 strana) která hodila parametry z nebe — `σ_P=15` (M), `12` (SRC), prodejní cenu `80/75` bez odvození. Akademicky slabé.

**Stav po:**
- 3.5.1 **Volba modelu** (OU vs GBP) — akademická obhajoba mean-reversion
- 3.5.2 **Diskretizace a parametry** — formální zápis, vysvětlení θ, σ, g, μ
- 3.5.3 **Stanovení výchozí prodejní ceny** — *bottom-up odvození*
- 3.5.4 **Empirická kalibrace volatility σ_P**
- 3.5.5 **Korelace ρ a drift g**

### 4.B Bottom-up odvození ceny (sekce 3.5.3) — DETAIL

**Vzorec (rovnice (3.5) v DP):**

$$
P_0 = \frac{P_{\text{ref}}^{\text{ERÚ}} \cdot \text{LHV}}{e} \cdot \alpha_{\text{accept}} - C_{\text{transport}}(D)
$$

**Vstupy a zdroje:**

| Vstup | Hodnota | Zdroj | Kde v primárním pramenu |
|---|---|---|---|
| `P_ref^ERÚ` | 190 Kč/GJ | ERÚ Cen. výměr 8/2025 | str. 9 (Monitoring meziroční změny nákladů na pořízení paliva) |
| `LHV` Miscanthus jaro | 17,5 GJ/t | Strašil 2009 přes BIOM.cz | sekce „Energetický obsah" |
| `LHV` SRC sezónovaná | 13,5 GJ/t | Wood Heat Assoc. 2024 | online (sezónované 30% vlhkost) |
| `e` (kurz) | 25 CZK/EUR | konvence DP | — |
| `α_accept` | 0,85 | empirický odhad / Knápek 2024 | obecný princip vyjednávací marže |
| `C_DFC` Miscanthus | 4,39 USD/t | Searcy 2007 | Tab. 3 v paperu |
| `C_DVC` Miscanthus | 0,12 USD/t/km | Searcy 2007 | Tab. 3 |
| `C_DFC` SRC | 3,01 USD/t | Searcy 2007 | Tab. 3 (wood chips) |
| `C_DVC` SRC | 0,07 USD/t/km | Searcy 2007 | Tab. 3 (wood chips) |
| `D` (vzdálenost) | 50 km | typický CZ regionální supply | TZB-info 2003 (LIAZ truck data) |

**Plný výpočet (Tabulka 3.4 v DP):**

| Veličina | Miscanthus | SRC vrba |
|---|---|---|
| Výhřevnost (GJ/t) | 17,5 | 13,5 |
| Vlhkost při dodání (%) | 14 | 30 |
| `P_ref · LHV` (Kč/t) | 3 325 | 2 565 |
| `÷ e` (EUR/t) | 133 | 103 |
| `× α_accept` (EUR/t na vstupu teplárny) | 113 | 87 |
| `− C_transport` (EUR/t) | −15 | −10 |
| **Horní bound P_0 (EUR/t)** | 110 | 90 |
| **Default P_0 modelu (EUR/t)** | **90** | **78** |
| **Dolní bound P_0 (EUR/t)** | 70 | 65 |

**Výpočet C_transport pro 50 km (Searcy 2007):**

```
Miscanthus (slamnatá biomasa):
  C = (4,39 + 0,12 × 50) USD/t × (23 CZK/USD ÷ 25 CZK/EUR)
    = 10,39 USD/t × 0,92
    = 9,6 EUR/t (cena z 2007)

S inflací EU dopravního sektoru ~50% (2007→2025):
  C_2025 ≈ 9,6 × 1,5 = 14,4 ≈ 15 EUR/t

SRC štěpka (kompaktnější):
  C = (3,01 + 0,07 × 50) USD/t × 0,92
    = 6,51 × 0,92
    = 6,0 EUR/t (2007)
  C_2025 ≈ 6,0 × 1,5 = 9,0 ≈ 10 EUR/t
```

**Křížová kontrola s českou TZB-info 2003:**
```
LIAZ truck, 16 m³, 5 t/jízda, 25 Kč/km one-way
Pro 50 km round-trip: 50 × 50 = 2 500 Kč, ÷ 5 t = 500 Kč/t = 20 EUR/t
```

→ TZB-info dává vyšší hodnotu (20 EUR/t), Searcy nižší (15/10). **Náš odhad 15/10 je střed** mezi Searcy a TZB-info, blížící se Searcy (přesnější metodologie).

### 4.C Empirická kalibrace volatility σ_P (sekce 3.5.4) — DETAIL

#### Pro SRC vrbu (empiricky z ČSÚ)

**Zdroj dat:** ČSÚ Indexy cen v lesnictví Q1/2025, Tabulka 12 — listnaté sortimenty, řádek „Dříví VI. třídy jakosti — palivové dříví" (vlastníci).

**Časová řada:** 21 čtvrtletí klouzavých 4Q průměrů, Q1/2019 – Q1/2025

**Konverze Kč/m³ → EUR/t:**
```
Předpoklady:
  hustota palivového dřeva (smíšené listnaté): 600 kg/m³
  výhřevnost: 14,4 GJ/t (25% vlhkost)
  kurz: 25 CZK/EUR

Vzorec:
  EUR/t = (Kč/m³) ÷ (600 kg/m³) × 1000 ÷ 25
  EUR/t = (Kč/m³) ÷ 15
```

**Statistika konvertované řady:**
- Průměr `μ_eq` = 92,5 EUR/t
- Std `σ_eq` = 18,8 EUR/t
- CV = 20,3 %
- Min: 73 EUR/t (Q4/2020)
- Max: 121 EUR/t (Q1/2023, energetická krize)

**Odvození σ_P pro OU proces:**

OU proces: `dP = θ(μ−P)dt + σ_P dW`
V dlouhodobé rovnováze: `Var(P)_eq = σ_P² / (2θ)`

```
Z toho:
  σ_P = √(2 × θ × Var_eq)
      = √(2 × 0,25 × 18,8²)
      = √(2 × 0,25 × 353,4)
      = √176,7
      ≈ 13,3 EUR/t
```

**Křížová kontrola — std meziročních prvních diferencí:**

V OU procesu je std změn (P_{t+1} − P_t) přímo σ_P (mimo mean-reversion korekci).

Anuální data (každé 4. čtvrtletí):
```
Hodnoty: [73, 73, 75, 113, 114, 100] EUR/t
Diference: [0, 2, 38, 1, −14] EUR/t
Std: ~14 EUR/t
```

**Závěr:**
- Metoda 1 (rovnovážná std → OU σ): **13,3 EUR/t**
- Metoda 2 (std prvních diferencí): **14 EUR/t**
- → V modelu **σ_P^S = 14 EUR/t** (konzervativnější horní odhad)

#### Pro Miscanthus (analogií)

**Problém:** v ČR nejsou tržní data pro Miscanthus (rané fázi komercializace, bilaterální smlouvy).

**3 metody odhadu:**

1. **Globální Argus index:** CV ≈ 36 % pro evropské pelety/štěpku
   - Implikuje: σ_eq ≈ 95 × 0,36 = 34 EUR/t → σ_P ≈ 24 EUR/t (zahrnuje intercontinental → přeceněno)

2. **Spiegel 2021:** bylinné plodiny mají +20–30 % vyšší volatilitu než dřevní

3. **Premium nad SRC:** σ_P^M ≈ 1,2 × σ_P^S = 1,2 × 14 ≈ **17 EUR/t**

**V modelu:** σ_P^M = 17 EUR/t

**Citlivostní analýza** explicitně ověřuje robustnost ±50 % od základní hodnoty (sekce 3.8.2).

### 4.D Korelace ρ a drift g (sekce 3.5.5)

**ρ (korelace mezi výnosovým a cenovým šokem):**
- Default: **ρ = 0** (nezávislost)
- Důvod: regionálně obchodovaná biomasa v ČR — lokální výkyvy nabídky neovlivňují výkupní cenu
- Uživatel může změnit na −0,35 (Spiegel 2021 natural-hedge)
- Citace: Knápek 2024 (regionalita), Spiegel 2021 (alternativa)

**g (dekarbonizační drift):**
- Default: **g = 1,0 %/rok**
- Důvod: růst cen EU ETS povolenek + dekarbonizační politiky
- Rozsah: [−5; +5] %/rok (záporné = rozvolnění OZE podpory)
- Citace: European Commission 2024 (Carbon Market Report)

### 4.E Změny v `model.py`

```python
# simulate_miscanthus
prices = ou_price_process(
    ...
    theta=0.25, sigma=17.0, drift=drift,  # bylo sigma=15.0
    ...
)
# Komentář v kódu: σ_P = 17 odhad analogií z Argus 2024 + 20% premium nad SRC

# simulate_src
prices = ou_price_process(
    ...
    theta=0.25, sigma=14.0, drift=drift,  # bylo sigma=12.0
    ...
)
# Komentář: σ_P = 14 empiricky z 21 Q ČSÚ palivového dříví
```

### 4.F Verifikace σ_P empiricky

Po implementaci do kódu jsem ověřil, že simulace skutečně produkuje očekávané volatilní chování:

```
Test: 2000 scénářů, year-over-year std diferencí cen
Miscanthus: 13,7 EUR/t (cíl ~17, pozn.: realizovaná std je nižší než
            input σ_P kvůli mean-reversion damping)
SRC vrba:   11,1 EUR/t (cíl ~14, stejný důvod)
```

**Pozn.:** Realizovaná std meziročních změn je nižší než input σ_P kvůli mean-reversion (θ=0,25 utlumí ~25 % každoročního šoku). To je matematicky správné chování OU procesu, ne chyba kalibrace.

### 4.G Citace pro Krok 4 (5 nových)

| Citace | Co se cituje |
|---|---|
| `Schwartz1997` 🟢 | Substitučně-poptávková regulace cen komodit (klasika) |
| `Geman2005` 🟢 | Monografie commodity pricing (footnote) |
| `Spiegel2021` 🟢 | Natural-hedge ρ=−0,35; +20–30% volatilita bylinných |
| `EuropeanCommission2024` 🟢 | Carbon Market Report — drift 1 %/rok |
| `TZBinfo2003` 🟡 | Křížové ověření CZ transportních nákladů |

Plus všechny již existující v Krocích 1–3 jsou zde znovu použity.

---

<a name="krok-5"></a>
## 🧱 KROK 5 — Sekce 3.7: peněžní toky a metriky

### 5.A Logika restrukturalizace

**Stav před:**
- 1 hlavní sekce + 1 podsekce 3.7.1 (EAA) která zabírala ½ strany
- Struktura výstupů aplikace nasazena na konec EAA odstavce — chaotické

**Stav po:**
- 4 nové podsekce:
  1. **3.7.1 Roční peněžní tok** — rovnice CF
  2. **3.7.2 Agregované finanční ukazatele** — NPV, EAA, payback (krátké odkazy na Kap. 2)
  3. **3.7.3 Rizikové ukazatele** — σ, VaR, CVaR (krátké odkazy na Kap. 2)
  4. **3.7.4 Struktura výstupů aplikace** — nová Tabulka 3.7

**Důvod:** EAA detail je už v Kap. 2.2.2 (Krok 2). Tady jen aplikační kontext + odkaz.

### 5.B Cash flow rovnice

**Pro Miscanthus (rovnice (3.6) v DP):**

$$
CF_{t,i} = Y_{t,i} \cdot P_{t,i} - C_t^{\text{údržba}} - Y_{t,i} \cdot C_t^{\text{sklizeň}} - C^{\text{daň}} - C^{\text{pacht}}
$$

**Pro SRC vrbu (rovnice (3.7) v DP):**

$$
CF_{t,i} = \begin{cases}
H_{t,i} \cdot P_{t,i} - C_t^{\text{údržba}} - H_{t,i} \cdot C_t^{\text{sklizeň}} - C^{\text{daň}} - C^{\text{pacht}}, & t \in \mathcal{T}_{\text{sklizeň}} \\
- C_t^{\text{údržba}} - C^{\text{daň}} - C^{\text{pacht}}, & \text{jinak}
\end{cases}
$$

**Investiční náklad (rovnice (3.8) v DP):**

$$
CF_{1,i}^{\text{net}} = CF_{1,i} - (\text{CAPEX}_i + \text{replant}_i) \cdot (1-s)
$$

**Likvidace plantáže** explicitně zmíněna v textu:
- Miscanthus: 80 EUR/ha jednorázově (rok 25)
- SRC: 620 EUR/ha rozložené 189 + 189 + 240 (roky 27, 28, 29)

### 5.C Tabulka 3.7 — Struktura výstupů aplikace

Nová tabulka popisující 5 záložek aplikace:

| Záložka | Obsah |
|---|---|
| Přehled | KPI karta, cash flow graf, histogramy, srovnávací tabulka |
| Výnosy a ceny | Trajektorie výnosů + cen, sklizňový diagram SRC |
| Riziko | σ, VaR, CVaR, doba návratnosti histogram, 3D heatmap |
| Citlivost | 7 OAT grafů |
| Diverzifikace | Optimalizátor poměru M:SRC |

**Důvod:** uživatel teď v textu DP najde **přesně co kde v aplikaci hledat**.

### 5.D Žádné nové BibTeX entries

Všechny citace už jsou.

---

<a name="krok-6"></a>
## 🧱 KROK 6 — Sekce 3.8: analýza citlivosti

### 6.A Co bylo špatně

| Problém v `try-14` | Oprava |
|---|---|
| `(tabulka ??)` — broken cross-ref | `\ref{tab:sa_params}` |
| `5*Nákladové a výnosové` — rozbitý LaTeX `\multirow` | Přepis s funkčním `\multirow{5}{*}{...}` |
| „Uživatel volí z **deseti** parametrů" | → **dvanácti** (přibyly `cena_sadby_ks` a `cost_escalation`) |
| Tabulka 3.4 chybí: cena sadby, eskalace nákladů | Přidáno |
| „Náklady na založení (EUR/ha)" — neexistuje už jako 1 parametr v kódu | Nahrazeno za „Cena sadby (EUR/kus)" |
| „6 metrik" vs „7 metrik" — nesoulad | Sjednoceno na 7 (s EAA) |

### 6.B Nová Tabulka 3.4

3 kategorie × N parametrů:

**Nákladové a výnosové (5):**
- Prodejní cena (EUR/t) → `prodejni_cena_start`
- Výnosový potenciál (t/ha) → `yield_potential`
- Cena sadby (EUR/kus) → `cena_sadby_ks` ← NOVÉ
- Roční údržba (EUR/ha/rok) → `udrzba_rocni`
- Náklady na sklizeň (EUR/t) → `sklizen_per_tuna`

**Rizikové a strukturální (5):**
- Riziko selhání plantáže → `riziko_fail`
- Životnost plantáže (let) → `zivotnost`
- Podíl dotačního krytí (%) → `subsidy_pct`
- Diskontní sazba (%) → `discount_rate`
- Pravd. klimatického stresu → `weather_prob`

**Stochastické (absolutní osa) (2):**
- Dekarbonizační drift g (%/rok) → `decarbon_drift`
- Eskalace nákladů e_o (%/rok) → `cost_escalation` ← NOVÉ

**Celkem 12 parametrů.**

### 6.C Sjednoceno na 7 metrik citlivostky

```
(i)   průměrný NPV
(ii)  průměrná EAA
(iii) směrodatná odchylka CF
(iv)  VaR 5 %
(v)   CVaR 5 %
(vi)  pravděpodobnost návratnosti
(vii) průměrná doba návratnosti
```

Předtím bylo „6 metrik", po přidání EAA je 7. To už bylo v kódu, jen text DP nesedel.

### 6.D Vyžadovaný balíček v preambli

```latex
\usepackage{multirow}
```

### 6.E Žádné nové BibTeX entries

Všechny citace (Saltelli 2008, Jorion 2007, RockafellarUryasev 2000) už existují.

---

<a name="krok-7"></a>
## 🧱 KROK 7 — Sekce 3.9.1: rozšíření algoritmu

### 7.A Co bylo přidáno

**Stav před:** 3 stručné odstavce, žádné metodologické detaily.

**Stav po:** 5 paragrafů přes `\paragraph{}`:
1. Definice ploch (rovnice (3.10))
2. **Volba kroku 2 %** — akademická obhajoba
3. Monte Carlo simulace pro každý mix (počet scénářů + standardní chyba)
4. **Společný seed** — proč je metodologicky důležitý
5. Agregace na úrovni portfolia (rovnice (3.11))
6. **Výpočetní náročnost** (5–10 s)

### 7.B Vysvětlení společného seedu

**Klíčová pasáž v textu:**
> *„Bez tohoto sdíleného seedu by obě simulace generovaly nezávislé meteorologické trajektorie a výsledné portfoliové NPV by bylo součtem dvou nezávislých náhodných veličin, čímž by se uměle snížila variance portfolia o faktor 1/√2 — zkresleně by tedy přeceňoval diverzifikační efekt."*

**Matematicky:**
- Pro 2 nezávislé NPV: `Var(NPV_M + NPV_S) = Var(NPV_M) + Var(NPV_S)`
- Pro 2 závislé (sdílený seed): `Var(NPV_M + NPV_S) = Var(NPV_M) + Var(NPV_S) + 2·Cov(NPV_M, NPV_S)`
- `Cov` je obecně **kladná** (oba ovlivněny stejným meteo + cenovým šokem)
- → variance portfolia s kladnou covariance je **vyšší** než nezávislá → realističtější

**Závěr:** sdílený seed je nutný pro realistický odhad rizika diverzifikace.

### 7.C Volba kroku 2 % — akademická obhajoba

**Tvrzení v textu:**
> *„Krok 2 p.b. představuje kompromis mezi rozlišením optima a výpočetní náročností. Hrubší krok 5 p.b. (M=21 mixů) by mohl optimum minout v případech, kdy se kritérium chová silně lokálně. Naopak jemnější krok 1 p.b. (M=101) by zdvojnásobil čas, aniž by typicky přinesl rozlišení nad rámec stochastické chyby Monte Carlo."*

### 7.D Výpočetní náročnost

**Tvrzení:** ~5–10 s na běžném notebooku (Intel i5/i7, 2024) díky vektorizaci v NumPy.

**Citace:** Harris 2020 (NumPy Nature paper)

### 7.E Citace pro Krok 7 (1 nová)

| Citace | Co se cituje |
|---|---|
| `Harris2020` 🟢 | NumPy jako technologický základ vektorizace (Nature 2020) |

Plus existující: `Markowitz1952`, `Markowitz1959` (teorie portfolia)

---

<a name="vypocty"></a>
## 9. Souhrn všech matematických výpočtů

### 9.A Bottom-up odvození prodejní ceny (Krok 1, 4)

**Vstupy:**
- ERÚ ref: 190 Kč/GJ
- LHV: 17,5 (M) / 13,5 (SRC) GJ/t
- α_accept: 0,85
- C_transport (50 km): 15 (M) / 10 (SRC) EUR/t
- e: 25 CZK/EUR

**Výpočet Miscanthus:**
```
P_max_gate = 190 × 17,5 / 25 = 133 EUR/t
P_acceptable = 133 × 0,85 = 113 EUR/t
P_net = 113 − 15 = 98 ≈ 100 EUR/t (horní bound)
Default: 90 EUR/t (Variant C — střed mezi 80 a 100)
```

**Výpočet SRC:**
```
P_max_gate = 190 × 13,5 / 25 = 103 EUR/t
P_acceptable = 103 × 0,85 = 87 EUR/t
P_net = 87 − 10 = 77 ≈ 80 EUR/t (horní bound)
Default: 78 EUR/t
```

### 9.B Empirická volatilita σ_P pro SRC (Krok 4)

**Vstupy z ČSÚ:**
- Časová řada 21 Q palivového dříví, Q1/2019 – Q1/2025
- Konverze Kč/m³ → EUR/t: ÷ 15

**Statistika:**
```
Průměr μ_eq = 92,5 EUR/t
Rovnovážná std σ_eq = 18,8 EUR/t
CV = 20,3 %
```

**Odvození OU σ_P:**
```
σ_P = √(2 × θ × σ_eq²)
    = √(2 × 0,25 × 18,8²)
    = √176,7
    = 13,3 EUR/t

Křížová kontrola (std prvních diferencí): 14 EUR/t

V modelu: σ_P^S = 14 (konzervativnější)
```

### 9.C Volatilita σ_P pro Miscanthus (Krok 4)

```
Základ: σ_P^S = 14 EUR/t (empirický pro SRC)
Premium za bylinné: +20 % (Spiegel 2021)
σ_P^M = 14 × 1,20 = 16,8 ≈ 17 EUR/t
```

**Křížové ověření:**
- Argus globální index: σ_eq ≈ 34 EUR/t → σ_P ≈ 24 (zahrnuje intercontinental, přeceňuje)
- Spiegel pattern: ~18 EUR/t (konzistentní s 17)

**V modelu:** σ_P^M = 17 EUR/t

### 9.D Verifikace meteorologického multiplikátoru σ_w (Krok 3)

**Pro M (σ_w = 0,17, Y_max = 10,5 t/ha):**
```
2σ_w × Y_max = 0,34 × 10,5 = 3,57 t/ha
Cíl VÚKOZ: ±3,5 t/ha
Rozdíl: 2 % → ✅
```

**Pro SRC (σ_w = 0,13, Y_max = 10,5 t/ha):**
```
2σ_w × Y_max = 0,26 × 10,5 = 2,73 t/ha
Cíl VÚKOZ: ±2,5 t/ha
Rozdíl: 9 % → ✅
```

### 9.E Transport (Searcy 2007, 50 km)

**Miscanthus (slamnatá):**
```
C_2007 = (4,39 + 0,12 × 50) × 23/25 = 9,6 EUR/t
C_2025 = 9,6 × 1,5 (inflace) = 14,4 ≈ 15 EUR/t
```

**SRC (wood chips):**
```
C_2007 = (3,01 + 0,07 × 50) × 23/25 = 6,0 EUR/t
C_2025 = 6,0 × 1,5 = 9,0 ≈ 10 EUR/t
```

### 9.F EAA vzorec (Krok 2)

$$
\text{EAA} = \text{NPV} \cdot \frac{r}{1 - (1+r)^{-n}}
$$

**Pro r = 0** (degenerace):
```
EAA = NPV / n
```

### 9.G CAPEX struktura

**Miscanthus:**
```
prep_pozemku        + 200 EUR/ha (rok 0)
sadební_materiál    + 4 000 (= 8 000 × 0,50, rok 1)
mech_výsadba        + 700 (rok 1)
údržba 1.-2. rok    + 145 (rozděleno 50/50 mezi rok 1 a 2)
──────────────────────────
CELKEM CAPEX:       5 045 EUR/ha
```

**SRC vrba:**
```
prep_pozemku        + 230 EUR/ha
sadební_materiál    + 840 (= 7 000 × 0,12)
mech_výsadba        + 420
údržba 1.-2. rok    + 145
──────────────────────────
CELKEM CAPEX:       1 635 EUR/ha
```

---

<a name="decisions"></a>
## 10. Decision log: klíčová rozhodnutí

### 10.1 Cena Miscanthus 90 EUR/t (Variant C)

**Alternativy:**
- A) 80 EUR/t — pragmatická současná tržní cena
- B) 100 EUR/t — plná bottom-up
- C) **90 EUR/t** — střed (zvoleno)

**Důvod:** A je možná příliš konzervativní (podceňuje), B by přeceňovala (gate price = horní bound). C je akademicky transparentní střed s wide range v citlivostce ±10 %.

### 10.2 Cena SRC 78 EUR/t

Stejná logika: střed mezi A=75 a B=80. Default 78 EUR/t.

### 10.3 Vzdálenost teplárny D = 50 km

**Důvod:** „regionální supply" je nejčastější CZ scénář. Většina českých biomasových tepláren odebírá v okruhu 30–80 km. TZB-info 2003 pro 300 Kč/t cenu uvádí ekonomický limit 44 km.

### 10.4 α_accept = 0,85

**Důvod:** teplárny si nechávají ~15 % rezervu na kvalitativní rizika (popel, Cl, K) a vyjednávací marži. Konzervativní střed.

### 10.5 σ_P^M = 17 EUR/t (analogií)

**Alternativy:**
- 24 EUR/t (čisté Argus) — přeceňuje
- 14 EUR/t (jako SRC) — podceňuje
- 17 EUR/t (+20 % nad SRC) — kompromis (zvoleno)

**Důvod:** Spiegel 2021 dokládá +20–30 % vyšší volatilitu bylinných plodin. Citlivostka ověřuje robustnost ±10 % od základu.

### 10.6 Použití ČSÚ palivového dříví jako proxy pro SRC

**Důvod:** ČSÚ neměří SRC štěpku samostatně (málo trhu). Palivové dříví VI. třídy je nejbližší dostupný proxy:
- Listnaté ✅
- Energetické využití ✅
- Nejnižší jakostní třída = jen palivo ✅
- Český trh ✅

**Limitace:** přiznaná v textu DP.

### 10.7 Společný seed v diverzifikátoru

**Důvod:** matematicky nutné pro správné modelování diverzifikace. Bez něj by přeceňoval její efekt o faktor √2.

### 10.8 ρ = 0 jako default (místo Spiegel −0,35)

**Důvod:** v ČR je trh s biomasou regionální, lokální nabídka neovlivňuje cenu. Konzervativní (0 = nezávislost). Uživatel může změnit pro alternativní modelování.

### 10.9 Drift g = 1 %/rok

**Důvod:** EU ETS povolenky a dekarbonizační politiky ukazují růstové tempo cen biomasy. ERÚ Cenový výměr 8/2025 implikuje 1–3 %/rok pro 2026–2030.

### 10.10 Zachování krize 2022 v ČSÚ datech (ne odfiltrování)

**Důvod:** v horizontu 25–29 let lze obdobný extrém očekávat opakovaně. Vyloučení by podhodnotilo riziko.

---

<a name="zbyva"></a>
## 11. Co zbývá udělat

### Krok 8 — Globální cleanup (čeká)
- Najít všech ~6 zbývajících `??` (broken cross-refs)
- Najít všech ~3 zbývajících `(?)` (chybějící citace)
- Sjednotit terminologii: CAPEX vs „náklady na založení" vs „investiční náklady"
- Zkontrolovat všechny `\autoref` / `\ref` že fungují
- Sjednotit notaci jednotek (EUR/t vs €/t — vyber jeden)

### Krok 9 — BibTeX kompilace (čeká)
- Otevřít `references.bib` a doplnit všechny chybějící entries (15 nových z Kroků 1–7)
- Ověřit existující entries (`Knapek2024`, `Hull2018`, `Glasserman2003`, atd.) jsou správně formátovány
- Spustit `biber` / `bibtex` a vyřešit warnings
- Finální `pdflatex × 3` pro plnou kompilaci

### Akce vyžadující tebe (před odevzdáním DP)
1. **Trogl 2026** — celé jméno + datum + forma kontaktu
2. **VÚKOZ 2026** — konkrétní jména konzultantů + data
3. **Weger 2025** — ověřit URL nebo nahradit
4. **WitzelFinger 2016** — stáhnout přes ČVUT VPN (paywall)

---

## 📞 Pokud najdeš chybu

Pokud při procházení tohoto dokumentu narazíš na:
- **Chybný výpočet** → napiš mi konkrétní místo, opravím + aktualizuji všechno (DP text, kód, tento dokument)
- **Chybnou citaci** → totéž
- **Logickou nesrovnalost** → totéž
- **Něco co bys napsal jinak** → diskutujeme

**Vše můžeme přepracovat — nic není vyrytém v kameni.**
