import json
import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium
import requests
from scipy.stats import norm
from scipy.optimize import brentq
try:
    from bpej_lookup import get_fertility_from_coords, format_bpej_info
    BPEJ_AVAILABLE = True
except ImportError:
    BPEJ_AVAILABLE = False

# ===========================================================================
# NAČTENÍ PŘEKLADŮ Z JSON
# ===========================================================================
_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_translations():
    with open(os.path.join(_DIR, "translations.json"), encoding="utf-8") as f:
        return json.load(f)

LANG = load_translations()

ZONE_KEYS = [
    "Tropické a Subtropické",
    "Jižní Evropa / Středomoří",
    "Střední Evropa / Mírné pásmo",
    "Severní Evropa / Chladné pásmo",
    "Suché / Marginální půdy",
]
SOIL_KEYS = ["Optimální", "Průměrná", "Neúrodná", "Nevhodná"]

# ===========================================================================
# DATA
# ===========================================================================
DEFAULT_COSTS = {
    # Hodnoty kalibrované dle Weger (2021), Wagner (2022), Knápek a kol. (2024),
    # konzultace VÚKOZ Průhonice 2026 a Trogl 2026 (kontakt na německého dodavatele
    # oddenků Miscanthus, region Cheb–Františkovy Lázně).
    #
    # KONVENCE: zivotnost = celková délka projektu v letech (vč. roku 0 = příprava).
    #   Miscanthus 25 = ROK 0 (příprava) až ROK 24 (poslední sklizeň + likvidace)
    #   SRC vrba   30 = ROK 0 (příprava) až ROK 29 (konečná orba)
    "Miscanthus": {
        # Příprava + výsadba (jednorázově)
        "prep_pozemku":      104.0,    # ROK 0 – aktualizace VÚKOZ 2026
        "hustota_vysadby":   8000,     # oddenků/ha (Weger 2021)
        "cena_sadby_ks":     0.30,     # EUR/ks – kompromis mezi Trogl 2026 (CZ, 0,50)
                                       # a EU dodavateli miscanthus.eu (0,14–0,38 dle objemu),
                                       # zahrnuje dopravu a manipulaci
        "cena_vysadby_ks":   0.06,     # EUR/ks = 1,5 Kč/oddenek ÷ 25 CZK/EUR
                                       # — sazba převzata od VÚKOZ 2026 pro výsadbu řízků SRC;
                                       # pro Miscanthus expertní odhad analogií (adaptované
                                       # řízkovací stroje, technologicky obdobná operace)
        # mech_vysadba se počítá automaticky: hustota × cena_vysadby_ks
        # = 8000 × 0,06 = 480 EUR/ha
        "udrzba_1_2_rok":    145.0,    # EUR/ha celkem za roky 1+2 (rozděleno 50/50)
        # Roční provozní (od ROK 3+)
        "udrzba_rocni":      100.0,    # EUR/ha/rok – hnojení 60 kg N/ha NPK každé 3 roky
                                       # + Ca po 5. sklizni + overheads 40 EUR/ha (Weger 2021)
        "dan_pozemku":       36.0,     # EUR/ha/rok
        # Sklizeň a likvidace
        "sklizen_per_tuna":  25.0,
        "likvidace":         80.0,     # EUR/ha – snadné zničení oddenků orbou (ROK 24,
                                       # poslední rok = poslední sklizeň + likvidace)
        # Tržní a strukturální
        # Cena 116 EUR/t DM = bottom-up:
        #   P_ref × LHV_DM − T_DM = 7,60 × 17,61 − 17,9 ≈ 116 EUR/t DM
        # Zdroje:
        #   P_ref = 7,60 EUR/GJ (190 Kč/GJ při kurzu 25 CZK/EUR; Vyhláška
        #     315/2025 Sb. novelizující 79/2022 Sb., příloha č. 2; cit. v ERÚ
        #     Cenovém výměru 8/2025, str. 9, 13 — biomasa kat. 1 cíleně pěstovaná)
        #   LHV_DM = 17,61 GJ/t DM (THETA Metodika 2024, Tab. 4)
        #   T_DM = 17,9 EUR/t DM (AKO Blatný 2025: 49 Kč/km + 100 Kč/15min
        #     manipulace; 1,4 jízdy/ha při sypné hmotnosti řezanky 125 kg/m³
        #     (střed 100–140 dle Caslin 2010, Smeets 2009) a vlhkosti 20 %,
        #     referenční vzdálenost 30 km jednosměrně → 134 EUR/jízda)
        # Detail v sekci 3.5 DP. Citlivostka v rozsahu ±20 %.
        "prodejni_cena_start": 116.0,
        "riziko_fail":         0.05,
        "zivotnost":           25,     # celková délka projektu (ROK 0..24)
        "capex_std":           300.0,  # SD CAPEX (ROK 1) – vyšší kvůli ceně oddenků
    },
    "SRC Vrba": {
        # Příprava + výsadba
        "prep_pozemku":      104.0,    # ROK 0 – aktualizace VÚKOZ 2026
        "hustota_vysadby":   7000,     # řízků/ha (Weger 2021)
        "cena_sadby_ks":     0.12,     # EUR/ks ≈ 3 Kč/řízek
        "cena_vysadby_ks":   0.06,     # EUR/ks = 1,5 Kč/řízek ÷ 25 CZK/EUR (VÚKOZ 2026)
        # mech_vysadba se počítá automaticky: hustota × cena_vysadby_ks
        # = 7000 × 0,06 = 420 EUR/ha
        "udrzba_1_2_rok":    145.0,    # EUR/ha celkem za roky 1+2
        # Roční provozní (od ROK 3+ údržba mezi sklizněmi)
        "udrzba_rocni":      50.0,     # EUR/ha/rok – aktualizace VÚKOZ 2026
        "dan_pozemku":       36.0,     # EUR/ha/rok
        # Sklizeň a likvidace
        "sklizen_per_tuna":  40.0,     # EUR/t – aktualizace VÚKOZ 2026 (omezená nabídka techniky v ČR)
        "likvidace":         620.0,    # EUR/ha – biologicko-mechanické rušení
                                       # rozděleno 189 + 189 + 240 v letech 27/28/29
        # Tržní a strukturální
        # Cena 128 EUR/t DM = bottom-up:
        #   P_ref × LHV_DM − T_DM = 7,60 × 18,39 − 11,3 ≈ 128 EUR/t DM
        # Zdroje:
        #   P_ref = 7,60 EUR/GJ (190 Kč/GJ; Vyhláška 315/2025 Sb., příl. č. 2;
        #     cit. v ERÚ Cenovém výměru 8/2025, str. 9, 13 — biomasa kat. 1)
        #   LHV_DM = 18,39 GJ/t DM (THETA Metodika 2024, Tab. 4 — RRD/SRC vrba)
        #   T_DM = 11,3 EUR/t DM (AKO Blatný 2025: 49 Kč/km + 100 Kč/15min
        #     manipulace; 9,5 t DM/ha průměrný výnos × 3 roky / 0,48 vlhkost
        #     ≈ 59,4 t mokré štěpky/ha → 2,4 jízdy/ha; sypná hmotnost
        #     333 kg/m³ čerstvé štěpky dle Stolarski 2019;
        #     referenční vzdálenost 30 km jednosměrně → 134 EUR/jízda)
        # Pozn.: Nižší T_DM než u Miscanthusu — SRC štěpka má vyšší sypnou
        # hmotnost, takže méně jízd/t DM (objemový vs hmotnostní limit).
        # Detail v sekci 3.5 DP. Citlivostka v rozsahu ±20 %.
        "prodejni_cena_start": 128.0,
        "riziko_fail":         0.05,
        "zivotnost":           30,     # celková délka projektu (ROK 0..29)
        "capex_std":           150.0,
    },
}

# Výnosové rozsahy [min, max] t/ha sušiny – diferencováno dle klimatické zóny a kvality půdy.
#
# Zdroje:
#   Miscanthus – Lewandowski et al. (2000), Clifton-Brown et al. (2024), OPTIMISC EU FP7,
#                Strašil (2003–2012), Weger/VÚKOZ, Castellano Albors (2025)
#   SRC vrba   – Stolarski et al. (2019), Aylott et al. (2008), Weger & Bubeník,
#                Castellano Albors (2025), Forest Research UK
#
# Klíčové odlišnosti oproti původnímu modelu (kde byly všechny zóny identické):
#   • Miscanthus dominuje v tropech a Jižní Evropě díky C4 fotosyntéze.
#   • SRC vrba je nejsilnější v Severní Evropě (domácí plodina Skandinávie).
#   • Miscanthus má kritické omezení v Severní Evropě (přežívání zimy pod −3.5 °C).
#   • Suché zóny postihují oba druhy, vrba je odolnější na písčitých půdách.
YIELD_DATA = {
    # Hodnoty dle Tabulky 3.1 diplomové práce (Dostál 2026).
    # Střední Evropa: konzultace s VÚKOZ Průhonice; ostatní zóny dle mezinárodní literatury
    # (Abdalla 2024, Lewandowski 2000, Jørgensen 2003, Castellano Albors 2025, Verwijst 2013,
    # Bacenetti 2016, Richard 2019, Ouattara 2022).
    "Tropické a Subtropické": {
        "Optimální": {"M_giganteus": [20, 30], "SRC": [18, 25]},
        "Průměrná":  {"M_giganteus": [12, 20], "SRC": [10, 18]},
        "Neúrodná":  {"M_giganteus": [6,  12], "SRC": [4,  10]},
        "Nevhodná":  {"M_giganteus": [0,   6], "SRC": [0,   4]},
    },
    "Jižní Evropa / Středomoří": {
        "Optimální": {"M_giganteus": [18, 28], "SRC": [12, 18]},
        "Průměrná":  {"M_giganteus": [10, 18], "SRC": [7,  12]},
        "Neúrodná":  {"M_giganteus": [4,  10], "SRC": [3,   7]},
        "Nevhodná":  {"M_giganteus": [0,   4], "SRC": [0,   3]},
    },
    "Střední Evropa / Mírné pásmo": {
        "Optimální": {"M_giganteus": [14, 16], "SRC": [12, 15]},
        "Průměrná":  {"M_giganteus": [7,  14], "SRC": [7,  12]},
        "Neúrodná":  {"M_giganteus": [4,   7], "SRC": [4,   7]},
        "Nevhodná":  {"M_giganteus": [0,   4], "SRC": [0,   4]},
    },
    "Severní Evropa / Chladné pásmo": {
        "Optimální": {"M_giganteus": [8,  12], "SRC": [6,  10]},
        "Průměrná":  {"M_giganteus": [4,   8], "SRC": [3,   6]},
        "Neúrodná":  {"M_giganteus": [2,   4], "SRC": [1,   3]},
        "Nevhodná":  {"M_giganteus": [0,   2], "SRC": [0,   1]},
    },
    "Suché / Marginální půdy": {
        "Optimální": {"M_giganteus": [6,  10], "SRC": [3,   7]},
        "Průměrná":  {"M_giganteus": [3,   6], "SRC": [1,   4]},
        "Neúrodná":  {"M_giganteus": [1,   3], "SRC": [0,   1]},
        "Nevhodná":  {"M_giganteus": [0,   1], "SRC": [0,   1]},
    },
}

# ===========================================================================
# CITLIVOSTNÍ ANALÝZA – 5 klíčových parametrů
# ===========================================================================
SENSITIVITY_PARAMS = {
    # --- Nákladové / výnosové ---
    "prodejni_cena_start": {"cs": "Prodejní cena (EUR/t DM)",     "en": "Selling price (EUR/t DM)",
                            "cs_short": "Prodejní cena",           "en_short": "Selling price"},
    "yield_potential":     {"cs": "Výnosový potenciál (t/ha)",   "en": "Yield potential (t/ha)",
                            "cs_short": "Výnos",                  "en_short": "Yield"},
    "cena_sadby_ks":       {"cs": "Cena sadby (EUR/ks)",          "en": "Planting material price (EUR/pc)",
                            "cs_short": "Cena sadby",              "en_short": "Plant. price"},
    "udrzba_rocni":        {"cs": "Roční údržba (EUR/ha)",       "en": "Annual maintenance (EUR/ha)",
                            "cs_short": "Údržba",                 "en_short": "Maintenance"},
    "sklizen_per_tuna":    {"cs": "Náklady na sklizeň (EUR/t)",  "en": "Harvest cost (EUR/t)",
                            "cs_short": "Sklizeň",                "en_short": "Harvest cost"},
    # --- Rizikové / strukturální ---
    "riziko_fail":         {"cs": "Riziko selhání plantáže",     "en": "Plantation failure risk",
                            "cs_short": "Riziko selhání",         "en_short": "Failure risk"},
    "zivotnost":           {"cs": "Životnost plantáže (let)",    "en": "Plantation lifetime (yr)",
                            "cs_short": "Životnost",              "en_short": "Lifetime"},
    "subsidy_pct":         {"cs": "Podíl dotačního krytí (%)",   "en": "Subsidy rate (%)",
                            "cs_short": "Dotace",                 "en_short": "Subsidy"},
    "discount_rate":       {"cs": "Diskontní sazba (%)",         "en": "Discount rate (%)",
                            "cs_short": "Disk. sazba",            "en_short": "Discount rate"},
    "weather_prob":        {"cs": "Prav. klimatického stresu",   "en": "Weather stress probability",
                            "cs_short": "Klim. stres",            "en_short": "Weather stress"},
    "decarbon_drift":      {"cs": "Eskalace prodejní ceny g (%/rok)", "en": "Selling price escalation g (%/yr)",
                            "cs_short": "Eskalace ceny",          "en_short": "Price esc."},
    "cost_escalation":     {"cs": "Eskalace nákladů eₒ (%/rok)",  "en": "Cost escalation eₒ (%/yr)",
                            "cs_short": "Eskalace nákladů",       "en_short": "Cost esc."},
}
SA_MAX_SELECTED = 5

SENSITIVITY_STRINGS = {
    "cs": {
        "sa_header":       "Citlivostní analýza",
        "sa_desc":         "Vyberte parametry, jejichž vliv na výsledky chcete analyzovat. "
                           "Pro každý parametr proběhne simulace při ±10 % odchylce od základní hodnoty.",
        "sa_select_label": "Parametry pro citlivostní analýzu",
        "sa_running":      "Počítám citlivostní analýzu pro {crop} – parametr {param}...",
        "sa_title_profit": "Citlivost průměrného zisku (±10 %)",
        "sa_title_std":    "Citlivost směrodatné odchylky (±10 %)",
        "sa_title_var":    "Citlivost VaR 5 % (±10 %)",
        "sa_title_cvar":   "Citlivost CVaR 5 % (±10 %)",
        "sa_base":         "Základ",
        "sa_low":          "−10 %",
        "sa_high":         "+10 %",
        "sa_title_pb_prob":"Citlivost pravděpodobnosti návratnosti (±10 %)",
        "sa_title_pb_yr":  "Citlivost doby návratnosti (±10 %)",
        "sa_header_recap": "Citlivostní analýza – rekapitulace",
    },
    "en": {
        "sa_header":       "Sensitivity Analysis",
        "sa_desc":         "Select parameters whose impact on results you want to analyze. "
                           "For each parameter, simulation runs at ±10 % deviation from base value.",
        "sa_select_label": "Parameters for sensitivity analysis",
        "sa_running":      "Running sensitivity analysis for {crop} – parameter {param}...",
        "sa_title_profit": "Mean Profit Sensitivity (±10 %)",
        "sa_title_std":    "Std. Deviation Sensitivity (±10 %)",
        "sa_title_var":    "VaR 5 % Sensitivity (±10 %)",
        "sa_title_cvar":   "CVaR 5 % Sensitivity (±10 %)",
        "sa_title_pb_prob":"Payback Probability Sensitivity (±10 %)",
        "sa_title_pb_yr":  "Payback Period Sensitivity (±10 %)",
        "sa_base":         "Base",
        "sa_low":          "−10 %",
        "sa_high":         "+10 %",
        "sa_header_recap": "Sensitivity Analysis – Recap",
    },
}


SA_STEPS = np.linspace(-0.10, 0.10, 11)   # −10 %, −8 %, … 0 %, … +8 %, +10 %
SA_PCT_LABELS = [f"{s*100:+.0f} %" for s in SA_STEPS]

# Drift: absolutní hodnoty −5 % až +5 %/rok (ne odchylka od základu)
SA_DRIFT_STEPS = np.linspace(-0.05, 0.05, 11)
# Eskalace nákladů: absolutní hodnoty 0 % až 5 %/rok
SA_ESC_STEPS   = np.linspace(0.0, 0.05, 11)


def run_param_sensitivity(param_key, base_params, y_bounds, crop_type, n_sim,
                          years, subsidy_perc, area_ha, discount_rate,
                          weather_prob=0.05, src_tech="Direct Chip",
                          rho=0.0, drift=0.010, cost_escalation=0.02,
                          pachtovne=0.0, soil_quality="Průměrná"):
    """
    Spustí simulaci pro jeden parametr v 11 krocích.
    Speciální absolutní osy: 'decarbon_drift' (−5 až +5 %), 'cost_escalation' (0 až 5 %).
    """
    profits, eaas, stds, vars_, cvars = [], [], [], [], []
    payback_probs, payback_years = [], []

    is_drift = (param_key == "decarbon_drift")
    is_esc   = (param_key == "cost_escalation")
    if is_drift:
        steps_iter = SA_DRIFT_STEPS
    elif is_esc:
        steps_iter = SA_ESC_STEPS
    else:
        steps_iter = SA_STEPS

    # Default σ_P pro každou plodinu (viz simulate_*: M=20.0, SRC=16.6)
    sigma_P_default = 20.0 if crop_type == "misc" else 16.6

    for step in steps_iter:
        factor = 1.0 + step
        p = dict(base_params)
        yb = list(y_bounds)
        sim_years = years
        sub = subsidy_perc
        r = discount_rate
        wp = weather_prob
        d = drift
        ce = cost_escalation
        sP = sigma_P_default

        # Rozřazení parametru
        if param_key == "yield_potential":
            yb = [y_bounds[0] * factor, y_bounds[1] * factor]
        elif param_key == "subsidy_pct":
            sub = np.clip(subsidy_perc * factor, 0.0, 1.0)
        elif param_key == "discount_rate":
            r = discount_rate * factor
        elif param_key == "weather_prob":
            wp = np.clip(weather_prob * factor, 0.0, 1.0)
        elif param_key == "zivotnost":
            sim_years = max(5, int(round(years * factor)))
            p["zivotnost"] = sim_years
        elif param_key == "decarbon_drift":
            d = float(step)
        elif param_key == "cost_escalation":
            ce = float(step)
        elif param_key == "sigma_P":
            sP = sigma_P_default * factor
        elif param_key in p:
            p[param_key] = base_params[param_key] * factor

        if crop_type == "misc":
            cf, _, _, _ = simulate_miscanthus(n_sim, sim_years, p, yb, sub, wp,
                                               rho=rho, drift=d, cost_escalation=ce,
                                               pachtovne=pachtovne, sigma_P=sP)
        else:
            cf, _, _, _ = simulate_src(n_sim, sim_years, p, yb, src_tech, sub, wp,
                                        rho=rho, drift=d, cost_escalation=ce,
                                        pachtovne=pachtovne,
                                        soil_quality=soil_quality, sigma_P=sP)

        cf_total = cf * area_ha
        if r > 0:
            df = (1 + r) ** -np.arange(sim_years)
            total_profits = np.sum(cf_total * df[:, np.newaxis], axis=0)
        else:
            total_profits = np.sum(cf_total, axis=0)

        var5 = float(np.percentile(total_profits, 5))
        mask = total_profits <= var5
        cvar5 = float(total_profits[mask].mean()) if mask.any() else var5

        # Pravděpodobnost a doba návratnosti
        cum_cf = np.cumsum(cf_total, axis=0)
        pb_yrs = np.argmax(cum_cf > 0, axis=0)
        succ   = pb_yrs[pb_yrs > 0]
        pb_prob = float(len(succ) / n_sim) if n_sim > 0 else 0.0
        avg_pb  = float(np.mean(succ) + 1) if len(succ) > 0 else float("nan")

        profits.append(float(np.mean(total_profits)))
        eaas.append(float(np.mean(equivalent_annual_annuity(total_profits, sim_years, r))))
        stds.append(float(np.mean(np.std(cf_total, axis=0))))
        vars_.append(var5)
        cvars.append(cvar5)
        payback_probs.append(pb_prob)
        payback_years.append(avg_pb)

    pct_axis = (np.array(steps_iter) * 100).tolist()
    return {
        "pct": pct_axis,
        "profit": profits,
        "eaa": eaas,
        "std": stds,
        "var": vars_,
        "cvar": cvars,
        "payback_prob": payback_probs,
        "payback_yr": payback_years,
        "is_absolute_axis": (is_drift or is_esc),
    }


SA_LINE_COLORS = [
    "#1f77b4",   # modrá  – Prodejní cena
    "#ff7f0e",   # oranžová – Výnosový potenciál
    "#2ca02c",   # zelená – Náklady na založení
    "#d62728",   # červená – Roční údržba
    "#9467bd",   # fialová – Náklady na sklizeň
]

# Sjednocená paleta plodin – používaná napříč VŠEMI grafy
COLOR_MISC      = "#7CB342"        # zelená – Miscanthus (C4 tráva)
COLOR_MISC_DARK = "#33691E"
COLOR_SRC       = "#8D6E63"        # hnědá – SRC vrba (dřevina)
COLOR_SRC_DARK  = "#3E2723"

CROP_COLOR = {
    "M_giganteus": COLOR_MISC,
    "Miscanthus":  COLOR_MISC,
    "SRC Vrba":    COLOR_SRC,
    "SRC":         COLOR_SRC,
    "misc":        COLOR_MISC,
    "src":         COLOR_SRC,
}


def fmt_eur_cs(val, decimals=0):
    """České formátování: 1 234 567 € (mezery jako oddělovač tisíců)."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "N/A"
    s = f"{val:,.{decimals}f}".replace(",", " ").replace(".", ",")
    return f"{s} €"


def fmt_int_cs(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{int(round(val)):,}".replace(",", " ")


def make_sensitivity_line_chart(all_sa, metric_key, title, yaxis_label,
                                param_labels_short, lang_code):
    """
    Spojnicový graf: X = odchylka parametru (%) nebo absolutní hodnota (drift), Y = metrika.
    Pokud některý parametr má is_absolute_axis=True, vykreslí se na sekundární ose X nahoře.
    """
    # Rozdělíme parametry: relativní odchylka (dolní osa) vs absolutní (horní osa)
    rel_keys = [k for k, v in all_sa.items() if not v.get("is_absolute_axis")]
    abs_keys = [k for k, v in all_sa.items() if v.get("is_absolute_axis")]

    T_local = LANG[lang_code]

    fig = go.Figure()
    color_idx = 0
    for pk in rel_keys:
        sa_data = all_sa[pk]
        fig.add_trace(go.Scatter(
            x=sa_data["pct"], y=sa_data[metric_key],
            mode="lines+markers",
            name=param_labels_short[pk],
            line=dict(color=SA_LINE_COLORS[color_idx % len(SA_LINE_COLORS)], width=2.5),
            marker=dict(size=5),
            xaxis="x",
        ))
        color_idx += 1

    for pk in abs_keys:
        sa_data = all_sa[pk]
        fig.add_trace(go.Scatter(
            x=sa_data["pct"], y=sa_data[metric_key],
            mode="lines+markers",
            name=param_labels_short[pk] + " (abs %)",
            line=dict(color=SA_LINE_COLORS[color_idx % len(SA_LINE_COLORS)],
                      width=2.5, dash="dot"),
            marker=dict(size=5, symbol="diamond"),
            xaxis="x2",
        ))
        color_idx += 1

    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

    # Titulek umístíme nad celý graf; pokud je aktivní sekundární osa (nahoře),
    # potřebuje víc místa, aby se popisek osy nepřekrýval s názvem grafu.
    has_abs = bool(abs_keys)
    top_margin = 120 if has_abs else 60
    title_y = 0.99 if has_abs else 0.95
    layout_kwargs = dict(
        title=dict(text=title, font=dict(size=15), x=0.5, xanchor="center",
                   y=title_y, yanchor="top"),
        yaxis_title=yaxis_label,
        xaxis=dict(
            title=T_local["sa_x_axis_label"],
            dtick=2, range=[-11, 11],
        ),
        legend=dict(
            orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        height=480,
        margin=dict(t=top_margin, b=110),
    )
    if has_abs:
        layout_kwargs["xaxis2"] = dict(
            title=dict(text=T_local["sa_x_axis2_label"], standoff=4),
            overlaying="x", side="top",
            dtick=1, range=[-5.5, 5.5],
        )
    fig.update_layout(**layout_kwargs)
    return fig


def make_tornado_chart(all_sa, metric_key, param_labels_full, lang_code,
                        baseline_value=None, top_n=10):
    """
    Tornádový graf: horizontální bary řazené sestupně podle |max impact|.

    Pro každý parametr v all_sa:
      - low_delta  = metric_at_lowest_step − metric_at_zero_step
      - high_delta = metric_at_highest_step − metric_at_zero_step
      - impact     = max(|low_delta|, |high_delta|)

    Modrý levý bar = efekt poklesu parametru (-10 %).
    Červený pravý bar = efekt nárůstu parametru (+10 %).
    Vertikální čára na 0 = baseline.

    Použití: jeden tornádo na jednu metriku (typicky 'profit' = NPV) per plodina.
    """
    if not all_sa:
        return None

    rows = []
    for pk, sa_data in all_sa.items():
        vals = sa_data.get(metric_key, [])
        if len(vals) < 3:
            continue
        # Najít index baseline (kde pct nejblíž 0; pro absolutní osa drift = pct=0)
        pct_arr = np.asarray(sa_data.get("pct", []))
        if len(pct_arr) == 0:
            continue
        zero_idx = int(np.argmin(np.abs(pct_arr)))
        baseline = vals[zero_idx]
        low_val  = vals[0]
        high_val = vals[-1]
        low_delta  = low_val  - baseline
        high_delta = high_val - baseline
        rows.append({
            "param":      pk,
            "label":      param_labels_full.get(pk, pk),
            "low_delta":  low_delta,
            "high_delta": high_delta,
            "impact":     max(abs(low_delta), abs(high_delta)),
            "low_pct":    pct_arr[0],
            "high_pct":   pct_arr[-1],
        })

    if not rows:
        return None

    rows.sort(key=lambda r: r["impact"], reverse=True)
    rows = rows[:top_n]
    rows.reverse()  # největší vliv navrch (Plotly kreslí odspoda)

    labels      = [r["label"]      for r in rows]
    low_deltas  = [r["low_delta"]  for r in rows]
    high_deltas = [r["high_delta"] for r in rows]

    T_local = LANG[lang_code]

    fig = go.Figure()
    # Levá strana (modrá: efekt -10 %)
    fig.add_trace(go.Bar(
        y=labels, x=low_deltas, orientation="h",
        name=T_local["sa_neg_param_name"],
        marker_color="#1976D2",
        text=[f"{v:+,.0f} €".replace(",", " ") for v in low_deltas],
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>−10 %% → Δ %{x:+,.0f} EUR<extra></extra>",
    ))
    # Pravá strana (červená: efekt +10 %)
    fig.add_trace(go.Bar(
        y=labels, x=high_deltas, orientation="h",
        name=T_local["sa_pos_param_name"],
        marker_color="#D32F2F",
        text=[f"{v:+,.0f} €".replace(",", " ") for v in high_deltas],
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>+10 %% → Δ %{x:+,.0f} EUR<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#000", width=1, dash="solid"))

    if baseline_value is not None:
        baseline_fmt = f"{baseline_value:,.0f}".replace(",", " ")
        chart_title = T_local["tornado_title_baseline"].format(baseline=baseline_fmt)
    else:
        chart_title = T_local["tornado_title_nobaseline"]

    fig.update_layout(
        title=chart_title,
        barmode="overlay",
        xaxis=dict(title=T_local["tornado_x_axis"],
                    zeroline=True, zerolinewidth=2, zerolinecolor="#000"),
        yaxis=dict(title=""),
        height=max(360, 40 * len(rows) + 120),
        margin=dict(t=60, b=60, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        bargap=0.25,
    )
    return fig


# ===========================================================================
# VĚDECKÉ FUNKCE
# ===========================================================================
@st.cache_data
def get_climate_data(lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": "2014-01-01", "end_date": "2023-12-31",
        "daily": ["temperature_2m_mean", "precipitation_sum"],
        "timezone": "auto",
    }
    try:
        data = requests.get(url, params=params).json()
        temps = [x for x in data["daily"]["temperature_2m_mean"] if x is not None]
        rain  = [x for x in data["daily"]["precipitation_sum"]   if x is not None]
        return sum(temps)/len(temps), (sum(rain)/len(rain))*365
    except Exception:
        return None, None


def determine_zone(lat, avg_temp, avg_rain):
    if avg_rain < 450:                    return "Suché / Marginální půdy"
    if abs(lat) < 23.5 or avg_temp > 22:  return "Tropické a Subtropické"
    if avg_temp > 13:                     return "Jižní Evropa / Středomoří"
    if avg_temp > 8:                      return "Střední Evropa / Mírné pásmo"
    return "Severní Evropa / Chladné pásmo"


# ===========================================================================
# Weger 2021 výnosová křivka pro Miscanthus × giganteus
# ===========================================================================
# Zdroj: Weger et al. (2021) "Can Miscanthus Fulfill Its Expectations as
# an Energy Biomass Source in the Current Conditions of the Czech Republic?"
# Agriculture 11(1):40, Figure 4 (Yc 1–6 yield curves).
# Hodnoty přečteny přímo z grafu pro křivku Yc 6 (peak 15,5 t DM/ha)
# a normalizovány na peak = 1,000. Všech 6 publikovaných křivek (Yc 1–6) má
# stejný relativní tvar, liší se pouze absolutním peak yieldem podle
# stanoviště – proto profil aplikujeme univerzálně přes Y_max scaling.
#
# DŮLEŽITÉ: Profil začíná hodnotou Weger Y3 (16 %), protože v modelu jsou
# ROK 0 (příprava) + ROK 1 (sadba/výsadba) + ROK 2 (údržba) zachyceny jako
# setup phase BEZ harvestu (yields[0:3] = 0). První produkční rok (ROK 3)
# odpovídá Weger Y3 = 16 %, ROK 4 = Weger Y4 = 94 % (PRUDKÝ NÁRŮST),
# ROK 5 = Weger Y5 = 100 % (PEAK).
#
# Profile[i] aplikuje na simulační rok (i + 3) v 0-indexaci (= i + 4 v
# 1-indexaci textu DP), tj. yields[n_setup + i] = Profile[i] × Y_max.
#
# Profile pokrývá 22 produkčních let (ROK 3–24 simulace, t.j. 25letá životnost),
# což odpovídá Weger Y3–Y16 (publikovaná data) plus Y17–Y24 (extrapolace
# mírnějšího úpadku).
MISCANTHUS_2021_PROFILE = np.array([
    0.00,  # ROK 1 simulace = Weger Y1 – rok výsadby, žádný harvestable výnos
    0.03,  # ROK 2 simulace = Weger Y2 – tiny early biomass (~3 % peaku)
    0.16,  # ROK 3 simulace = Weger Y3 – plantáž stále zakládá rhizomový systém
    0.94,  # ROK 4 = Weger Y4 – PRUDKÝ NÁRŮST (rhizomový systém dospívá)
    1.00,  # ROK 5 = Weger Y5 – PEAK
    0.97,  # ROK 6 = Weger Y6
    1.00,  # ROK 7 = Weger Y7
    1.00,  # ROK 8 = Weger Y8
    0.94,  # ROK 9 = Weger Y9 – mírný pokles startuje
    0.90,  # ROK 10 = Weger Y10
    0.77,  # ROK 11 = Weger Y11
    0.77,  # ROK 12 = Weger Y12
    0.71,  # ROK 13 = Weger Y13
    0.65,  # ROK 14 = Weger Y14
    0.55,  # ROK 15 = Weger Y15
    0.48,  # ROK 16 = Weger Y16 – konec publikovaných dat
    # Extrapolace pro plnou 26letou životnost (mírnější úpadek než Y14–Y16):
    0.46,  # ROK 17 = Y17
    0.44,  # ROK 18 = Y18
    0.42,  # ROK 19
    0.40,  # ROK 20
    0.38,  # ROK 21
    0.36,  # ROK 22
    0.35,  # ROK 23
    0.33,  # ROK 24
    0.30,  # ROK 25 – poslední sklizeň před likvidací plantáže
])


def gompertz_growth(t_arr, y_max=1.0, b=None, c=None):
    """
    Výnosová křivka Miscanthus × giganteus dle Weger 2021 (Agriculture 11(1):40).

    Funkce vrací relativní roční výnos (peak = 1,000) pro daný rok produkce
    (t = 1 odpovídá prvnímu produkčnímu roku po setup fázi, t.j. ROKu 3
    simulace, což je Weger Y3 = 16 % peaku).

    Použití v simulaci: roční výnos = gompertz_growth(t) × Y_max × weather(t).
    Y_max je interpretován jako PEAK roční výnos (špičkový rok plantáže).

    Parametry b, c zachovány v signatuře pro zpětnou kompatibilitu, ale nejsou
    používány — funkce nyní využívá piecewise-konstantní profil z grafu Weger
    2021 Yc 6.

    Charakteristika křivky (číslováno dle ROK simulace):
      - ROK 0 (mimo profil): příprava pozemku, žádný harvest
      - ROK 1 (mimo profil): sadba, žádný harvest
      - ROK 2 (mimo profil): údržba, žádný harvest
      - ROK 3 (Profile[0]): 16 % peaku – první malý výnos
      - ROK 4 (Profile[1]): 94 % – PRUDKÝ NÁRŮST
      - ROK 5–8 (Profile[2..5]): peak plateau (97–100 %)
      - ROK 9–14 (Profile[6..11]): postupný úpadek (90 → 65 %)
      - ROK 15–16 (Profile[12..13]): zrychlený úpadek (55 → 48 %)
      - ROK 17–24 (Profile[14..21]): extrapolovaný mírný úpadek (46 → 33 %)
      - Y17–Y25: extrapolovaný mírný úpadek (46 → 32 %)
    """
    t = np.asarray(t_arr, dtype=float)
    n_p = len(MISCANTHUS_2021_PROFILE)
    idx = np.clip((t - 1).astype(int), 0, n_p - 1)
    # Steady-state extrapolace: pro t > n_p vrací poslední hodnotu profilu
    # (cca 30 % peaku pro Miscanthus). Konzervativní předpoklad — plantáž
    # ke konci životnosti drží nízkou úroveň úpadku, ne náhlý pád na 0.
    valid = (t >= 1)
    return np.where(valid, y_max * MISCANTHUS_2021_PROFILE[idx], 0.0)


# ===========================================================================
# Weger 2025 expertní harvest profily pro SRC vrbu (4 kvality půdy)
# ===========================================================================
# Zdroj: Aktualizace Weger 2025 (VÚKOZ Průhonice), tabulka výnosových křivek
# RRD plantáže pro 7 sklizní v 3letém obmýtí, sloupce PO0–PO6.
# Mapování PO tříd → naše 4 kvality půdy (per Dostál 2026):
#   - Nevhodná  = PO0  (Ȳ = 2 t DM/ha/rok)
#   - Neúrodná  = průměr(PO1, PO2)  (Ȳ ≈ 5)
#   - Průměrná  = průměr(PO3, PO4)  (Ȳ ≈ 9)
#   - Optimální = průměr(PO5, PO6)  (Ȳ ≈ 13)
# Hodnoty H1–H7 získány vydělením fresh-yield v každé sklizni hodnotou max
# sklizně příslušného PO sloupce (normalizace na peak = 1,000).
# Osmá hodnota (H8) je extrapolace pokračujícího úpadku: H7 × (H7/H6),
# tj. závislá na kvalitě půdy — horší půda → rychlejší úpadek.
WEGER_2025_HARVEST = {
    "Nevhodná":  np.array([0.090, 0.607, 1.000, 1.000, 0.848, 0.545, 0.152, 0.042]),
    "Neúrodná":  np.array([0.186, 0.601, 0.932, 1.000, 0.932, 0.660, 0.312, 0.148]),
    "Průměrná":  np.array([0.355, 0.768, 0.955, 1.000, 0.916, 0.749, 0.497, 0.329]),
    "Optimální": np.array([0.309, 0.877, 1.000, 0.974, 0.859, 0.686, 0.500, 0.364]),
}


def src_yield_curve(t_arr, soil_quality="Průměrná"):
    """
    Relativní roční produkce suché hmoty SRC vrby v průběhu životnosti plantáže.

    Model: piecewise-konstantní v rámci 3letých sklizňových cyklů, kalibrovaný
    na expertní data Weger 2025 (VÚKOZ Průhonice). Funkce vrací přímo
    relativní výnos vůči peaku (max = 1,000), bez další normalizace.

    Profil dle kvality půdy:
      - Nevhodná: rychlý úpadek po peak (H8 ≈ 4 % peaku)
      - Neúrodná: středně rychlý úpadek (H8 ≈ 15 %)
      - Průměrná: postupný úpadek (H8 ≈ 33 %)
      - Optimální: pomalý úpadek (H8 ≈ 36 %)

    Použití v simulaci:
        annual_growth(t) = src_yield_curve(t, q) × Y_max × weather(t)

    Y_max je interpretován jako PEAK roční výnos (t DM/ha/rok ve špičce
    plantáže). Sklizeň za 3letý cyklus h v roce sklizně:
        H_h = sum_{t in cycle h} annual_growth(t) ≈ 3 × WEGER[h] × Y_max
    (neboť všechny roky v rámci jednoho cyklu mají stejný faktor a
     průměrné weather ≈ 1,0).

    Charakteristiky všech profilů:
      - Etablace (rok 1-3): velmi nízký výnos, plantáž zakládá kořenový systém
      - Růst do peaku (rok 4-9): zrychlený růst, dorůstání pařezového systému
      - Peak plateau (rok 9-12): maximum produktivity = Y_max × 1,000
      - Postupný úpadek (rok 13-24): empiricky pozorovaný pokles dlouhých plantáží,
        rychlost úpadku závisí na kvalitě půdy
    """
    t = np.asarray(t_arr, dtype=float)
    if soil_quality not in WEGER_2025_HARVEST:
        soil_quality = "Průměrná"  # fallback pro neznámé kategorie
    H = WEGER_2025_HARVEST[soil_quality]
    n_h = len(H)

    # Index cyklu pro každý rok t (t=1..24): cycle = (t-1) // 3
    cycle = np.clip(((t - 1) // 3).astype(int), 0, n_h - 1)
    # Steady-state extrapolace: pro t > 24 (8 sklizní) vrací poslední H[7]
    # (= H8 dané kvality půdy: 0,042 nevhodná → 0,364 optimální).
    # Konzervativní předpoklad — plantáž drží úroveň posledního cyklu.
    valid = (t >= 1)

    return np.where(valid, H[cycle], 0.0)


def equivalent_annual_annuity(npv, years, discount_rate):
    """
    Equivalent Annual Annuity – ročně přepočtená hodnota NPV.
    Umožňuje férové srovnání plodin s různou životností (Miscanthus 25 let vs SRC 30 let).
    Pro r=0 degeneruje na prostý průměr NPV/n.
    npv může být skalár nebo numpy array (pro vektorizaci přes scénáře).
    """
    if discount_rate <= 0:
        return npv / years
    factor = discount_rate / (1.0 - (1.0 + discount_rate) ** (-years))
    return npv * factor


def compute_irr_distribution(cf_matrix, hi=2.00):
    """
    Pro každý MC scénář (sloupec cf_matrix) řeší NPV(r) = 0 a vrátí distribuci IRR.

    Strategie: hledáme \"investor IRR\" — tj. nejvyšší r, nad nímž je projekt
    ztrátový (NPV < 0). To je ekonomicky relevantní IRR i u projektů
    s end-of-life negativními CF (likvidace SRC vrby), kde NPV(r) může být
    nemonotónní a~mít více kořenů.

    Algoritmus:
      1. Pokud NPV(0) > 0  → IRR leží v~[0, hi]  (ziskový projekt)
      2. Pokud NPV(0) < 0  → IRR leží v~[lo, 0]  (ztrátový projekt)
      3. Jinak NaN.

    cf_matrix : np.ndarray  shape (years, n_sim)
    hi        : horní mez prohledávání pro ziskové projekty (default 200 %)

    Vrací:
      irr   : np.ndarray shape (n_sim,)
      valid : np.ndarray bool shape (n_sim,)
    """
    years, n_sim = cf_matrix.shape
    t_arr = np.arange(years, dtype=float)
    irrs  = np.full(n_sim, np.nan)
    LO_LOSS = -0.30   # dolní mez pro ztrátové projekty (vyhne se overflow)

    for i in range(n_sim):
        cf = cf_matrix[:, i]
        if not (np.any(cf > 0) and np.any(cf < 0)):
            continue

        def npv_func(r, _cf=cf, _t=t_arr):
            return float(np.sum(_cf / (1.0 + r) ** _t))

        try:
            npv_at_zero = float(np.sum(cf))      # NPV(0) = prostý součet CF
            npv_at_hi   = npv_func(hi)
            if not np.isfinite(npv_at_hi):
                continue

            if npv_at_zero > 0 and npv_at_hi < 0:
                # Ziskový projekt — IRR leží v [0, hi]
                irrs[i] = brentq(npv_func, 0.0, hi, maxiter=100, xtol=1e-6)
            elif npv_at_zero < 0:
                # Ztrátový projekt — IRR leží v [LO_LOSS, 0]
                npv_at_lo = npv_func(LO_LOSS)
                if np.isfinite(npv_at_lo) and npv_at_lo > 0:
                    irrs[i] = brentq(npv_func, LO_LOSS, 0.0, maxiter=100, xtol=1e-6)
            # Jinak (NPV(0) > 0 a NPV(hi) > 0): IRR > hi — extrémně ziskové,
            # nezvyklé; nereportujeme.
        except (ValueError, OverflowError, ZeroDivisionError):
            continue

    valid = ~np.isnan(irrs)
    return irrs, valid


def derive_discount_rate_from_irr(irrs, r_f=0.025, sharpe_target=0.4):
    """
    Z distribuce IRR odvodí 3 varianty doporučeného diskontu:

    Vrací dict:
      median       : medián IRR (50 % scénářů ho překoná) — "fair return"
      sharpe_based : r_f + S* × σ(IRR) — Sharpe-based hurdle rate
      var_5        : 5. percentil IRR — 95% pravděpodobnost akceptace
      var_25       : 25. percentil IRR — 75% pravděpodobnost akceptace
      mean         : průměr IRR — očekávaný výnos
      std          : σ(IRR) — volatilita výnosu
      risk_premium : E[IRR] − r_f — implikovaná riziková prémie nad bezrizikovou
    """
    valid_irrs = irrs[~np.isnan(irrs)]
    if len(valid_irrs) == 0:
        return {k: float("nan") for k in
                ("median", "sharpe_based", "var_5", "var_25", "mean", "std", "risk_premium")}

    mean_irr   = float(np.mean(valid_irrs))
    median_irr = float(np.median(valid_irrs))
    std_irr    = float(np.std(valid_irrs))
    return {
        "median":       median_irr,
        "sharpe_based": r_f + sharpe_target * std_irr,
        "var_5":        float(np.percentile(valid_irrs, 5)),
        "var_25":       float(np.percentile(valid_irrs, 25)),
        "mean":         mean_irr,
        "std":          std_irr,
        "risk_premium": mean_irr - r_f,
    }


def ou_price_process(years, n_sim, p0, mu, theta=0.25, sigma=15.0,
                     drift=0.010, correlated_noise=None, clip_lo=35, clip_hi=160):
    """
    Ornstein-Uhlenbeck mean-reverting cenový model s dlouhodobým cenovým driftem.

    Na rozdíl od GBM má cena tendenci vracet se k dlouhodobému průměru (mu),
    což lépe odpovídá chování komoditních trhů s biomasou – při vysoké ceně
    poptávka klesá (substituce jinými palivy), při nízké nabídka klesá.

    Parametry:
      p0     – počáteční cena (EUR/t)
      mu     – výchozí dlouhodobý průměr ceny (EUR/t), „gravitační střed"
      theta  – rychlost návratu k průměru (0.25 = cca 4 roky na polovinu odchylky)
      sigma  – roční volatilita v EUR/t (ne v procentech)
      drift  – roční reálný růst průměrné ceny (default 0.010 = 1.0 %/rok),
               odráží dekarbonizační prémii a rostoucí poptávku po biomase
      correlated_noise – korelovaný šum z Gaussovské kopule (shape: years × n_sim)
      clip_lo, clip_hi – cenové limity (pojistka, ne hlavní regulátor)

    Dle literatury (Hull 2018, Ornstein-Uhlenbeck pro komodity):
      dP = θ(μ(t) − P)dt + σdW,  kde μ(t) = μ₀ · (1 + drift)^t
    Diskrétní verze (dt=1 rok):
      mu_t      = mu * (1 + drift)^t
      P(t+1)    = P(t) + θ(mu_t − P(t)) + σ·ε(t)
    """
    prices = np.zeros((years, n_sim))
    prices[0, :] = p0
    for t in range(1, years):
        mu_t = mu * ((1.0 + drift) ** t)   # rostoucí gravitační střed
        if correlated_noise is not None:
            # 70 % korelovaný šok, 30 % idiosynkratický (zachováno z původního modelu)
            noise = 0.70 * correlated_noise[t, :] + 0.30 * np.random.standard_normal(n_sim)
        else:
            noise = np.random.standard_normal(n_sim)
        prices[t, :] = (prices[t-1, :]
                         + theta * (mu_t - prices[t-1, :])
                         + sigma * noise)
    prices = np.clip(prices, clip_lo, clip_hi)
    return prices


def generate_correlated_shocks(years, n_sim, rho=0.0):
    """
    Korelované uniformní veličiny (Choleskyho dekompozice + Φ-transformace).
    rho = 0 → nezávislé počasí a cena (default po revizi 2026).
    rho < 0 → natural hedge (vysoký výnos → nižší cena).
    """
    L     = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z_cor = L @ np.random.standard_normal((2, years * n_sim))
    u     = norm.cdf(z_cor)
    return u[0].reshape(years, n_sim), u[1].reshape(years, n_sim)


def apply_catastrophic_scenario(wm, n_sim, years, duration=10,
                                factor_range=(0.40, 0.60),
                                start_window=None):
    """
    Katastrofický scénář: pro každou MC simulaci náhodně vybere
    interval `duration` po sobě jdoucích let, kdy je výnosový multiplikátor
    v každém roce nezávisle redukován na náhodný faktor v rozsahu
    `factor_range` (default 40--60 %).

    Modeluje dlouhotrvající nepříznivé klimatické období (sucho, opakované
    extrémní teploty) s vnitrodekádní variabilitou intenzity — každý rok
    v zasaženém intervalu má vlastní nezávislý redukční faktor, což lépe
    odráží reálnou agronomickou variabilitu sucha (rok co rok jiná
    intenzita), než kdyby všechny roky měly stejnou redukci.

    `start_window` : (min_start, max_start) — povolený rozsah startovního
        roku. Default None = (0, years - duration), tj. kdekoli kde se
        celé okno vejde do horizontu. Pro restrikci na produktivní fázi
        plantáže lze předat užší okno (např. (3, years - duration - 3)
        pro SRC se setupem 0--2 a likvidací posledních 3 roky).

    Garantuje, že vybraný interval bude **celých `duration` let**
    a~vejde se v~horizontu (nezačíná v~předposledním roce ap.).
    Modifikuje `wm` in-place a vrací jej.
    """
    if duration > years:
        return wm  # horizont je kratší než katastrofa

    if start_window is None:
        min_start, max_start = 0, years - duration
    else:
        min_start, max_start = start_window
        # Bezpečnostní oříznutí, aby se okno vždy celé vešlo
        max_start = min(max_start, years - duration)
        min_start = max(0, min_start)
    if max_start < min_start:
        return wm  # neexistuje validní startovní okno

    # Pro každý scénář: náhodný startovní rok
    starts = np.random.randint(min_start, max_start + 1, size=n_sim)

    # Nezávislé redukční faktory: per rok × scénář (ne jeden per scénář)
    factors_per_year = np.random.uniform(
        factor_range[0], factor_range[1], size=(years, n_sim)
    )

    # Vektorizovaná maska: (years, n_sim) — True kde patří do katastrofy
    t_grid = np.arange(years)[:, None]                     # (years, 1)
    mask   = (t_grid >= starts) & (t_grid < starts + duration)
    # Multiplikátor: nezávislý factor v zasažených letech, 1.0 jinde
    cat_mult = np.where(mask, factors_per_year, 1.0)
    wm *= cat_mult
    return wm


def simulate_miscanthus(n_sim, years, params, yield_bounds, subsidy_perc,
                        weather_prob=0.05, rho=0.0, drift=0.010,
                        cost_escalation=0.02, pachtovne=0.0,
                        catastrophic=False, sigma_P=20.0):
    """
    Miscanthus × giganteus – simulace s časově rozloženým investičním nákladem.

    Časová struktura (1-indexed jako v textu DP, mapováno na cf[X]):
      ROK 0 = cf[0]   – příprava pozemku (200 EUR/ha)
      ROK 1 = cf[1]   – sadba + mech. výsadba + údržba 1. roku
                        (riziko selhání plantáže)
      ROK 2 = cf[2]   – údržba 2. roku
      ROK 3..(years-1) = produkce: výnos × cena − roční údržba − sklizeň
      ROK (years-1)   – navíc náklady na likvidaci plantáže (orba)

    Roční náklady (daň z pozemku, pachtovné) se účtují ve všech letech.
    """
    years = int(years)

    # ---- Stochastické realizace ------------------------------------------
    y_max_sim = np.clip(np.random.normal(np.mean(yield_bounds),
                        (yield_bounds[1]-yield_bounds[0])/4, n_sim), 0, 60)

    # Korelované šoky pro CELÝ horizont; producci aplikujeme jen na ROK 3+
    # Meziroční variabilita výnosu Miscanthus = 15 % (Knápek 2024,
    # konzultace VÚKOZ Průhonice 2026).
    wu, pu = generate_correlated_shocks(years, n_sim, rho=rho)
    wm     = np.clip(norm.ppf(wu, loc=1.0, scale=0.15), 0.5, 1.5)

    if catastrophic:
        # Katastrofický scénář: 10 po sobě jdoucích špatných let (sucho/extrém)
        # Okno omezeno na produktivní fázi (start ≥ ROK 3 = po setupu) tak,
        # aby celých 10 let padlo do života plantáže.
        wm = apply_catastrophic_scenario(
            wm, n_sim, years, duration=10, factor_range=(0.40, 0.60),
            start_window=(3, years - 10))
    else:
        # Standardní stresové roky (pravděpodobnost weather_prob na rok)
        wm[np.random.rand(years, n_sim) < weather_prob] *= 0.50

    psc = norm.ppf(pu)
    # Volatilita σ_P = 20 EUR/t DM pro Miscanthus – analogicky odvozeno přes
    # konzervativní přirážku 1,20 × σ_SRC nad empirickou hodnotou pro SRC
    # (Kristöfel et al. 2014, Biomass and Bioenergy 65: 112–124, GARCH analýza
    # rakouských dat: agricultural biomass má až 7× vyšší volatilitu než
    # roundwood; volíme dolní mez rozsahu jako konzervativní odhad).
    # σ_SRC^DM = 16,6 EUR/t DM (z OU vzorce; viz simulate_src).
    # Pro Miscanthus: 1,20 × 16,6 ≈ 20 EUR/t DM.
    # Detail v sekci 3.5.2 DP. σ_P^M je v citlivostní analýze (sekce 3.8.2).
    prices = ou_price_process(
        years, n_sim,
        p0=params["prodejni_cena_start"],
        mu=params["prodejni_cena_start"],
        theta=0.25, sigma=sigma_P, drift=drift,
        correlated_noise=psc, clip_lo=55, clip_hi=200,
    )

    # Růstová křivka dle Weger 2021 pokrývá ROK 1..25 (profile[0..24]):
    #   profile[0] = 0.00 (ROK 1 = výsadba, žádný harvest)
    #   profile[1] = 0.03 (ROK 2 = drobný early biomass ~3 % peaku)
    #   profile[2] = 0.16 (ROK 3 = první komerční sklizeň)
    #   profile[3..] = ramp-up + plateau + úpadek
    # CAPEX zůstává v ROK 0–2 (n_setup_capex = 3); drobný výnos v ROK 2
    # se řeší zvlášť níže (kombinace CAPEX + tržby).
    n_setup_capex = 3                            # ROK 0, 1, 2 = CAPEX fáze
    n_prod        = max(0, years - n_setup_capex)
    yields        = np.zeros((years, n_sim))
    if n_prod > 0:
        # Produkční fáze ROK 3..(years-1): t=3 → profile[2]=0.16
        t_prod       = np.arange(n_setup_capex, years, dtype=float)
        growth_curve = gompertz_growth(t_prod)
        yields[n_setup_capex:, :] = (growth_curve[:, np.newaxis]
                                      * y_max_sim[np.newaxis, :]
                                      * wm[n_setup_capex:, :])

    # ROK 2: drobný výnos ~3 % peaku dle Weger 2021 (Y2)
    if years > 2:
        yields[2, :] = (MISCANTHUS_2021_PROFILE[1]
                        * y_max_sim
                        * wm[2, :])

    # ---- Eskalace provozních nákladů -------------------------------------
    esc_factor = (1.0 + cost_escalation) ** np.arange(years, dtype=float)
    udrzba_t   = params["udrzba_rocni"]      * esc_factor   # shape (years,)
    sklizen_t  = params["sklizen_per_tuna"]  * esc_factor

    # ---- Cash flow ------------------------------------------------------
    cf    = np.zeros((years, n_sim))
    # Nákladová trajektorie (kladné hodnoty = výdaje), souběžně s cf
    # Rozkládá se na 3 kategorie pro grafický rozklad:
    costs_setup   = np.zeros((years, n_sim))   # CAPEX, sadba, výsadba, příprava
    costs_oper    = np.zeros((years, n_sim))   # údržba + sklizeň + fixní (daň + pacht)
    costs_liquid  = np.zeros((years, n_sim))   # likvidace plantáže

    # Roční fixní náklady (daň z pozemku + případné pachtovné) – ve všech letech
    fixed_yearly = params.get("dan_pozemku", 0.0) + pachtovne

    # ROK 0: jen příprava pozemku + fixní náklady, bez dotace, bez výnosu
    cf[0, :] = -(params["prep_pozemku"] + fixed_yearly)
    costs_setup[0, :] = params["prep_pozemku"]
    costs_oper[0, :]  = fixed_yearly

    # ROK 1: sadba + výsadba + údržba 1. roku (= polovina udrzba_1_2_rok)
    # Sadební materiál i mechanizovaná výsadba se škálují lineárně s hustotou výsadby
    # (jednotná modelová formule pro obě plodiny — viz § 3.X DP).
    sadebni_material = params["hustota_vysadby"] * params["cena_sadby_ks"]
    mech_vysadba     = params["hustota_vysadby"] * params["cena_vysadby_ks"]
    udrzba_y12_half  = params["udrzba_1_2_rok"] / 2.0
    capex_y1_base    = sadebni_material + mech_vysadba + udrzba_y12_half
    capex_y1         = np.random.normal(capex_y1_base,
                                         params.get("capex_std", 300.0), n_sim)
    cf[1, :] = -capex_y1 * (1.0 - subsidy_perc) - fixed_yearly
    costs_setup[1, :] = capex_y1 * (1.0 - subsidy_perc)
    costs_oper[1, :]  = fixed_yearly

    # ROK 2: údržba 2. roku + fixní náklady + drobný výnos (~3 % peaku)
    if years > 2:
        rev_rok2 = yields[2, :] * prices[2, :]
        sklizen_cost_rok2 = yields[2, :] * sklizen_t[2]
        cf[2, :] = (-udrzba_y12_half - fixed_yearly
                    + rev_rok2 - sklizen_cost_rok2)
        costs_oper[2, :] = udrzba_y12_half + fixed_yearly + sklizen_cost_rok2

    # ROK 3..(years-1): produkce
    if n_prod > 0:
        prod_revenue = yields[n_setup_capex:, :] * prices[n_setup_capex:, :]
        prod_costs   = (udrzba_t[n_setup_capex:, np.newaxis]
                         + yields[n_setup_capex:, :] * sklizen_t[n_setup_capex:, np.newaxis])
        cf[n_setup_capex:, :] = prod_revenue - prod_costs - fixed_yearly
        costs_oper[n_setup_capex:, :] = prod_costs + fixed_yearly

    # ROK (years-1): navíc náklady na likvidaci plantáže
    cf[-1, :] -= params["likvidace"]
    costs_liquid[-1, :] = params["likvidace"]

    # ---- Selhání plantáže (decided in ROK 1) -----------------------------
    fail   = np.random.binomial(1, params["riziko_fail"], n_sim)
    renew  = np.random.binomial(1, 0.5, n_sim)

    # Obnova: extra +60 % CAPEX_y1. Dotace se aplikuje POUZE na původní založení,
    # nikoli na obnovu po selhání (pěstitel nese plné riziko replantáže).
    replant_cost = fail * renew * 0.60 * capex_y1
    cf[1, :] -= replant_cost
    costs_setup[1, :] += replant_cost

    # Opuštění: ROK 0 a ROK 1 jsou sunk costs (zaplaceno před zjištěním selhání).
    # Od ROK 2 dál žádné CF (žádný příjem, žádné další náklady).
    abandoned = (fail == 1) & (renew == 0)
    if years > 2:
        cf[2:, abandoned] = 0.0
        costs_setup[2:, abandoned]  = 0.0
        costs_oper[2:, abandoned]   = 0.0
        costs_liquid[2:, abandoned] = 0.0

    costs = {"setup": costs_setup, "oper": costs_oper, "liquid": costs_liquid}
    return cf, yields, prices, costs


def simulate_src(n_sim, years, params, yield_bounds, tech_type, subsidy_perc,
                 weather_prob=0.05, rho=0.0, drift=0.010,
                 cost_escalation=0.02, pachtovne=0.0,
                 soil_quality="Průměrná", catastrophic=False, sigma_P=16.6):
    """
    SRC vrba – simulace s časově rozloženým investičním nákladem a 3-letou likvidací.

    Časová struktura (mapováno na cf[X]):
      ROK 0  – příprava pozemku (230 EUR/ha)
      ROK 1  – sadba + mech. výsadba + údržba 1. roku (riziko selhání)
      ROK 2  – údržba 2. roku
      ROK 3..4 – jen roční údržba (60 EUR/ha), žádná sklizeň
      ROK 5, 8, 11, 14, 17, 20, 23, 26 – sklizeň + údržba
      ROK 27, 28 – likvidace (mulčování / sečení), bez výnosu, bez údržby
      ROK 29 – konečná orba

    Likvidace celkem 620 EUR/ha rozdělena ≈ 189 + 189 + 240 v ROK 27/28/29.
    Roční fixní náklady (daň + pachtovné) se účtují v letech 0..(years-1)
    s výjimkou likvidačních let (27, 28), kde se předpokládá, že na pozemku
    jen probíhá rozklad a samostatný náklad na nájem je již zahrnut v "likvidace".
    """
    years = int(years)

    y_max_sim = np.clip(np.random.normal(np.mean(yield_bounds),
                        (yield_bounds[1]-yield_bounds[0])/4, n_sim), 0, 40)

    # Korelované šoky pro celý horizont
    # Meziroční variabilita výnosu SRC vrby = 10 % (Knápek 2024,
    # konzultace VÚKOZ Průhonice 2026; nižší než Miscanthus díky
    # robustnějšímu kořenovému systému dřevin).
    wu, pu = generate_correlated_shocks(years, n_sim, rho=rho)
    wm     = np.clip(norm.ppf(wu, loc=1.0, scale=0.10), 0.5, 1.5)

    if catastrophic:
        # Katastrofický scénář: 10 po sobě jdoucích špatných let (sucho/extrém)
        # Okno omezeno na produktivní fázi: start ≥ ROK 3 (po setupu) a
        # zároveň konec ≤ ROK (years − 4) tak, aby celých 10 let padlo
        # mimo 3 likvidační roky (poslední 3 roky horizontu).
        n_liq = 3
        wm = apply_catastrophic_scenario(
            wm, n_sim, years, duration=10, factor_range=(0.40, 0.60),
            start_window=(3, years - n_liq - 10))
    else:
        # Standardní stresové roky (pravděpodobnost weather_prob na rok)
        wm[np.random.rand(years, n_sim) < weather_prob] *= 0.50

    # OU cenový proces přes celý horizont (využíváme jen ve sklizňových letech)
    psc = norm.ppf(pu)
    # Volatilita σ_P = 16,6 EUR/t DM pro SRC vrbu – empiricky odvozeno z 21
    # čtvrtletí ČSÚ palivového dříví VI. třídy (2019Q1–2025Q1):
    #   σ_eq^wet ≈ 18,8 EUR/t mokré (mean 92,5 EUR/t, palivové dříví ~20 % vlhkosti)
    # Přepočet na DM basis (vlhkost ~20 %):
    #   σ_eq^DM = 18,8 / 0,80 ≈ 23,5 EUR/t DM
    # Striktně z OU vzorce: σ_P^DM = √(2θ · σ²_eq) = √(0,5 · 23,5²) ≈ 16,6 EUR/t DM
    # Cross-check (1. diference, DM basis): 14 / 0,80 ≈ 17,5 EUR/t DM
    # V modelu volíme striktní hodnotu 16,6 (formula-derived) jako referenční;
    # cross-check ji přibližně potvrzuje.
    # Detail v sekci 3.5.2 DP.
    prices = ou_price_process(
        years, n_sim,
        p0=params["prodejni_cena_start"],
        mu=params["prodejni_cena_start"],
        theta=0.25, sigma=sigma_P, drift=drift,
        correlated_noise=psc, clip_lo=60, clip_hi=220,
    )

    # Eskalace nákladů
    esc_factor = (1.0 + cost_escalation) ** np.arange(years, dtype=float)
    udrzba_t   = params["udrzba_rocni"]      * esc_factor
    sklizen_t  = params["sklizen_per_tuna"]  * esc_factor

    # ---- Aktivní růst pouze v letech 3 až (years-4) -----------------------
    # ROK 0,1,2 = setup (žádný růst), ROK (years-3),(years-2),(years-1) = likvidace
    n_setup = 3
    n_liq   = 3                                  # ROK 27, 28, 29 pro zivotnost=30
    n_growth = max(0, years - n_setup - n_liq)

    annual_growth = np.zeros((years, n_sim))
    if n_growth > 0:
        t_growth = np.arange(1, n_growth + 1, dtype=float)
        curve    = src_yield_curve(t_growth, soil_quality=soil_quality)  # shape (n_growth,)
        annual_growth[n_setup:n_setup + n_growth, :] = (
            curve[:, np.newaxis]
            * y_max_sim[np.newaxis, :]
            * wm[n_setup:n_setup + n_growth, :]
        )

    # ---- Sklizňové roky (1-indexed v textu DP, 0-indexed v cf) -----------
    # Sklizňové ROKy: 5, 8, 11, 14, 17, 20, 23, 26 → cf indexy 5, 8, ..., 26
    harvest_years = {y for y in range(5, years - n_liq, 3)}

    # Akumulace mezi sklizněmi (jen z let aktivního růstu)
    harvested = np.zeros((years, n_sim))
    accum = np.zeros(n_sim)
    for t in range(years - n_liq):                # nepokračuj do likvidace
        accum += annual_growth[t, :]
        if t in harvest_years:
            harvested[t, :] = accum
            accum = np.zeros(n_sim)

    # ---- Cash flow ------------------------------------------------------
    cf    = np.zeros((years, n_sim))
    # Nákladová trajektorie (kladné hodnoty = výdaje)
    costs_setup  = np.zeros((years, n_sim))
    costs_oper   = np.zeros((years, n_sim))
    costs_liquid = np.zeros((years, n_sim))

    fixed_yearly = params.get("dan_pozemku", 0.0) + pachtovne

    # ROK 0: příprava pozemku + fixní
    cf[0, :] = -(params["prep_pozemku"] + fixed_yearly)
    costs_setup[0, :] = params["prep_pozemku"]
    costs_oper[0, :]  = fixed_yearly

    # ROK 1: sadba + výsadba + údržba 1. roku + fixní
    sadebni_material = params["hustota_vysadby"] * params["cena_sadby_ks"]
    udrzba_y12_half  = params["udrzba_1_2_rok"] / 2.0
    capex_y1_base    = sadebni_material + params["mech_vysadba"] + udrzba_y12_half
    capex_y1         = np.random.normal(capex_y1_base,
                                         params.get("capex_std", 150.0), n_sim)
    cf[1, :] = -capex_y1 * (1.0 - subsidy_perc) - fixed_yearly
    costs_setup[1, :] = capex_y1 * (1.0 - subsidy_perc)
    costs_oper[1, :]  = fixed_yearly

    # ROK 2: údržba 2. roku + fixní
    if years > 2:
        cf[2, :] = -(udrzba_y12_half + fixed_yearly)
        costs_oper[2, :] = udrzba_y12_half + fixed_yearly

    # ROK 3..(years - n_liq - 1): roční údržba + fixní; v sklizňových letech přidat příjem
    for t in range(n_setup, years - n_liq):
        cf[t, :] = -(udrzba_t[t] + fixed_yearly)
        costs_oper[t, :] = udrzba_t[t] + fixed_yearly
        if t in harvest_years:
            v = harvested[t, :]
            cf[t, :] += v * prices[t, :] - v * sklizen_t[t]
            costs_oper[t, :] += v * sklizen_t[t]

    # Likvidační roky 27, 28, 29 (pro zivotnost=30)
    # Podíly: 189/618 ≈ 0,306; 189/618 ≈ 0,306; 240/618 ≈ 0,388
    if years >= n_setup + n_liq:
        liq_total  = params["likvidace"]
        liq_shares = (189.0 / 618.0, 189.0 / 618.0, 240.0 / 618.0)
        for offset, share in enumerate(liq_shares):
            idx = years - n_liq + offset          # cf indices 27, 28, 29
            cf[idx, :] = -liq_total * share       # bez fixed_yearly v likvidaci
            costs_liquid[idx, :] = liq_total * share

    # ---- Selhání plantáže (decided in ROK 1) -----------------------------
    fail   = np.random.binomial(1, params["riziko_fail"], n_sim)
    renew  = np.random.binomial(1, 0.5, n_sim)

    # Obnova: extra +60 % CAPEX_y1. Dotace se aplikuje POUZE na původní založení,
    # nikoli na obnovu po selhání (pěstitel nese plné riziko replantáže).
    replant_cost = fail * renew * 0.60 * capex_y1
    cf[1, :] -= replant_cost
    costs_setup[1, :] += replant_cost

    abandoned = (fail == 1) & (renew == 0)
    if years > 2:
        cf[2:, abandoned] = 0.0
        costs_setup[2:, abandoned]  = 0.0
        costs_oper[2:, abandoned]   = 0.0
        costs_liquid[2:, abandoned] = 0.0

    costs = {"setup": costs_setup, "oper": costs_oper, "liquid": costs_liquid}
    return cf, harvested, prices, costs


def calculate_sensitivity_matrix(crop_type, years, base_params, y_bounds,
                                  src_tech, area_ha, n_sim, subsidy_perc,
                                  rho=0.0, drift=0.010, cost_escalation=0.02,
                                  pachtovne=0.0, soil_quality="Průměrná"):
    steps = 10
    xr, yr = np.linspace(0, 0.2, steps), np.linspace(0, 0.2, steps)
    zm = []
    for wp in yr:
        row = []
        for fp in xr:
            p = {**base_params, "riziko_fail": fp}
            if crop_type == "misc":
                cf, _, _, _ = simulate_miscanthus(n_sim, years, p, y_bounds, subsidy_perc,
                                                   wp, rho=rho, drift=drift,
                                                   cost_escalation=cost_escalation,
                                                   pachtovne=pachtovne)
            else:
                cf, _, _, _ = simulate_src(n_sim, years, p, y_bounds, src_tech, subsidy_perc,
                                            wp, rho=rho, drift=drift,
                                            cost_escalation=cost_escalation,
                                            pachtovne=pachtovne,
                                            soil_quality=soil_quality)
            row.append(np.mean(np.std(cf * area_ha, axis=0)))
        zm.append(row)
    return xr, yr, np.array(zm)


# ===========================================================================
# DIVERZIFIKAČNÍ OPTIMALIZÁTOR – optimální mix Miscanthus + SRC
# ===========================================================================
DIVERSIFY_METRICS = ["profit", "eaa", "var", "cvar", "std"]   # std minimalizujeme, ostatní maximalizujeme
DIVERSIFY_STEP_PCT = 2                                  # poměr po 2 % → 51 mixů


def _scenario_metrics(cf_total, years, discount_rate):
    """Z matice CF (years × n_sim) na jeden scénář spočítá NPV a EAA."""
    if discount_rate > 0:
        df = (1 + discount_rate) ** -np.arange(years)
        npv = np.sum(cf_total * df[:, np.newaxis], axis=0)
    else:
        npv = np.sum(cf_total, axis=0)
    eaa = equivalent_annual_annuity(npv, years, discount_rate)
    return npv, eaa


def run_diversification(misc_params, misc_y_bounds,
                         src_params,  src_y_bounds,
                         total_area_ha, n_sim, subsidy_perc, discount_rate,
                         rho=0.0, drift=0.010, weather_prob=0.05,
                         cost_escalation=0.02, pachtovne=0.0,
                         step_pct=DIVERSIFY_STEP_PCT,
                         soil_quality="Průměrná"):
    """
    Pro 51 poměrů Miscanthus:SRC (krok 2 %) spočítá distribuci kombinovaného
    portfolia a vrátí všechny metriky pro každý poměr.

    DETERMINISTICKÝ přístup: CF se generuje pouze JEDNOU (jeden seed pro M,
    jeden pro SRC), pak se v cyklu pouze deterministicky kombinují podle
    váhy w_M. To zajišťuje, že:
      - μ portfolia je lineární kombinací μ_M a μ_S (matematicky korektní)
      - CVaR/VaR portfolia jsou stabilní a reprodukovatelné napříč běhy
      - Optimum diverzifikace je matematicky správné, nikoli artefakt MC šumu

    Plocha se rozdělí: area_M = total_area × pct_M, area_S = total_area × (1-pct_M).
    """
    misc_years = int(misc_params["zivotnost"])
    src_years  = int(src_params["zivotnost"])

    pct_grid = np.arange(0, 100 + step_pct, step_pct)   # 0, 2, …, 100
    out = {"pct_misc": pct_grid.tolist(),
           "profit": [], "eaa": [], "var": [], "cvar": [], "std": []}

    # === Generuj CF JEDNOU mimo cyklus (per ha) ===
    # STEJNÝ seed pro M i SRC → fyzicky správné portfolio na jednom poli:
    # obě plodiny "vidí" stejné makro počasí, ceny i klimatický stres.
    # Tím vzniká realistická korelace ρ ≈ 0.4–0.7 mezi plodinami.
    SHARED_SEED = 1_000_003
    np.random.seed(SHARED_SEED)
    cf_m_perha, _, _, _ = simulate_miscanthus(
        n_sim, misc_years, misc_params, misc_y_bounds, subsidy_perc,
        weather_prob, rho=rho, drift=drift,
        cost_escalation=cost_escalation, pachtovne=pachtovne)

    np.random.seed(SHARED_SEED)  # ← stejný seed = sdílené počasí
    cf_s_perha, _, _, _ = simulate_src(
        n_sim, src_years, src_params, src_y_bounds, "Direct Chip",
        subsidy_perc, weather_prob, rho=rho, drift=drift,
        cost_escalation=cost_escalation, pachtovne=pachtovne,
        soil_quality=soil_quality)

    # === Iteruj přes poměry — pouze deterministická kombinace ===
    for pct in pct_grid:
        area_M = total_area_ha * (pct / 100.0)
        area_S = total_area_ha * (1.0 - pct / 100.0)

        cf_m = cf_m_perha * area_M
        cf_s = cf_s_perha * area_S

        npv_m, eaa_m = _scenario_metrics(cf_m, misc_years, discount_rate)
        npv_s, eaa_s = _scenario_metrics(cf_s, src_years, discount_rate)

        # Agregace na úroveň portfolia (sčítáme NPV/EAA per scénář)
        port_npv = npv_m + npv_s
        port_eaa = eaa_m + eaa_s

        # VaR/CVaR počítáme z rozdělení EAA (konzistentně s kap04 a kap05)
        var5  = float(np.percentile(port_eaa, 5))
        mask  = port_eaa <= var5
        cvar5 = float(port_eaa[mask].mean()) if mask.any() else var5

        # Roční odchylka kombinovaného CF (across years, averaged across scenarios)
        max_y  = max(misc_years, src_years)
        cf_pad = np.zeros((max_y, n_sim))
        cf_pad[:misc_years, :] += cf_m
        cf_pad[:src_years,  :] += cf_s
        std_yr = float(np.mean(np.std(cf_pad, axis=0)))

        out["profit"].append(float(np.mean(port_npv)))
        out["eaa"].append(float(np.mean(port_eaa)))
        out["var"].append(var5)
        out["cvar"].append(cvar5)
        out["std"].append(std_yr)

    return out


def find_optimum(div_results, metric_key):
    """
    Najde index optima dle metriky:
      - profit, eaa, var, cvar → maximalizujeme (pro VaR/CVaR = nejméně záporné)
      - std → minimalizujeme (nižší volatilita CF = lepší diverzifikace)
    """
    arr = np.array(div_results[metric_key])
    if metric_key == "std":
        idx = int(np.argmin(arr))
    else:
        idx = int(np.argmax(arr))
    return idx, float(arr[idx])


# ===========================================================================
# HTML TABULKA – zaručené centrování bez závislosti na Streamlit CSS
# ===========================================================================
def _fmt_eur(val):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "N/A"
    s = f"{val:,.0f}".replace(",", "&nbsp;")
    return f"{s}&nbsp;€"

def _fmt_pct(val):
    return f"{val:.1%}" if val >= 0.01 else f"{val:.4%}"

def _fmt_yr(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.2f}"

def render_recap_table(summary_data: list, T: dict) -> str:
    rc   = T["recap_cols"]
    crop = T["recap_crop"]

    col_order = [
        "Průměrný Zisk", "EAA (EUR/rok)", "Potenciál (95%)", "Doba návratnosti (roky)",
        "Pravděpodobnost návratnosti",
        "VaR (5%)", "CVaR (5%)",
        "VaR EAA (5%)", "CVaR EAA (5%)",
        "Prům. roční odchylka",
    ]
    # rc.get(k, k) — fallback na klíč pokud překlad chybí (např. po hot-reloadu)
    headers = [crop] + [rc.get(k, k) for k in col_order]

    TH = ("background:#1e3a5f;color:#fff;padding:11px 18px;"
          "text-align:center;font-weight:600;white-space:nowrap;"
          "border-right:1px solid #2d5282;")
    TD_E = ("padding:9px 16px;text-align:center;"
            "border-bottom:1px solid #dde3ec;background:#f4f7fb;")
    TD_O = ("padding:9px 16px;text-align:center;"
            "border-bottom:1px solid #dde3ec;background:#ffffff;")

    header_html = "".join(f"<th style='{TH}'>{h}</th>" for h in headers)

    rows_html = ""
    for i, row in enumerate(summary_data):
        td = TD_E if i % 2 == 0 else TD_O
        payback = row.get("Doba návratnosti (roky)")
        cells = [
            f"<td style='{td}'><b>{row['Plodina']}</b></td>",
            f"<td style='{td}'>{_fmt_eur(row['Průměrný Zisk'])}</td>",
            f"<td style='{td}'>{_fmt_eur(row['EAA (EUR/rok)'])}</td>",
            f"<td style='{td}'>{_fmt_eur(row['Potenciál (95%)'])}</td>",
            f"<td style='{td}'>{_fmt_yr(payback)}</td>",
            f"<td style='{td}'>{_fmt_pct(row['Pravděpodobnost návratnosti'])}</td>",
            f"<td style='{td}'>{_fmt_eur(row['VaR (5%)'])}</td>",
            f"<td style='{td}'>{_fmt_eur(row['CVaR (5%)'])}</td>",
            f"<td style='{td}'>{_fmt_eur(row.get('VaR EAA (5%)', 0))}</td>",
            f"<td style='{td}'>{_fmt_eur(row.get('CVaR EAA (5%)', 0))}</td>",
            f"<td style='{td}'>{_fmt_eur(row['Prům. roční odchylka'])}</td>",
        ]
        rows_html += f"<tr>{''.join(cells)}</tr>"

    return f"""
<div style="overflow-x:auto;margin-top:14px;">
  <table style="width:100%;border-collapse:collapse;font-size:14px;
                box-shadow:0 2px 10px rgba(0,0,0,0.10);border-radius:8px;overflow:hidden;">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
"""



# ===========================================================================
# VSTUPNI TABULKA NAKLADU - centralovana, bez AG Grid
# ===========================================================================
def render_cost_inputs(crop_key, T, widget_key_prefix):
    cost_keys_map = T["cost_keys"]
    col_label     = T["cost_col_label"]
    defaults      = DEFAULT_COSTS[crop_key]

    header = (
        "<div style='display:grid;grid-template-columns:1fr 1fr;"
        "background:#1e3a5f;border-radius:6px 6px 0 0;'>"
        "<div style='color:#fff;font-weight:600;padding:10px 16px;text-align:center;'>"
        "Parametr</div>"
        "<div style='color:#fff;font-weight:600;padding:10px 16px;text-align:center;"
        "border-left:1px solid #2d5282;'>"
        + col_label +
        "</div></div>"
    )
    st.markdown(header, unsafe_allow_html=True)

    result = {}
    for i, (internal_key, display_label) in enumerate(cost_keys_map.items()):
        default_val = defaults[internal_key]
        bg = "#f4f7fb" if i % 2 == 0 else "#ffffff"
        row_l, row_r = st.columns([1, 1])
        with row_l:
            cell_html = (
                "<div style='background:" + bg + ";padding:8px 16px;"
                "text-align:center;border-bottom:1px solid #dde3ec;"
                "min-height:50px;display:flex;align-items:center;"
                "justify-content:center;'><b>" + display_label + "</b></div>"
            )
            st.markdown(cell_html, unsafe_allow_html=True)
        with row_r:
            is_ratio = internal_key == "riziko_fail"
            is_int   = internal_key == "zivotnost"
            step_val = 0.01 if is_ratio else (1.0 if is_int else 10.0)
            fmt      = "%.2f" if is_ratio else "%.0f"
            val = st.number_input(
                label=display_label,
                value=float(default_val),
                min_value=0.0,
                step=step_val,
                format=fmt,
                key=widget_key_prefix + "_" + internal_key,
                label_visibility="collapsed",
            )
            result[internal_key] = val

    st.markdown(
        "<div style='border-bottom:2px solid #1e3a5f;"
        "border-radius:0 0 6px 6px;margin-bottom:8px;'></div>",
        unsafe_allow_html=True,
    )
    return result


# ===========================================================================
# GLOBÁLNÍ CSS
# ===========================================================================
GLOBAL_CSS = """
<style>
.block-container { padding-top: 1.8rem !important; max-width: 1400px; }

/* Centrovaný titulek + jemná oranžová podtržení */
h1 { color: #1B2733; }
h2 { color: #33691E; border-bottom: 2px solid #E8EAE0; padding-bottom: 6px; margin-top: 1.6rem; }
h3 { color: #1B2733; margin-top: 1.2rem; }

/* Karty jednotlivých konfiguračních kroků */
[data-testid="stContainer"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #E0E2D9 !important;
    border-radius: 14px !important;
    padding: 14px 20px !important;
    background: #FFFFFF !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* Step-badge – kruhový čísel u kroků */
.step-badge {
    display:inline-flex; align-items:center; justify-content:center;
    width: 30px; height: 30px; border-radius: 50%;
    background: #33691E; color: white; font-weight: 700;
    margin-right: 10px; font-size: 14px;
    vertical-align: middle;
}
.step-title {
    font-size: 1.25rem; font-weight: 600; color: #1B2733;
    display: inline-block; vertical-align: middle;
}
.step-sub {
    color: #666; font-size: 0.9rem; margin-left: 40px; margin-top: -4px;
}

/* Hero KPI karta (po Run) */
.hero-card {
    background: linear-gradient(135deg, #33691E 0%, #558B2F 100%);
    color: white; padding: 22px 28px; border-radius: 16px;
    margin: 16px 0 22px 0;
    box-shadow: 0 4px 12px rgba(51,105,30,0.18);
}
.hero-title { font-size: 1.3rem; font-weight: 600; margin-bottom: 14px; opacity: 0.95; }
.hero-grid  { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 18px; }
.hero-item  { }
.hero-label { font-size: 0.78rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 0.5px; }
.hero-value { font-size: 1.6rem; font-weight: 700; margin-top: 2px; }
.hero-sub   { font-size: 0.78rem; opacity: 0.75; margin-top: 2px; }

/* Tlačítka vlajek – menší, bez nadbytečného odsazení */
[data-testid="column"]:last-child [data-testid="stButton"] button {
    font-size: 1.4rem !important;
    padding: 2px 8px !important;
    min-width: 0 !important;
    line-height: 1.3 !important;
    border-radius: 6px !important;
}

/* Primární tlačítka výraznější */
.stButton > button[kind="primary"] {
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(51,105,30,0.25) !important;
}

/* Tabs – výraznější, s ikonami */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #E0E2D9; }
.stTabs [data-baseweb="tab"] {
    padding: 10px 18px;
    font-weight: 600;
    font-size: 0.95rem;
    border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    background: #F0EFEA !important;
    color: #33691E !important;
}

/* AG Grid centrování */
.ag-header-cell-label { justify-content: center !important; }
.ag-header-cell-text  { text-align: center !important; }
.ag-cell {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}
.ag-cell-value { width: 100% !important; text-align: center !important; }
.ag-cell input,
.ag-cell .ag-input-field-input { text-align: center !important; }
</style>
"""


def step_header(num, title, subtitle=""):
    """Vykreslí číslovaný krok ve stylu 'badge + nadpis'."""
    sub_html = f"<div class='step-sub'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"<div><span class='step-badge'>{num}</span>"
        f"<span class='step-title'>{title}</span></div>{sub_html}",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# BPEJ TEXTY (přidány inline – nejsou v JSON kvůli dynamickému obsahu)
# ---------------------------------------------------------------------------
BPEJ_STRINGS = {
    "cs": {
        "bpej_spinner":     "Stahuji data o bonitě půdy (BPEJ)...",
        "bpej_success":     "🌱 Bonita půdy automaticky určena z BPEJ",
        "bpej_outside":     "📍 Souřadnice je mimo ČR – kvalitu půdy vyberte ručně.",
        "bpej_nodata":      "⚠️ BPEJ data nenalezena (nezem. půda?) – vyberte ručně.",
        "bpej_error":       "⚠️ BPEJ nelze načíst ({err}) – vyberte ručně.",
        "bpej_unavailable": "ℹ️ Modul bpej_lookup není dostupný – vyberte ručně.",
        "bpej_metric_label": "Bonita půdy (BPEJ)",
        "bpej_metric_na":    "Mimo ČR",
        "bpej_metric_noag":  "Nezem. půda",
        "bpej_metric_err":   "Nedostupné",
        "bpej_metric_cap_ok":  "Automaticky z BPEJ/VÚMOP",
        "bpej_metric_cap_out": "Bod mimo ČR – vyberte ručně",
        "bpej_metric_cap_no":  "Nezem. půda – vyberte ručně",
        "bpej_metric_cap_err": "Chyba načítání – vyberte ručně",
        "climate_zone_label":  "Klimatické pásmo",
        "climate_zone_cap":    "Automaticky dle GPS a klimatických dat",
    },
    "en": {
        "bpej_spinner":     "Downloading soil quality data (BPEJ)...",
        "bpej_success":     "🌱 Soil fertility automatically determined from BPEJ",
        "bpej_outside":     "📍 Coordinate is outside CZ – please select soil quality manually.",
        "bpej_nodata":      "⚠️ No BPEJ data found (non-agricultural land?) – select manually.",
        "bpej_error":       "⚠️ BPEJ unavailable ({err}) – select manually.",
        "bpej_unavailable": "ℹ️ bpej_lookup module not available – select manually.",
        "bpej_metric_label": "Soil quality (BPEJ)",
        "bpej_metric_na":    "Outside CZ",
        "bpej_metric_noag":  "Non-agric.",
        "bpej_metric_err":   "Unavailable",
        "bpej_metric_cap_ok":  "Auto-detected from BPEJ/VÚMOP",
        "bpej_metric_cap_out": "Outside CZ – select manually",
        "bpej_metric_cap_no":  "Non-agric. land – select manually",
        "bpej_metric_cap_err": "Load error – select manually",
        "climate_zone_label":  "Climate Zone",
        "climate_zone_cap":    "Auto-detected from GPS and climate data",
    },
}

# ===========================================================================
# SESSION STATE – jazyk
# ===========================================================================
if "lang" not in st.session_state:
    st.session_state["lang"] = "cs"

# ===========================================================================
# PAGE CONFIG
# ===========================================================================
st.set_page_config(layout="wide", page_title="Rizikový model biomasy")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ===========================================================================
# HORNÍ LIŠTA – nadpis + vlajkové přepínače vpravo
# ===========================================================================
T = LANG[st.session_state["lang"]]

col_ttl, col_flag = st.columns([8, 1])

with col_ttl:
    st.title(T["app_title"])
    st.markdown(T["app_subtitle"])

with col_flag:
    st.markdown("<div style='display:flex;flex-direction:row;justify-content:flex-end;align-items:center;gap:4px;padding-top:22px;'>", unsafe_allow_html=True)
    cs_active = st.session_state["lang"] == "cs"
    en_active = st.session_state["lang"] == "en"
    if st.button("🇨🇿", key="btn_cs", help="Čeština",
                 type="primary" if cs_active else "secondary"):
        st.session_state["lang"] = "cs"
        st.rerun()
    if st.button("🇬🇧", key="btn_en", help="English",
                 type="primary" if en_active else "secondary"):
        st.session_state["lang"] = "en"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Přegeneruj T po možném rerun
T = LANG[st.session_state["lang"]]

st.markdown("---")

# ===========================================================================
# KROK 1 – LOKALITA & KLIMA (vše uvnitř karty)
# ===========================================================================
detected_zone_key = "Střední Evropa / Mírné pásmo"
real_rain, real_temp = 600.0, 10.0

if "bpej_result"      not in st.session_state: st.session_state["bpej_result"] = None
if "bpej_soil_key"    not in st.session_state: st.session_state["bpej_soil_key"] = None
if "last_clicked_pos" not in st.session_state: st.session_state["last_clicked_pos"] = None

BS = BPEJ_STRINGS[st.session_state["lang"]]
lang_code = st.session_state["lang"]

with st.container(border=True):
    step_header(1, T["sec1_header"], T["sec1_desc"])

    fmap = folium.Map(location=[49.8, 15.5], zoom_start=8)
    fmap.add_child(folium.LatLngPopup())
    map_data = st_folium(fmap, height=420, use_container_width=True)

    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        current_pos = (round(lat, 5), round(lon, 5))

        if current_pos != st.session_state["last_clicked_pos"]:
            st.session_state["last_clicked_pos"] = current_pos
            st.session_state["bpej_result"]      = None
            st.session_state["bpej_soil_key"]    = None

        st.success(f"{T['coord_success']} {lat:.4f}, {lon:.4f}")

        with st.spinner(T["spinner_climate"]):
            tv, rv = get_climate_data(lat, lon)
            if tv is not None:
                real_temp, real_rain = tv, rv
                detected_zone_key = determine_zone(lat, real_temp, real_rain)

        if st.session_state["bpej_result"] is None:
            if BPEJ_AVAILABLE:
                with st.spinner(BS["bpej_spinner"]):
                    bpej_res = get_fertility_from_coords(lat, lon)
                    st.session_state["bpej_result"] = bpej_res
                    if bpej_res["fertility"] is not None:
                        st.session_state["bpej_soil_key"] = bpej_res["fertility"]
            else:
                st.session_state["bpej_result"] = {"source": "unavailable", "fertility": None, "error": None, "bpej_code": None, "bpej_decoded": None}

    detected_zone_label = T["zones"][detected_zone_key]

    bpej_res  = st.session_state.get("bpej_result")
    bpej_src  = bpej_res.get("source") if bpej_res else None
    bpej_fert = bpej_res.get("fertility") if bpej_res else None
    bpej_code = bpej_res.get("bpej_code") if bpej_res else None

    if bpej_src == "BPEJ/vumop" and bpej_fert:
        bpej_fmt     = f"{bpej_code[0]}.{bpej_code[1:3]}.{bpej_code[3:]}" if bpej_code else ""
        bpej_value   = f"{bpej_fert}  ({bpej_fmt})" if bpej_fmt else bpej_fert
        bpej_caption = BS["bpej_metric_cap_ok"]
        _bpej_to_soil = {"Velmi úrodná": "Optimální", "Úrodná": "Průměrná", "Neúrodná": "Neúrodná"}
        _mapped = _bpej_to_soil.get(bpej_fert)
        if _mapped and _mapped in SOIL_KEYS:
            st.session_state["bpej_soil_key"] = _mapped
    elif bpej_src == "mimo_CR":
        bpej_value, bpej_caption = BS["bpej_metric_na"], BS["bpej_metric_cap_out"]
    elif bpej_src == "chyba":
        err_msg = (bpej_res.get("error") or "")
        if "nezem" in err_msg.lower() or "nenalezeno" in err_msg.lower():
            bpej_value, bpej_caption = BS["bpej_metric_noag"], BS["bpej_metric_cap_no"]
        else:
            bpej_value, bpej_caption = BS["bpej_metric_err"], BS["bpej_metric_cap_err"]
    else:
        bpej_value   = "—"
        bpej_caption = T["bpej_click_map"]

    # 4 metriky pod mapou – sjednocený styl
    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
    ci1, ci2, ci3, ci4 = st.columns(4)
    ci1.metric(T["metric_temp"], f"{real_temp:.1f} °C")
    ci1.caption(T["metric_temp_cap"])
    ci2.metric(T["metric_rain"], f"{real_rain:.0f} mm")
    ci2.caption(T["metric_rain_cap"])
    ci3.markdown(f"""
<div style="font-size:12px;color:#666;font-weight:500;margin-bottom:2px">{BS["climate_zone_label"]}</div>
<div style="font-size:18px;font-weight:600;line-height:1.3;color:#33691E">{detected_zone_label}</div>
<div style="font-size:12px;color:#888;margin-top:3px">{BS["climate_zone_cap"]}</div>
""", unsafe_allow_html=True)
    ci4.markdown(f"""
<div style="font-size:12px;color:#666;font-weight:500;margin-bottom:2px">{BS["bpej_metric_label"]}</div>
<div style="font-size:18px;font-weight:600;line-height:1.3;color:#33691E">{bpej_value}</div>
<div style="font-size:12px;color:#888;margin-top:3px">{bpej_caption}</div>
""", unsafe_allow_html=True)

# ===========================================================================
# KROK 2 – PLODINY & PARAMETRY (vše v 1 kartě)
# ===========================================================================
with st.container(border=True):
    step_header(2, T["sec2_header"], T["sec2_subtitle"])

    # Řada 1: půda + plocha + plodiny
    cp1, cp2, cp3 = st.columns([1, 1, 1.2])
    with cp1:
        _bpej_key = st.session_state.get("bpej_soil_key")
        _default_soil_idx = (SOIL_KEYS.index(_bpej_key)
                             if _bpej_key and _bpej_key in SOIL_KEYS else 2)
        soil_idx = st.selectbox(T["soil_quality"], range(4),
                                 format_func=lambda i: T["soil_opts"][i],
                                 index=_default_soil_idx)
        soil_key = SOIL_KEYS[soil_idx]
    with cp2:
        plocha_ha = st.number_input(T["area_label"], min_value=1.0, value=10.0, step=1.0)
    with cp3:
        st.write(T["crops_label"])
        ck1, ck2 = st.columns(2)
        with ck1:
            chk_gig = st.checkbox(T["crop_misc_chk"], value=True)
        with ck2:
            chk_src = st.checkbox(T["crop_src_chk"])
        plodiny = []
        if chk_gig: plodiny.append("M_giganteus")
        if chk_src: plodiny.append("SRC Vrba")

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    # Řada 2: stochastické a finanční parametry – 4 sloupce
    sp1, sp2, sp3, sp4 = st.columns(4)
    with sp1:
        n_sim = int(st.number_input(T["nsim_label"], min_value=1, max_value=50000,
                                     value=10000, step=1000))
    with sp2:
        discount_pct = st.number_input(
            T.get("discount_label", "Diskontní sazba (%)"),
            min_value=0.0, max_value=20.0, value=0.0, step=0.1, format="%.1f",
            help=T.get("discount_help",
                       "Reálná diskontní sazba pro výpočet NPV. 0 % = bez diskontování."),
        )
    with sp3:
        rho_input = st.number_input(
            T.get("rho_label", "Korelace počasí ↔ cena (ρ)"),
            min_value=-1.0, max_value=1.0, value=0.0, step=0.05, format="%.2f",
            help=T.get("rho_help",
                       "Záporná = natural hedge. 0 = nezávislé. Default 0 (revize 2026)."),
        )
    with sp4:
        drift_pct = st.number_input(
            T.get("drift_label", "Drift ceny (%/rok)"),
            min_value=-5.0, max_value=5.0, value=1.0, step=0.1, format="%.1f",
            help=T.get("drift_help",
                       "Roční reálný růst dlouhodobé ceny biomasy."),
        )
    drift_rate = drift_pct / 100.0

    # Slider dotace + input eskalace nákladů
    sub_col, esc_col = st.columns([3, 1])
    with sub_col:
        subsidy_pct = st.slider(T["subsidy_label"], 0, 100, 0, 5)
    with esc_col:
        cost_esc_pct = st.number_input(
            T.get("cost_esc_label", "Eskalace nákladů (%/rok)"),
            min_value=0.0, max_value=10.0, value=2.0, step=0.1, format="%.1f",
            help=T.get("cost_esc_help",
                       "Roční růst provozních nákladů (údržba, sklizeň). "
                       "Default 2 % ≈ inflace."),
        )
    cost_escalation_rate = cost_esc_pct / 100.0

    # Katastrofický scénář – 10 po sobě jdoucích nepříznivých let v každé sim.
    catastrophic_input = st.checkbox(
        T["catastrophic_label"],
        value=False,
        key="catastrophic_scenario",
        help=T["catastrophic_help"],
    )

    # Vlastnictví pozemku → pachtovné
    own_col, pacht_col = st.columns([1, 2])
    with own_col:
        land_own_label = T.get("land_own_label", "Pozemek")
        land_options   = T.get("land_own_opts",
                                ["🏡 Vlastní pozemek", "🤝 Pronajatý pozemek"])
        land_choice = st.radio(land_own_label, options=land_options,
                                index=0, horizontal=True, key="land_own_radio")
    with pacht_col:
        is_rented = (land_choice == land_options[1])
        if is_rented:
            pachtovne = st.slider(
                T.get("pachtovne_label", "Pachtovné (EUR/ha/rok)"),
                min_value=120, max_value=280, value=200, step=10,
                help=T.get("pachtovne_help",
                           "Roční nájem zemědělské půdy. Typicky 120–280 EUR/ha "
                           "v ČR (ČSÚ 2024)."),
            )
        else:
            st.markdown(
                "<div style='padding-top:32px;color:#666;font-size:13px'>"
                + T["land_owned_msg"] + "</div>",
                unsafe_allow_html=True,
            )
            pachtovne = 0

# ===========================================================================
# KROK 3 – NÁKLADY (sekcionovaný formulář s tooltipy)
# ===========================================================================
editable_costs = {}
show_misc = "M_giganteus" in plodiny
show_src  = "SRC Vrba"    in plodiny


def _render_crop_costs(crop_label, crop_key, color_dark, color_main):
    """Vykreslí 4 sekce nákladů pro plodinu a vrátí dict s parametry."""
    defaults = DEFAULT_COSTS[crop_label]
    p = {}
    is_misc = (crop_key == "M")

    st.markdown(
        f"<h3 style='text-align:center;color:{color_dark};margin-top:0'>"
        f"{'🌾' if is_misc else '🌳'} {crop_label}</h3>",
        unsafe_allow_html=True)

    # ---------- Sekce 1: Příprava + výsadba (jednorázově) ----------
    st.markdown(
        f"<div style='background:{color_main};color:white;padding:6px 12px;"
        f"border-radius:6px;font-weight:600;font-size:12px;margin-bottom:6px'>"
        f"{T['cost_sec1_header']}</div>", unsafe_allow_html=True)
    p["prep_pozemku"] = st.number_input(
        T["cost_prep_label"],
        value=float(defaults["prep_pozemku"]), min_value=0.0, step=10.0, format="%.0f",
        help=T["cost_prep_help"],
        key=f"{crop_key}_prep")

    # Hustota a cena sadby ⇒ vypočtený sadební materiál
    dens_min, dens_max, dens_step = (6000, 10000, 500) if is_misc else (6000, 8000, 500)
    price_min, price_max, price_step = (0.10, 0.70, 0.02) if is_misc else (0.08, 0.20, 0.01)
    price_fmt = "%.2f"
    plant_help = T["cost_density_misc_help"] if is_misc else T["cost_density_src_help"]
    price_help = T["cost_seed_price_misc_help"] if is_misc else T["cost_seed_price_src_help"]
    dens_label = T["cost_density_misc_label"] if is_misc else T["cost_density_src_label"]
    seed_price_label = (T["cost_seed_price_misc_label"] if is_misc
                        else T["cost_seed_price_src_label"])

    h1, h2 = st.columns(2)
    with h1:
        p["hustota_vysadby"] = st.slider(
            dens_label,
            min_value=dens_min, max_value=dens_max,
            value=int(defaults["hustota_vysadby"]), step=dens_step,
            help=plant_help, key=f"{crop_key}_dens")
    with h2:
        p["cena_sadby_ks"] = st.slider(
            seed_price_label,
            min_value=price_min, max_value=price_max,
            value=float(defaults["cena_sadby_ks"]), step=price_step,
            format=price_fmt, help=price_help, key=f"{crop_key}_pks")

    sadebni_material = p["hustota_vysadby"] * p["cena_sadby_ks"]
    st.markdown(
        f"<div style='font-size:13px;color:#555;padding:4px 0 8px 0'>"
        f"{T['cost_seed_total_label']} <b>{sadebni_material:,.0f} €/ha</b> "
        f"({p['hustota_vysadby']:,} × {p['cena_sadby_ks']:.2f} €/ks)"
        f"</div>".replace(",", " "),
        unsafe_allow_html=True)

    p["cena_vysadby_ks"] = st.number_input(
        T["cost_planting_price_label"],
        value=float(defaults["cena_vysadby_ks"]),
        min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
        help=T["cost_planting_price_help"],
        key=f"{crop_key}_cvys")

    mech_vysadba_calc = p["hustota_vysadby"] * p["cena_vysadby_ks"]
    p["mech_vysadba"]  = mech_vysadba_calc   # pro zpětnou kompatibilitu
    st.markdown(
        f"<div style='font-size:13px;color:#555;padding:4px 0 8px 0'>"
        f"{T['cost_mech_planting_total_label']} <b>{mech_vysadba_calc:,.0f} €/ha</b> "
        f"({p['hustota_vysadby']:,} × {p['cena_vysadby_ks']:.2f} €/ks)"
        f"</div>".replace(",", " "),
        unsafe_allow_html=True)
    p["udrzba_1_2_rok"] = st.number_input(
        T["cost_udrzba_y12_label"],
        value=float(defaults["udrzba_1_2_rok"]), min_value=0.0, step=10.0, format="%.0f",
        help=T["cost_udrzba_y12_help"],
        key=f"{crop_key}_u12")

    # ---------- Sekce 2: Roční provozní (od ROK 3+) ----------
    st.markdown(
        f"<div style='background:{color_main};color:white;padding:6px 12px;"
        f"border-radius:6px;font-weight:600;font-size:12px;"
        f"margin:14px 0 6px 0'>"
        f"{T['cost_sec2_header']}</div>", unsafe_allow_html=True)
    udr_help = (T["cost_udrzba_rocni_misc_help"] if is_misc
                else T["cost_udrzba_rocni_src_help"])
    p["udrzba_rocni"] = st.number_input(
        T["cost_udrzba_rocni_label"],
        value=float(defaults["udrzba_rocni"]), min_value=0.0, step=10.0, format="%.0f",
        help=udr_help, key=f"{crop_key}_uroc")
    p["dan_pozemku"] = st.number_input(
        T["cost_dan_label"],
        value=float(defaults["dan_pozemku"]), min_value=0.0, step=1.0, format="%.0f",
        key=f"{crop_key}_dan")

    # ---------- Sekce 3: Sklizeň + likvidace ----------
    st.markdown(
        f"<div style='background:{color_main};color:white;padding:6px 12px;"
        f"border-radius:6px;font-weight:600;font-size:12px;"
        f"margin:14px 0 6px 0'>"
        f"{T['cost_sec3_header']}</div>", unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        p["sklizen_per_tuna"] = st.number_input(
            T["cost_sklizen_label"],
            value=float(defaults["sklizen_per_tuna"]), min_value=0.0, step=1.0, format="%.0f",
            key=f"{crop_key}_skliz")
    with s2:
        p["likvidace"] = st.number_input(
            T["cost_likvidace_label"],
            value=float(defaults["likvidace"]), min_value=0.0, step=10.0, format="%.0f",
            help=(T["cost_likvidace_misc_help"] if is_misc
                  else T["cost_likvidace_src_help"]),
            key=f"{crop_key}_likv")

    # ---------- Sekce 4: Tržní + strukturální ----------
    st.markdown(
        f"<div style='background:{color_main};color:white;padding:6px 12px;"
        f"border-radius:6px;font-weight:600;font-size:12px;"
        f"margin:14px 0 6px 0'>"
        f"{T['cost_sec4_header']}</div>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    with t1:
        p["prodejni_cena_start"] = st.number_input(
            T["cost_cena_label"],
            value=float(defaults["prodejni_cena_start"]),
            min_value=0.0, step=1.0, format="%.0f",
            help=T["cost_cena_help"],
            key=f"{crop_key}_pcena")
    with t2:
        p["riziko_fail"] = st.number_input(
            T["cost_riziko_label"],
            value=float(defaults["riziko_fail"]),
            min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
            help=T["cost_riziko_help"],
            key=f"{crop_key}_rfail")
    with t3:
        p["zivotnost"] = st.number_input(
            T["cost_zivotnost_label"],
            value=int(defaults["zivotnost"]),
            min_value=5, max_value=40, step=1, format="%d",
            help=(T["cost_zivotnost_misc_help"] if is_misc
                  else T["cost_zivotnost_src_help"]),
            key=f"{crop_key}_ziv")

    # capex_std zůstává z defaults (neuživatelský parametr)
    p["capex_std"] = float(defaults["capex_std"])
    return p


with st.container(border=True):
    step_header(3, T["sec3_header"], T["sec3_desc"])

    if show_misc and show_src:
        cc1, cc2 = st.columns(2)
    elif show_misc or show_src:
        cc1 = cc2 = st.container()
    else:
        cc1 = cc2 = None
        st.info(T["cost_warn_no_crop"])

    if show_misc and cc1 is not None:
        with cc1:
            editable_costs["Miscanthus"] = _render_crop_costs(
                "Miscanthus", "M", COLOR_MISC_DARK, COLOR_MISC)

    if show_src and cc2 is not None:
        with cc2:
            editable_costs["SRC Vrba"] = _render_crop_costs(
                "SRC Vrba", "S", COLOR_SRC_DARK, COLOR_SRC)

# ===========================================================================
# KROK 4 – CITLIVOSTNÍ ANALÝZA (výběr parametrů)
# ===========================================================================
SA = SENSITIVITY_STRINGS[st.session_state["lang"]]
lang_sa = st.session_state["lang"]
sa_all_keys = list(SENSITIVITY_PARAMS.keys())
sa_options = {k: v[lang_sa] for k, v in SENSITIVITY_PARAMS.items()}

with st.container(border=True):
    step_header(4, SA["sa_header"], T["sa_subtitle"])

    sa_selected = []

    # Rozdělit VŠECHNY parametry do řádků po 5 (jinak by se poslední
    # parametry — eskalace ceny g, eskalace nákladů e_o, sigma_P — nezobrazily)
    for row_start in range(0, len(sa_all_keys), 5):
        row_keys = sa_all_keys[row_start:row_start + 5]
        sa_cols = st.columns(5)
        for col, pk in zip(sa_cols, row_keys):
            with col:
                if st.checkbox(sa_options[pk], value=False, key=f"sa_chk_{pk}"):
                    sa_selected.append(pk)

    if len(sa_selected) > SA_MAX_SELECTED:
        warn_msg = T["sa_warn_too_many"].format(
            n=len(sa_selected), max=SA_MAX_SELECTED)
        st.warning(warn_msg)
        sa_selected = sa_selected[:SA_MAX_SELECTED]

# ===========================================================================
# RUN BUTTON + SIMULACE → ulož do session_state pro perzistentní výsledky
# ===========================================================================
st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    run_clicked = st.button(T["run_button"], type="primary", use_container_width=True)

if run_clicked:
    if not plodiny:
        st.error(T["cost_warn_no_crop"])
    else:
        results, summary_data = {}, []
        bar = st.progress(0)

        for i, plodina in enumerate(plodiny):
            try:
                yk       = "SRC" if plodina == "SRC Vrba" else "M_giganteus"
                y_bounds = YIELD_DATA[detected_zone_key][soil_key][yk]
            except KeyError:
                y_bounds = [5, 10]
                st.warning(T["warn_missing"].format(crop=plodina, zone=detected_zone_label))

            if plodina == "M_giganteus":
                params = editable_costs["Miscanthus"]
                cf, yields, prices, costs = simulate_miscanthus(
                    n_sim, int(params["zivotnost"]), params, y_bounds, subsidy_pct/100.0,
                    rho=rho_input, drift=drift_rate,
                    cost_escalation=cost_escalation_rate,
                    pachtovne=pachtovne, catastrophic=catastrophic_input)
                results[plodina] = {"cf": cf, "yields": yields, "prices": prices,
                                    "costs": costs,
                                    "years": int(params["zivotnost"]),
                                    "params": params, "y_bounds": y_bounds, "type": "misc"}
            elif plodina == "SRC Vrba":
                params = editable_costs["SRC Vrba"]
                cf, yields, prices, costs = simulate_src(
                    n_sim, int(params["zivotnost"]), params, y_bounds, "Direct Chip", subsidy_pct/100.0,
                    rho=rho_input, drift=drift_rate,
                    cost_escalation=cost_escalation_rate,
                    pachtovne=pachtovne, soil_quality=soil_key,
                    catastrophic=catastrophic_input)
                results[plodina] = {"cf": cf, "yields": yields, "prices": prices,
                                    "costs": costs,
                                    "years": int(params["zivotnost"]),
                                    "params": params, "y_bounds": y_bounds,
                                    "soil_quality": soil_key, "type": "src"}

            bar.progress((i+1)/len(plodiny))

        # Spočítej souhrny pro hero a tabs
        r = discount_pct / 100.0
        for plodina, data in results.items():
            cf_total = data["cf"] * plocha_ha
            if r > 0:
                df = (1 + r) ** -np.arange(data["years"])
                total_profits = np.sum(cf_total * df[:, np.newaxis], axis=0)
            else:
                total_profits = np.sum(cf_total, axis=0)
            var_5  = float(np.percentile(total_profits, 5))
            cvar_5 = float(total_profits[total_profits <= var_5].mean())
            avg_yearly_std = float(np.mean(np.std(cf_total, axis=0)))
            cum_cf = np.cumsum(cf_total, axis=0)
            pb_yrs = np.argmax(cum_cf > 0, axis=0)
            # KONVENCE: payback_prob = P(NPV > 0) — projekt skončí kladným
            # finálním kumulativním CF (po likvidaci). Toto je v souladu
            # s kapitolou 4 práce. Pouze "cumsum > 0 kdykoliv" by bylo zavádějící
            # pro SRC (rok ~17 dočasně >0, pak likvidace stáhne zpět).
            npv_pos_mask = total_profits > 0
            payback_prob = float(npv_pos_mask.mean())
            # Doba návratnosti: počítáme jen pro úspěšné scénáře (NPV > 0)
            # — ostatní by zkreslovaly průměr (vrátit se a zase ztratit ≠ návratnost).
            succ_pb = pb_yrs[(pb_yrs > 0) & npv_pos_mask]
            avg_payback  = float(np.mean(succ_pb)+1) if len(succ_pb) > 0 else float("nan")
            total_eaas = equivalent_annual_annuity(total_profits, data["years"], r)
            eaa_mean = float(np.mean(total_eaas))
            # VaR a CVaR 5 % počítané z rozdělení EAA (nejen NPV)
            var_5_eaa  = float(np.percentile(total_eaas, 5))
            cvar_pool_eaa = total_eaas[total_eaas <= var_5_eaa]
            cvar_5_eaa = float(cvar_pool_eaa.mean()) if len(cvar_pool_eaa) else var_5_eaa

            data["total_profits"] = total_profits
            data["total_eaas"]    = total_eaas
            data["succ_pb"]       = succ_pb
            data["cf_total"]      = cf_total   # pro IRR výpočet v záložce IRR
            data["metrics"] = {
                "mean_profit": float(np.mean(total_profits)),
                "potential":   float(np.percentile(total_profits, 95)),
                "eaa":         eaa_mean,
                "var":         var_5,
                "cvar":        cvar_5,
                "var_eaa":     var_5_eaa,
                "cvar_eaa":    cvar_5_eaa,
                "std":         avg_yearly_std,
                "payback":     avg_payback,
                "payback_prob":payback_prob,
            }
            summary_data.append({
                "Plodina": plodina,
                "Průměrný Zisk": data["metrics"]["mean_profit"],
                "EAA (EUR/rok)": eaa_mean,
                "Potenciál (95%)": data["metrics"]["potential"],
                "Doba návratnosti (roky)": avg_payback,
                "Pravděpodobnost návratnosti": payback_prob,
                "VaR (5%)": var_5,
                "CVaR (5%)": cvar_5,
                "VaR EAA (5%)": var_5_eaa,
                "CVaR EAA (5%)": cvar_5_eaa,
                "Prům. roční odchylka": avg_yearly_std,
            })

        # Persistujeme – aby výsledky přežily změny ostatních widgetů
        st.session_state["sim_results"]   = results
        st.session_state["sim_summary"]   = summary_data
        # IRR distribuce — spočítáme rovnou při běhu simulace
        irr_session = {}
        for plodina, data in results.items():
            irrs, valid = compute_irr_distribution(data["cf_total"])
            irr_session[plodina] = {"irrs": irrs, "valid": valid}
        st.session_state["irr_distributions"] = irr_session
        st.session_state.pop("irr_params", None)
        st.session_state["sim_meta"]      = {
            "area": plocha_ha, "n_sim": n_sim,
            "subsidy": subsidy_pct, "discount": discount_pct,
            "rho": rho_input, "drift_pct": drift_pct,
            "catastrophic": catastrophic_input,
            "cost_esc_pct": cost_esc_pct,
            "pachtovne": pachtovne,
        }
        st.session_state["sim_sa_selected"] = list(sa_selected)

# ===========================================================================
# RESULTS – vždy renderuj taby (s placeholderem dokud nedoběhne první sim)
# ===========================================================================
TAB_LABELS = [
    T["tab_overview"], T["tab_yields"], T["tab_risk"], T["tab_irr"],
    T["tab_sens"], T["tab_compare"], T["tab_div"],
]

st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
(tab_overview, tab_yields, tab_risk, tab_irr, tab_sens, tab_compare,
 tab_div) = st.tabs(TAB_LABELS)

sim_results = st.session_state.get("sim_results")
sim_summary = st.session_state.get("sim_summary", [])
sim_meta    = st.session_state.get("sim_meta", {})
sim_sa_sel  = st.session_state.get("sim_sa_selected", [])
sim_area    = sim_meta.get("area", plocha_ha)


def _placeholder(msg_cs=None, msg_en=None):
    # Backward-compatible signature; new code passes no args and uses
    # the translated placeholder string.
    msg = T["placeholder_run_sim"]
    st.info("👆 " + msg)


def generate_excel_export(sim_results, sim_meta, sim_summary, sim_area):
    """
    Sestaví kompletní Excel report ze simulace (7 sheetů) jako BytesIO.

    Sheety:
      1_Souhrn         – KPI per plodina (NPV, EAA, VaR, CVaR, σ, P>0, payback)
      2_Konfigurace    – vstupní parametry (sim_meta + per-crop parametry)
      3_Roční výnosy   – year × percentily (mean, p5, p25, p50, p75, p95)
      4_Roční ceny     – year × percentily (OU proces)
      5_Roční náklady  – year × kategorie (setup, oper, liquid, total)
      6_Cash flow      – year × percentily (s plochou)
      7_NPV distribuce – histogram NPV (50 binů)
    """
    from io import BytesIO
    import pandas as pd

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as wr:
        # --- Sheet 1: Souhrn ---
        if sim_summary:
            df_souhrn = pd.DataFrame(sim_summary)
            df_souhrn.to_excel(wr, sheet_name=T["excel_sheet_souhrn"], index=False)
        else:
            pd.DataFrame({"info": [T["excel_no_sim"]]}).to_excel(
                wr, sheet_name=T["excel_sheet_souhrn"], index=False)

        # --- Sheet 2: Konfigurace ---
        cfg_rows = []
        cfg_rows.append((T["excel_cfg_area"], sim_meta.get("area", "")))
        cfg_rows.append((T["excel_cfg_nsim"], sim_meta.get("n_sim", "")))
        cfg_rows.append((T["excel_cfg_subsidy"], sim_meta.get("subsidy", "")))
        cfg_rows.append((T["excel_cfg_discount"], sim_meta.get("discount", "")))
        cfg_rows.append((T["excel_cfg_rho"], sim_meta.get("rho", "")))
        cfg_rows.append((T["excel_cfg_drift"], sim_meta.get("drift_pct", "")))
        cfg_rows.append((T["excel_cfg_costesc"], sim_meta.get("cost_esc_pct", "")))
        cfg_rows.append((T["excel_cfg_rent"], sim_meta.get("pachtovne", "")))
        cfg_rows.append(("", ""))
        for plodina, data in sim_results.items():
            cfg_rows.append((f"=== {plodina} ===", ""))
            cfg_rows.append((T["excel_cfg_soil"], data.get("soil_quality", "")))
            cfg_rows.append((T["excel_cfg_life"], data.get("years", "")))
            cfg_rows.append((T["excel_cfg_yields"],
                             f"{data['y_bounds'][0]}–{data['y_bounds'][1]}"))
            for k, v in data.get("params", {}).items():
                cfg_rows.append((f"  {k}", v))
            cfg_rows.append(("", ""))
        df_cfg = pd.DataFrame(
            cfg_rows,
            columns=[T["excel_cfg_paramcol"], T["excel_cfg_valuecol"]],
        )
        df_cfg.to_excel(wr, sheet_name=T["excel_sheet_konfig"], index=False)

        # Helper: percentily transponované — řádky = (Plodina × Statistika),
        # sloupce = roky (Rok 0, Rok 1, ..., Rok N-1)
        def percentile_wide_df(arr, plodina_label, prefix=""):
            n_yrs = arr.shape[0]
            year_cols = [f"{T['excel_year_prefix']} {i}" for i in range(n_yrs)]
            stats_data = {
                T["excel_col_plodina"]: [plodina_label] * 6,
                T["excel_col_statistika"]: ["Mean", "P5", "P25", "Median", "P75", "P95"],
            }
            data_arr = np.vstack([
                arr.mean(axis=1),
                np.percentile(arr, 5,  axis=1),
                np.percentile(arr, 25, axis=1),
                np.percentile(arr, 50, axis=1),
                np.percentile(arr, 75, axis=1),
                np.percentile(arr, 95, axis=1),
            ])
            for j, yc in enumerate(year_cols):
                stats_data[yc] = data_arr[:, j].tolist()
            return pd.DataFrame(stats_data)

        # --- Sheet 3: Roční výnosy (years jako sloupce) ---
        rows = []
        for plodina, data in sim_results.items():
            rows.append(percentile_wide_df(data["yields"], plodina))
        if rows:
            pd.concat(rows, ignore_index=True).to_excel(
                wr, sheet_name=T["excel_sheet_vynosy"], index=False)

        # --- Sheet 4: Roční ceny (years jako sloupce) ---
        rows = []
        for plodina, data in sim_results.items():
            rows.append(percentile_wide_df(data["prices"], plodina))
        if rows:
            pd.concat(rows, ignore_index=True).to_excel(
                wr, sheet_name=T["excel_sheet_ceny"], index=False)

        # --- Sheet 5: Roční náklady (years jako sloupce) ---
        # Řádky: Plodina × Kategorie (Setup/CAPEX, Provoz, Likvidace, Celkem)
        rows = []
        for plodina, data in sim_results.items():
            if "costs" not in data:
                continue
            c = data["costs"]
            n_yrs = c["setup"].shape[0]
            year_cols = [f"{T['excel_year_prefix']} {i}" for i in range(n_yrs)]
            cat_data = {
                T["excel_col_plodina"]:   [plodina] * 4,
                T["excel_col_category"]: [T["excel_cat_setup"], T["excel_cat_oper"],
                                          T["excel_cat_liquid"], T["excel_cat_total"]],
            }
            data_arr = np.vstack([
                c["setup"].mean(axis=1),
                c["oper"].mean(axis=1),
                c["liquid"].mean(axis=1),
                (c["setup"] + c["oper"] + c["liquid"]).mean(axis=1),
            ])
            for j, yc in enumerate(year_cols):
                cat_data[yc] = data_arr[:, j].tolist()
            rows.append(pd.DataFrame(cat_data))
        if rows:
            pd.concat(rows, ignore_index=True).to_excel(
                wr, sheet_name=T["excel_sheet_naklady"], index=False)

        # --- Sheet 6: Cash flow (× area, years jako sloupce) ---
        rows = []
        for plodina, data in sim_results.items():
            cf_total = data["cf"] * sim_area
            rows.append(percentile_wide_df(cf_total, plodina))
        if rows:
            pd.concat(rows, ignore_index=True).to_excel(
                wr, sheet_name=T["excel_sheet_cf"], index=False)

        # --- Sheet 7: NPV distribuce (histogram) ---
        rows = []
        for plodina, data in sim_results.items():
            if "total_profits" not in data:
                continue
            tp = data["total_profits"]
            counts, edges = np.histogram(tp, bins=50)
            centers = 0.5 * (edges[:-1] + edges[1:])
            df = pd.DataFrame({
                T["excel_col_plodina"]:   plodina,
                T["excel_npv_bin_mid"]:   centers,
                T["excel_npv_bin_start"]: edges[:-1],
                T["excel_npv_bin_end"]:   edges[1:],
                T["excel_npv_freq"]:      counts,
                T["excel_npv_density"]:   counts / counts.sum(),
            })
            rows.append(df)
        if rows:
            pd.concat(rows, ignore_index=True).to_excel(
                wr, sheet_name=T["excel_sheet_npv"], index=False)

    bio.seek(0)
    return bio.getvalue()


# ----- TAB 1: PŘEHLED ------------------------------------------------------
with tab_overview:
    if not sim_results:
        _placeholder()
    else:
        # ---------- Excel export tlačítko (úplně nahoře) ----------
        try:
            excel_bytes = generate_excel_export(sim_results, sim_meta,
                                                  sim_summary, sim_area)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            crops_in_run = "_".join(p[:4] for p in sim_results.keys())
            xls_filename = f"BioFarm_simulace_{crops_in_run}_{timestamp}.xlsx"

            st.download_button(
                label=T["dl_excel_label"],
                data=excel_bytes,
                file_name=xls_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_excel_export",
                help=T["dl_excel_help"],
            )
            st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(T["excel_export_error"].format(err=e))

        # Hero KPI karta
        hero_inner = ""
        for plodina, data in sim_results.items():
            m = data["metrics"]
            crop_color = COLOR_MISC if data["type"] == "misc" else COLOR_SRC
            crop_icon  = "🌾" if data["type"] == "misc" else "🌳"
            payback_str = (f"{m['payback']:.1f} {T['hero_payback_unit']}"
                           if not np.isnan(m['payback']) else "N/A")
            pb_prob_str = (f"{m['payback_prob']:.1%}" if m['payback_prob'] >= 0.01
                           else f"{m['payback_prob']:.4%}")
            hero_inner += f"""
<div style='border-left:4px solid {crop_color};padding-left:14px;margin-bottom:14px'>
  <div class='hero-title' style='margin-bottom:10px'>{crop_icon} {plodina}</div>
  <div class='hero-grid'>
    <div class='hero-item'><div class='hero-label'>{T['hero_label_npv']}</div>
      <div class='hero-value'>{fmt_eur_cs(m['mean_profit'])}</div></div>
    <div class='hero-item'><div class='hero-label'>{T['hero_label_eaa']}</div>
      <div class='hero-value'>{fmt_eur_cs(m['eaa'])}</div></div>
    <div class='hero-item'><div class='hero-label'>{T['hero_label_potential']}</div>
      <div class='hero-value'>{fmt_eur_cs(m['potential'])}</div></div>
    <div class='hero-item'><div class='hero-label'>{T['hero_label_payback']}</div>
      <div class='hero-value'>{payback_str}</div></div>
    <div class='hero-item'><div class='hero-label'>{T['hero_label_payback_prob']}</div>
      <div class='hero-value'>{pb_prob_str}</div></div>
  </div>
</div>"""
        st.markdown(f"<div class='hero-card'>{hero_inner}</div>", unsafe_allow_html=True)

        # Cash flow grafy – pod sebou pro přímé porovnání
        st.subheader("📈 " + T["cf_trajectory_header"])
        for plodina, data in sim_results.items():
            cf_total = data["cf"] * sim_area
            yrs = np.arange(0, data["years"])
            mean_cf = np.mean(cf_total, axis=1)
            p5_cf = np.percentile(cf_total, 5, axis=1)
            p95_cf = np.percentile(cf_total, 95, axis=1)
            color = CROP_COLOR.get(data["type"], "#1f77b4")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=yrs, y=p95_cf, line=dict(width=0), showlegend=False, name="P95"))
            fig.add_trace(go.Scatter(x=yrs, y=p5_cf, fill="tonexty",
                                      fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)",
                                      line=dict(width=0), showlegend=False, name="P5"))
            fig.add_trace(go.Scatter(x=yrs, y=mean_cf, name=T["cf_mean"],
                                      line=dict(color=color, width=3)))
            fig.update_layout(
                title=f"{plodina} ({sim_area:.0f} ha)",
                xaxis_title=T["cf_xaxis"], yaxis_title=T["cf_yaxis"],
                height=360, margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"cf_tab_{plodina}")

        # Histogram celkového zisku – pod sebou
        st.subheader("📊 " + T["profit_dist_header"])
        for plodina, data in sim_results.items():
            color = CROP_COLOR.get(data["type"], "#1f77b4")
            fig = go.Figure(data=[go.Histogram(
                x=data["total_profits"], nbinsx=50,
                marker=dict(color=color, line=dict(color="white", width=0.5)),
            )])
            fig.update_layout(
                title=f"{plodina}", xaxis_title=T["hist_xaxis"], yaxis_title=T["hist_yaxis"],
                height=320, margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"hist_tab_{plodina}")

        # Histogram EAA (ekvivalentní roční anuita) – umožňuje srovnání plodin s rozdílnou životností
        st.subheader("📊 " + T["eaa_dist_header"])
        st.caption(T["eaa_dist_caption"])
        for plodina, data in sim_results.items():
            color = CROP_COLOR.get(data["type"], "#1f77b4")
            eaas = data["total_eaas"]
            mean_eaa = float(np.mean(eaas))
            median_eaa = float(np.median(eaas))
            p5_eaa = float(np.percentile(eaas, 5))
            cvar_pool = eaas[eaas <= p5_eaa]
            cvar5_eaa = float(cvar_pool.mean()) if len(cvar_pool) else p5_eaa

            fig = go.Figure(data=[go.Histogram(
                x=eaas, nbinsx=50,
                marker=dict(color=color, line=dict(color="white", width=0.5)),
                name=plodina,
            )])
            # Vertikální čáry: nula, průměr, VaR 5%, CVaR 5%
            fig.add_vline(x=0, line=dict(color="black", width=1, dash="dot"),
                          annotation_text="0", annotation_position="top")
            fig.add_vline(x=mean_eaa, line=dict(color="#1B5E20", width=2, dash="dash"),
                          annotation_text=f"μ = {mean_eaa:,.0f}".replace(",", " "),
                          annotation_position="top right")
            fig.add_vline(x=p5_eaa, line=dict(color="#E65100", width=2, dash="dash"),
                          annotation_text=f"VaR₅ = {p5_eaa:,.0f}".replace(",", " "),
                          annotation_position="top")
            fig.add_vline(x=cvar5_eaa, line=dict(color="#B71C1C", width=2, dash="dash"),
                          annotation_text=f"CVaR₅ = {cvar5_eaa:,.0f}".replace(",", " "),
                          annotation_position="top left")
            xaxis_title = T["eaa_xaxis"]
            yaxis_title = T["scenarios_count"]
            median_fmt = f"{median_eaa:,.0f}".replace(",", " ")
            p5_fmt = f"{p5_eaa:,.0f}".replace(",", " ")
            fig.update_layout(
                title=T["eaa_hist_title"].format(
                    crop=plodina, median=median_fmt, p5=p5_fmt),
                xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                height=340, margin=dict(t=50, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"hist_eaa_tab_{plodina}")

        # Rekapitulační tabulka – pokud 2 plodiny
        if len(sim_summary) > 1:
            st.subheader("🏆 " + T["recap_header"].lstrip("🏆 "))
            st.markdown(render_recap_table(sim_summary, T), unsafe_allow_html=True)

# ----- TAB 2: VÝNOSY & CENY -----------------------------------------------
with tab_yields:
    if not sim_results:
        _placeholder()
    else:
        st.subheader("🌱 " + T["yields_header"])
        for plodina, data in sim_results.items():
            yrs = np.arange(0, data["years"])
            color = CROP_COLOR.get(data["type"], "#ff7f0e")
            my   = np.mean(data["yields"], axis=1)
            p5y  = np.percentile(data["yields"], 5, axis=1)
            p95y = np.percentile(data["yields"], 95, axis=1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=yrs, y=p95y, line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=yrs, y=p5y, fill="tonexty",
                                      fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)",
                                      line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=yrs, y=my, name=T["yield_mean"],
                                      line=dict(color=color, width=3)))
            fig.update_layout(title=f"{plodina}",
                               xaxis_title=T["yield_xaxis"], yaxis_title=T["yield_yaxis"],
                               height=360, margin=dict(t=40, b=40))
            st.plotly_chart(fig, use_container_width=True, key=f"yield_tab_{plodina}")

        st.subheader("💶 " + T["prices_header"])
        for plodina, data in sim_results.items():
            yrs = np.arange(0, data["years"])
            color = CROP_COLOR.get(data["type"], "purple")
            mp   = np.mean(data["prices"], axis=1)
            p5p  = np.percentile(data["prices"], 5, axis=1)
            p95p = np.percentile(data["prices"], 95, axis=1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=yrs, y=p95p, line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=yrs, y=p5p, fill="tonexty",
                                      fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)",
                                      line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=yrs, y=mp, name=T["price_mean"],
                                      line=dict(color=color, width=3)))
            fig.update_layout(title=f"{plodina}",
                               xaxis_title=T["price_xaxis"], yaxis_title=T["price_yaxis"],
                               height=360, margin=dict(t=40, b=40))
            st.plotly_chart(fig, use_container_width=True, key=f"price_tab_{plodina}")

        # --- Nákladové křivky (rozklad: CAPEX / provoz / likvidace) ---------
        st.subheader("💸 " + T["costs_header"])
        for plodina, data in sim_results.items():
            if "costs" not in data:
                continue
            yrs   = np.arange(0, data["years"])
            color = CROP_COLOR.get(data["type"], "#8E24AA")
            c     = data["costs"]
            mean_setup  = np.mean(c["setup"],  axis=1)   # EUR/ha
            mean_oper   = np.mean(c["oper"],   axis=1)
            mean_liquid = np.mean(c["liquid"], axis=1)
            total       = c["setup"] + c["oper"] + c["liquid"]
            mean_total  = np.mean(total, axis=1)
            p5_total    = np.percentile(total, 5,  axis=1)
            p95_total   = np.percentile(total, 95, axis=1)

            fig = go.Figure()
            # Stacked bary pro rozklad
            fig.add_trace(go.Bar(
                x=yrs, y=mean_setup,
                name=T["costs_legend_setup"],
                marker_color="#5C6BC0",
            ))
            fig.add_trace(go.Bar(
                x=yrs, y=mean_oper,
                name=T["costs_legend_oper"],
                marker_color="#FFB300",
            ))
            fig.add_trace(go.Bar(
                x=yrs, y=mean_liquid,
                name=T["costs_legend_liquid"],
                marker_color="#8D6E63",
            ))
            # Pásmo 5-95 % celkových nákladů (jako sekundární křivky)
            fig.add_trace(go.Scatter(
                x=yrs, y=p95_total,
                name="P95",
                line=dict(color=color, width=1, dash="dot"),
                mode="lines",
            ))
            fig.add_trace(go.Scatter(
                x=yrs, y=p5_total,
                name="P5",
                line=dict(color=color, width=1, dash="dot"),
                mode="lines",
            ))
            fig.update_layout(
                title=f"{plodina}",
                xaxis_title=T["costs_x_axis"],
                yaxis_title=T["costs_y_axis"],
                barmode="stack",
                height=380,
                margin=dict(t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"costs_tab_{plodina}")

            # KPI karta pro náklady
            total_lifetime = float(np.mean(np.sum(total, axis=0)))
            avg_annual = total_lifetime / data["years"]
            k1, k2, k3 = st.columns(3)
            k1.metric(
                T["kpi_costs_lifetime"],
                fmt_eur_cs(total_lifetime),
            )
            k2.metric(
                T["kpi_costs_annual"],
                fmt_eur_cs(avg_annual),
            )
            k3.metric(
                T["kpi_costs_capex"],
                fmt_eur_cs(float(np.mean(np.sum(c["setup"][:3] + c["oper"][:3], axis=0)))),
            )

        # SRC learning chart – pokud SRC v simulaci, ukáže Weger 2025 harvest
        # profil pro vybranou kvalitu půdy (8 sklizní v rámci životnosti plantáže)
        src_data_lc = next((d for d in sim_results.values() if d["type"] == "src"), None)
        if src_data_lc:
            st.subheader("🔄 " + T["learn_header"].lstrip("🔄 "))
            sq_lc = src_data_lc.get("soil_quality", "Průměrná")
            H_lc = WEGER_2025_HARVEST.get(sq_lc, WEGER_2025_HARVEST["Průměrná"])
            n_h = len(H_lc)
            src_harvest_yrs = [5 + i*3 for i in range(n_h)]   # roky 5, 8, …, 26
            rm_pct = (H_lc / H_lc.max() * 100).tolist()
            rl = [T["learn_rot_label"].format(r=r+1, yr=src_harvest_yrs[r])
                  for r in range(n_h)]
            peak_idx = int(np.argmax(H_lc))
            colors = ["#33691E" if i == peak_idx else
                      ("#7CB342" if v >= 80 else "#D6483D")
                      for i, v in enumerate(rm_pct)]
            fig = go.Figure(data=[go.Bar(
                x=rl, y=[round(v, 1) for v in rm_pct],
                marker=dict(color=colors, line=dict(color="white", width=1)),
                text=[f"{v:.0f} %" for v in rm_pct], textposition="outside",
            )])
            fig.update_layout(
                title=f"{T['learn_title']} ({sq_lc})",
                xaxis_title=T["learn_xaxis"],
                yaxis_title=T["learn_yaxis"], yaxis=dict(range=[0, 115]),
                height=380, margin=dict(t=50, b=40))
            st.plotly_chart(fig, use_container_width=True, key="learn_tab")

# ----- TAB 3: RIZIKO -------------------------------------------------------
with tab_risk:
    if not sim_results:
        _placeholder()
    else:
        # Risk metriky karty
        st.subheader("⚠️ " + T["risk_kpi_header"])
        for plodina, data in sim_results.items():
            m = data["metrics"]
            color = COLOR_MISC_DARK if data["type"] == "misc" else COLOR_SRC_DARK
            st.markdown(f"<h4 style='color:{color};margin-top:12px'>{plodina}</h4>",
                         unsafe_allow_html=True)
            # 1. řádek: NPV-based metriky (celkové)
            st.markdown(f"<p style='color:#666;font-size:13px;margin:4px 0 6px;'>"
                        f"{T['risk_horizon_label']}</p>",
                        unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            r1.metric(T["risk_std"],  fmt_eur_cs(m['std']))
            r2.metric(T["risk_var"],  fmt_eur_cs(m['var']),
                      delta_color="inverse", help=T["risk_var_help"])
            r3.metric(T["risk_cvar"], fmt_eur_cs(m['cvar']),
                      delta_color="inverse", help=T["risk_cvar_help"])
            # 2. řádek: EAA-based metriky (roční ekvivalent — srovnatelné mezi M a SRC)
            st.markdown(f"<p style='color:#666;font-size:13px;margin:10px 0 6px;'>"
                        f"{T['risk_annual_label']}</p>",
                        unsafe_allow_html=True)
            e1, e2, e3 = st.columns(3)
            e1.metric(T.get("kpi_eaa", "EAA"), fmt_eur_cs(m['eaa']),
                      help=T["risk_eaa_help"])
            e2.metric(T.get("risk_var_eaa", "VaR EAA (5 %)"),
                      fmt_eur_cs(m.get('var_eaa', 0)),
                      delta_color="inverse",
                      help=T.get("risk_var_eaa_help", ""))
            e3.metric(T.get("risk_cvar_eaa", "CVaR EAA (5 %)"),
                      fmt_eur_cs(m.get('cvar_eaa', 0)),
                      delta_color="inverse",
                      help=T.get("risk_cvar_eaa_help", ""))

        # Histogramy doby návratnosti
        st.subheader("📅 " + T["payback_title"].lstrip("📅 "))
        pb_cols = st.columns(len(sim_results))
        for col, (plodina, data) in zip(pb_cols, sim_results.items()):
            with col:
                color = CROP_COLOR.get(data["type"], "#2ca02c")
                fig = go.Figure(data=[go.Histogram(
                    x=data["succ_pb"]+1, marker=dict(color=color, line=dict(color="white", width=0.5)),
                )])
                fig.update_layout(title=f"{plodina}", xaxis_title=T["payback_xaxis"],
                                   yaxis_title=T["payback_yaxis"], height=320,
                                   margin=dict(t=40, b=40))
                st.plotly_chart(fig, use_container_width=True, key=f"pb_tab_{plodina}")

        # 3D heatmap – riziko selhání × pravd. počasí
        st.subheader("🎲 " + T["sens_header"].lstrip("🎲 "))
        st.markdown(T["sens_desc"])
        for plodina, data in sim_results.items():
            color = COLOR_MISC_DARK if data["type"] == "misc" else COLOR_SRC_DARK
            st.markdown(f"<h4 style='color:{color};margin-top:8px'>{plodina}</h4>",
                         unsafe_allow_html=True)
            with st.spinner(T["sens_spinner"].format(crop=plodina)):
                xv, yv, zm = calculate_sensitivity_matrix(
                    data["type"], data["years"], data["params"], data["y_bounds"],
                    "Direct Chip" if data["type"] == "src" else None,
                    sim_area, sim_meta.get("n_sim", n_sim),
                    sim_meta.get("subsidy", subsidy_pct)/100.0,
                    rho=sim_meta.get("rho", rho_input),
                    drift=sim_meta.get("drift_pct", drift_pct)/100.0,
                    cost_escalation=sim_meta.get("cost_esc_pct", cost_esc_pct)/100.0,
                    pachtovne=sim_meta.get("pachtovne", pachtovne),
                    soil_quality=data.get("soil_quality", soil_key))
                f3d = go.Figure(data=[go.Surface(
                    z=zm, x=xv, y=yv, colorscale="YlOrRd",
                    colorbar=dict(title=T["sens_colorbar"]))])
                f3d.update_layout(
                    scene=dict(xaxis_title=T["sens_xaxis"],
                                yaxis_title=T["sens_yaxis"],
                                zaxis_title=T["sens_zaxis"]),
                    height=520, margin=dict(l=0, r=0, b=20, t=10))
                st.plotly_chart(f3d, use_container_width=True, key=f"3d_tab_{plodina}")

# ----- TAB 4: IRR ----------------------------------------------------------
with tab_irr:
    if not sim_results:
        _placeholder()
    else:
        st.subheader("💰 " + T["irr_header"])

        # === Vysvětlující úvod (bez ~ aby se neformátovalo přes Markdown) ===
        n_sim_str = f"{n_sim:,}".replace(",", " ")
        st.markdown(T["irr_intro"].format(n=n_sim_str))

        # IRR distribuce je předpočítaná při běhu simulace
        irr_dist = st.session_state.get("irr_distributions", {})

        def _fmt_pct(v):
            return "—" if pd.isna(v) else f"{v*100:+.2f} %"

        # === Tabulka metrik IRR per plodina ===
        st.markdown("### " + T["irr_table_header"])

        irr_rows = []
        for plodina, dist in irr_dist.items():
            valid_irrs = dist["irrs"][dist["valid"]]
            if len(valid_irrs) > 0:
                median_irr = float(np.median(valid_irrs))
                mean_irr = float(np.mean(valid_irrs))
                std_irr = float(np.std(valid_irrs))
                p5_irr = float(np.percentile(valid_irrs, 5))
                p25_irr = float(np.percentile(valid_irrs, 25))
                p75_irr = float(np.percentile(valid_irrs, 75))
            else:
                median_irr = mean_irr = std_irr = float("nan")
                p5_irr = p25_irr = p75_irr = float("nan")
            irr_rows.append({
                T["compare_col_plodina"]: plodina,
                T["irr_valid_col"]: f"{int(dist['valid'].sum())} / {len(dist['irrs'])}",
                "Median IRR":      median_irr,
                "Mean IRR":        mean_irr,
                "σ(IRR)":          std_irr,
                "P5 IRR":          p5_irr,
                "P25 IRR":         p25_irr,
                "P75 IRR":         p75_irr,
            })
        df_irr = pd.DataFrame(irr_rows)
        styled = df_irr.style.format({
            "Median IRR": _fmt_pct,
            "Mean IRR":   _fmt_pct,
            "σ(IRR)":     _fmt_pct,
            "P5 IRR":     _fmt_pct,
            "P25 IRR":    _fmt_pct,
            "P75 IRR":    _fmt_pct,
        }).set_properties(**{"text-align": "center"})
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # === Histogram distribuce IRR ===
        st.markdown("### " + T["irr_hist_header"])

        CROP_COLORS = {"Miscanthus": "#90EE90", "SRC Vrba": "#8B4513"}
        fig_irr = go.Figure()
        for plodina, dist in irr_dist.items():
            valid_irrs = dist["irrs"][dist["valid"]] * 100
            if len(valid_irrs) == 0:
                continue
            color = CROP_COLORS.get(plodina, "#1f77b4")
            fig_irr.add_trace(go.Histogram(
                x=valid_irrs, name=plodina, opacity=0.65,
                nbinsx=60, marker_color=color,
                hovertemplate=T["irr_hover_count"],
            ))
            # Pouze medián jako vertikální čára
            median_pct = float(np.median(valid_irrs))
            fig_irr.add_vline(x=median_pct, line_dash="dash",
                              line_color=color, line_width=2, opacity=0.8,
                              annotation_text=f"Median {plodina}: {median_pct:.1f} %",
                              annotation_position="top",
                              annotation_font_size=9)

        fig_irr.update_layout(
            barmode="overlay",
            xaxis_title="IRR (%)",
            yaxis_title=T["scenarios_count"],
            template="plotly_white", height=460,
            margin=dict(l=20, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )
        st.plotly_chart(fig_irr, use_container_width=True, key="irr_hist")

        # === Metodologická poznámka ===
        with st.expander("ℹ️ " + T["irr_explainer_title"]):
            st.markdown(T["irr_explainer_body"])

# ----- TAB 5: CITLIVOSTKA --------------------------------------------------
with tab_sens:
    if not sim_results:
        _placeholder()
    elif not sim_sa_sel:
        st.info(T["sens_tab_warn_no_param"])
    else:
        param_labels_full  = {k: v[lang_sa] for k, v in SENSITIVITY_PARAMS.items()}
        short_key = f"{lang_sa}_short"
        param_labels_short = {k: v[short_key] for k, v in SENSITIVITY_PARAMS.items()}
        sa_n_sim = sim_meta.get("n_sim", n_sim)

        for plodina, data in sim_results.items():
            color = COLOR_MISC_DARK if data["type"] == "misc" else COLOR_SRC_DARK
            st.markdown(f"<h3 style='color:{color}'>{plodina}</h3>", unsafe_allow_html=True)
            all_sa = {}
            sa_bar = st.progress(0)
            for idx, pk in enumerate(sim_sa_sel):
                with st.spinner(SA["sa_running"].format(
                        crop=plodina, param=param_labels_full[pk])):
                    all_sa[pk] = run_param_sensitivity(
                        param_key=pk, base_params=data["params"],
                        y_bounds=data["y_bounds"], crop_type=data["type"],
                        n_sim=sa_n_sim, years=data["years"],
                        subsidy_perc=sim_meta.get("subsidy", subsidy_pct)/100.0,
                        area_ha=sim_area,
                        discount_rate=sim_meta.get("discount", discount_pct)/100.0,
                        weather_prob=0.05,
                        src_tech="Direct Chip" if data["type"] == "src" else None,
                        rho=sim_meta.get("rho", rho_input),
                        drift=sim_meta.get("drift_pct", drift_pct)/100.0,
                        cost_escalation=sim_meta.get("cost_esc_pct", cost_esc_pct)/100.0,
                        pachtovne=sim_meta.get("pachtovne", pachtovne),
                        soil_quality=data.get("soil_quality", soil_key),
                    )
                sa_bar.progress((idx + 1) / len(sim_sa_sel))

            y_labels = {
                "profit":       T["sa_y_profit"],
                "eaa":          T["sa_y_eaa"],
                "std":          T["sa_y_std"],
                "var":          T["sa_y_var"],
                "cvar":         T["sa_y_cvar"],
                "payback_prob": T["sa_y_pb_prob"],
                "payback_yr":   T["sa_y_pb_yr"],
            }
            sa_title_eaa = T["sa_title_eaa"]

            charts = [
                ("profit", SA["sa_title_profit"], y_labels["profit"]),
                ("eaa",    sa_title_eaa,           y_labels["eaa"]),
                ("std",    SA["sa_title_std"],     y_labels["std"]),
                ("var",    SA["sa_title_var"],     y_labels["var"]),
                ("cvar",   SA["sa_title_cvar"],    y_labels["cvar"]),
                ("payback_prob", SA["sa_title_pb_prob"], y_labels["payback_prob"]),
                ("payback_yr",   SA["sa_title_pb_yr"],   y_labels["payback_yr"]),
            ]
            for i in range(0, len(charts), 2):
                cols = st.columns(2)
                for col, (key, title, ylabel) in zip(cols, charts[i:i+2]):
                    with col:
                        st.plotly_chart(make_sensitivity_line_chart(
                            all_sa, key, title, ylabel, param_labels_short, lang_sa),
                            use_container_width=True, key=f"sa_{key}_{plodina}")

            # ---------- Tornádový graf (NPV impact ranking) ----------
            tornado_baseline = data["metrics"].get("mean_profit")
            tornado_fig = make_tornado_chart(
                all_sa,
                metric_key="profit",
                param_labels_full=param_labels_full,
                lang_code=lang_sa,
                baseline_value=tornado_baseline,
                top_n=10,
            )
            if tornado_fig is not None:
                st.markdown(
                    "<div style='margin-top:24px'></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(T["tornado_caption"])
                st.plotly_chart(tornado_fig, use_container_width=True,
                                 key=f"tornado_{plodina}")
            st.divider()

# ----- TAB 5: SROVNÁNÍ KVALIT PŮDY (4 půdy × 2 plodiny) ------------------
COMPARE_T = {
    "cs": {
        "header":     "Srovnání 4 kvalit půdy × 2 plodiny",
        "desc":       ("Spustí 8 nezávislých Monte Carlo simulací (4 kvality půdy × "
                       "Miscanthus + SRC vrba) v aktuálně vybrané klimatické zóně. "
                       "Výsledek je risk–return scatter graf pro vyhodnocení, "
                       "kde leží efektivní volby investora."),
        "btn":        "🚀 Spustit srovnání 8 konfigurací",
        "spinner":    "Probíhá 8 simulací (~10 s)…",
        "scatter_t":  "Risk–return scatter (Markowitz) — výnos vs. riziko",
        "scatter_x":  "σ ročního cash flow [EUR/rok]",
        "scatter_y":  "Mean EAA [EUR/rok]",
        "scatter_cvar_t":  "Downside risk scatter — výnos vs. CVaR 5 %",
        "scatter_cvar_x":  "CVaR 5 % [EUR] (méně záporné = lepší)",
        "scatter_cvar_y":  "Mean EAA [EUR/rok]",
        "scatter_cvar_note": "Doplňkový pohled zaměřený výhradně na downside-risk: "
                              "horizontální osa zobrazuje průměrnou ztrátu v 5 % "
                              "nejhorších scénářů (CVaR), vertikální osa roční výnos "
                              "(EAA). Bod blíže pravému hornímu rohu = atraktivnější "
                              "investice (vyšší výnos, mírnější downside).",
        "table_t":    "Souhrnná tabulka (8 řádků)",
        "params_t":   "Vstupní parametry pro každou kombinaci",
        "warn_zone":  "Před spuštěním vyber klimatickou zónu a plochu v sekci 1.",
        "ideal_note": ("💡 **Konvence (Markowitz 1952):** osa X = riziko "
                       "(σ ročního CF), osa Y = výnos (mean NPV). "
                       "**Ideální oblast je vlevo nahoře** (vysoký NPV × nízké σ). "
                       "Body **vpravo dole** jsou rizikově dominované."),
        "dl_btn":     "📥 Stáhnout srovnání jako Excel",
    },
    "en": {
        "header":     "Comparison: 4 soil qualities × 2 crops",
        "desc":       ("Runs 8 independent Monte Carlo simulations (4 soil qualities × "
                       "Miscanthus + SRC willow) in the currently selected climate zone. "
                       "Output is a risk–return scatter chart showing where efficient "
                       "investment choices lie."),
        "btn":        "🚀 Run comparison of 8 configurations",
        "spinner":    "Running 8 simulations (~10 s)…",
        "scatter_t":  "Risk–return scatter (Markowitz) — return vs. risk",
        "scatter_x":  "σ of annual cash flow [EUR/yr]",
        "scatter_y":  "Mean EAA [EUR/yr]",
        "scatter_cvar_t":  "Downside risk scatter — return vs. CVaR 5%",
        "scatter_cvar_x":  "CVaR 5% [EUR] (less negative = better)",
        "scatter_cvar_y":  "Mean EAA [EUR/yr]",
        "scatter_cvar_note": "Complementary downside-focused view: horizontal axis "
                              "shows average loss in worst 5% of scenarios (CVaR), "
                              "vertical axis shows annual return (EAA). Top-right "
                              "corner = more attractive (higher return, milder downside).",
        "table_t":    "Summary table (8 rows)",
        "params_t":   "Input parameters per scenario",
        "warn_zone":  "Pick climate zone and area in section 1 before running.",
        "ideal_note": ("💡 **Convention (Markowitz 1952):** X = risk "
                       "(σ annual CF), Y = return (mean NPV). "
                       "**Ideal region is top-left** (high NPV × low σ). "
                       "Points in **bottom-right** are risk-dominated."),
        "dl_btn":     "📥 Download comparison as Excel",
    },
}[lang_code]


def _run_compare_scenarios(zone_label, area_ha, n_sim_compare, subsidy_pct,
                            discount_pct, rho_input, drift_pct, cost_esc_pct,
                            pachtovne_v, editable_costs):
    """
    Spustí 8 simulací (4 kvality půdy × 2 plodiny) pro aktuální zónu
    a vrátí seznam dictů s metrikami pro každou kombinaci.

    Pokud uživatel nemá v sekci 2 vybranou některou plodinu (chybí klíč
    v editable_costs), použije se DEFAULT_COSTS jako fallback. Tabulka
    Srovnání kvalit je nezávislá na běžné simulaci a vždy ukazuje obě plodiny.
    """
    from copy import deepcopy
    soil_keys = ["Optimální", "Průměrná", "Neúrodná", "Nevhodná"]
    crop_keys = ["M_giganteus", "SRC Vrba"]
    # Sloučení uživatelských úprav s defaulty (pro nevybrané plodiny)
    fallback = {
        "Miscanthus": deepcopy(DEFAULT_COSTS["Miscanthus"]),
        "SRC Vrba":   deepcopy(DEFAULT_COSTS["SRC Vrba"]),
    }
    fallback["Miscanthus"].update(editable_costs.get("Miscanthus", {}))
    fallback["SRC Vrba"].update(editable_costs.get("SRC Vrba", {}))

    out = []
    r = discount_pct / 100.0

    progress = st.progress(0.0, text=COMPARE_T["spinner"])
    total = len(soil_keys) * len(crop_keys)
    step = 0

    for soil in soil_keys:
        # Pro tuto kvalitu půdy vytáhneme y_bounds z YIELD_DATA
        try:
            zone_dict = YIELD_DATA[zone_label][soil]
        except KeyError:
            continue
        for crop in crop_keys:
            crop_label = "Miscanthus" if crop == "M_giganteus" else "SRC Vrba"
            params = deepcopy(fallback[crop_label])
            yb = zone_dict[crop if crop == "SRC" else
                            ("M_giganteus" if crop == "M_giganteus" else "SRC")]
            yb = zone_dict.get("M_giganteus" if crop == "M_giganteus" else "SRC", [0, 1])
            years_p = int(params["zivotnost"])
            if crop == "M_giganteus":
                cf, _, _, _ = simulate_miscanthus(
                    n_sim_compare, years_p, params, list(yb),
                    subsidy_pct/100.0,
                    rho=rho_input, drift=drift_pct/100.0,
                    cost_escalation=cost_esc_pct/100.0,
                    pachtovne=pachtovne_v)
            else:
                cf, _, _, _ = simulate_src(
                    n_sim_compare, years_p, params, list(yb),
                    "Direct Chip", subsidy_pct/100.0,
                    rho=rho_input, drift=drift_pct/100.0,
                    cost_escalation=cost_esc_pct/100.0,
                    pachtovne=pachtovne_v,
                    soil_quality=soil)
            cf_total = cf * area_ha
            if r > 0:
                df = (1 + r) ** -np.arange(years_p)
                npvs = np.sum(cf_total * df[:, np.newaxis], axis=0)
            else:
                npvs = np.sum(cf_total, axis=0)
            var5  = float(np.percentile(npvs, 5))
            cvar5 = float(npvs[npvs <= var5].mean()) if (npvs <= var5).any() else var5
            cum = np.cumsum(cf_total, axis=0)
            pb_yrs = np.argmax(cum > 0, axis=0)
            succ_pb = pb_yrs[pb_yrs > 0]
            payback_prob = float(len(succ_pb)/n_sim_compare) if n_sim_compare > 0 else 0.0
            mean_npv     = float(np.mean(npvs))
            # Yearly std CF — průměrná ročně-roční volatilita cash flow
            # (konzistentní s metrics["std"] v hlavní simulaci)
            std_yearly   = float(np.mean(np.std(cf_total, axis=0)))
            eaa_mean     = float(np.mean(equivalent_annual_annuity(npvs, years_p, r)))
            p_pos        = float((npvs > 0).mean())
            y_min, y_max = list(yb)

            out.append({
                "Plodina":      crop_label,
                "Půda":         soil,
                "Y_min":        y_min,
                "Y_max":        y_max,
                "mean NPV":     mean_npv,
                "σ ročního CF": std_yearly,
                "VaR 5%":       var5,
                "CVaR 5%":      cvar5,
                "EAA":          eaa_mean,
                "P(NPV>0)":     p_pos,
                "Payback prob.": payback_prob,
            })
            step += 1
            progress.progress(step/total, text=f"{COMPARE_T['spinner']} ({step}/{total})")
    progress.empty()
    return out


with tab_compare:
    st.subheader("📈 " + COMPARE_T["header"])
    st.markdown(COMPARE_T["desc"])

    if not detected_zone_label or detected_zone_label not in YIELD_DATA:
        st.warning(COMPARE_T["warn_zone"])
    else:
        st.info(T["compare_meta_info"].format(
            zone=detected_zone_label, area=plocha_ha, n=n_sim))

        if st.button(COMPARE_T["btn"], type="primary", key="btn_compare_run"):
            with st.spinner(COMPARE_T["spinner"]):
                cmp_n_sim = n_sim
                cmp_res = _run_compare_scenarios(
                    detected_zone_label, plocha_ha, cmp_n_sim,
                    subsidy_pct, discount_pct, rho_input, drift_pct,
                    cost_esc_pct, pachtovne, editable_costs)
                st.session_state["compare_results"] = cmp_res

        cmp_res = st.session_state.get("compare_results", [])
        if cmp_res:
            import pandas as pd

            # Helper: podbarvení řádku dle plodiny
            def _highlight_crop(row):
                if row["Plodina"] == "Miscanthus":
                    return ["background-color: #DCEDC8"] * len(row)  # světle zelená
                elif row["Plodina"] == "SRC Vrba":
                    return ["background-color: #5D4037; color: white"] * len(row)  # tmavá hnědá
                return [""] * len(row)

            # Fallback: pokud user nemá vybranou plodinu v sekci 2, použij DEFAULT_COSTS
            from copy import deepcopy as _dc
            crop_params = {
                "Miscanthus": _dc(DEFAULT_COSTS["Miscanthus"]),
                "SRC Vrba":   _dc(DEFAULT_COSTS["SRC Vrba"]),
            }
            crop_params["Miscanthus"].update(editable_costs.get("Miscanthus", {}))
            crop_params["SRC Vrba"].update(editable_costs.get("SRC Vrba", {}))

            # ---------- 1) Tabulka vstupních parametrů ----------
            st.divider()
            st.markdown("### " + COMPARE_T["params_t"])
            params_rows = []
            for r in cmp_res:
                cp = crop_params["Miscanthus" if r["Plodina"] == "Miscanthus" else "SRC Vrba"]
                params_rows.append({
                    "Plodina":              r["Plodina"],
                    "Kvalita půdy":         r["Půda"],
                    "Y_min (t DM/ha)":      int(r["Y_min"]),
                    "Y_max (t DM/ha)":      int(r["Y_max"]),
                    "Životnost (let)":      int(cp["zivotnost"]),
                    "Default cena (€/t DM)": int(round(cp["prodejni_cena_start"])),
                    "Roční údržba (€/ha)":  int(round(cp["udrzba_rocni"])),
                })
            df_params = pd.DataFrame(params_rows)
            st.dataframe(
                df_params.style.apply(_highlight_crop, axis=1).format({
                    "Y_min (t DM/ha)":      "{:d}",
                    "Y_max (t DM/ha)":      "{:d}",
                    "Životnost (let)":      "{:d}",
                    "Default cena (€/t DM)": "{:d}",
                    "Roční údržba (€/ha)":  "{:d}",
                }),
                use_container_width=True, hide_index=True,
            )
            st.caption(T["compare_common_params"].format(
                zone=detected_zone_label, area=plocha_ha,
                subsidy=subsidy_pct, discount=discount_pct,
                rho=rho_input, drift=drift_pct,
                esc=cost_esc_pct, rent=pachtovne,
            ))

            # ---------- 2) Risk-return scatter ----------
            st.divider()
            st.markdown("### " + COMPARE_T["scatter_t"])
            st.info(COMPARE_T["ideal_note"])

            # Mapy pro barvy a tvary
            soil_colors = {
                "Optimální":  "#33691E",  # tmavě zelená
                "Průměrná":   "#FBC02D",  # žlutá
                "Neúrodná":   "#FB8C00",  # oranžová
                "Nevhodná":   "#9E9E9E",  # šedá
            }
            crop_symbols = {"Miscanthus": "circle", "SRC Vrba": "diamond"}

            fig_sc = go.Figure()
            for r in cmp_res:
                lbl = ("M" if r["Plodina"] == "Miscanthus" else "S") + " · " + r["Půda"][:4]
                size = 14 + 26 * max(0, min(1, r["P(NPV>0)"]))  # 14-40 px
                # Markowitz konvence: X = riziko (σ), Y = výnos (EAA)
                fig_sc.add_trace(go.Scatter(
                    x=[r["σ ročního CF"]], y=[r["EAA"]],
                    mode="markers+text",
                    name=lbl,
                    marker=dict(
                        symbol=crop_symbols[r["Plodina"]],
                        color=soil_colors[r["Půda"]],
                        size=size,
                        line=dict(color="#000", width=1),
                    ),
                    text=[lbl], textposition="top center",
                    textfont=dict(size=10),
                    hovertemplate=(
                        f"<b>{r['Plodina']} × {r['Půda']}</b><br>"
                        f"Mean EAA: {r['EAA']:,.0f} €/rok<br>"
                        f"σ ročního CF: {r['σ ročního CF']:,.0f} €/rok<br>"
                        f"VaR 5%: {r['VaR 5%']:,.0f} €<br>"
                        f"CVaR 5%: {r['CVaR 5%']:,.0f} €<br>"
                        f"P(NPV>0): {r['P(NPV>0)']:.1%}<br>"
                        f"<extra></extra>".replace(",", " ")
                    ),
                    showlegend=False,
                ))
            # Kvadrant guide: cross-hair na medianech
            mn = np.median([r["EAA"] for r in cmp_res])
            sd = np.median([r["σ ročního CF"] for r in cmp_res])
            fig_sc.add_hline(y=mn, line=dict(dash="dot", color="#888", width=1),
                              annotation_text=T["compare_median_eaa"],
                              annotation_position="left")
            fig_sc.add_vline(x=sd, line=dict(dash="dot", color="#888", width=1),
                              annotation_text=T["compare_median_sigma"],
                              annotation_position="top")
            # Zvýraznit "ideální" oblast (vlevo nahoře = nízké σ + vysoké EAA)
            x_min = min(r["σ ročního CF"] for r in cmp_res) * 0.9
            y_max_v = max(r["EAA"] for r in cmp_res) * 1.05
            fig_sc.add_shape(type="rect", x0=x_min, x1=sd, y0=mn, y1=y_max_v,
                              fillcolor="rgba(76,175,80,0.06)", line=dict(width=0),
                              layer="below")
            fig_sc.update_layout(
                xaxis_title=COMPARE_T["scatter_x"],
                yaxis_title=COMPARE_T["scatter_y"],
                height=560, margin=dict(t=30, b=50, l=10, r=10),
                hovermode="closest",
            )
            st.plotly_chart(fig_sc, use_container_width=True, key="compare_scatter")

            # Legenda manuálně (čistý markdown bez div, aby se renderoval bold)
            legend_md = T["compare_legend"]
            st.markdown(legend_md)

            # ---------- 3) Downside risk scatter (CVaR vs EAA) -----------
            st.divider()
            st.markdown("### " + COMPARE_T["scatter_cvar_t"])
            st.info(COMPARE_T["scatter_cvar_note"])

            fig_cv = go.Figure()
            for r in cmp_res:
                lbl  = ("M" if r["Plodina"] == "Miscanthus" else "S") + " · " + r["Půda"][:4]
                size = 14 + 26 * max(0, min(1, r["P(NPV>0)"]))
                fig_cv.add_trace(go.Scatter(
                    x=[r["CVaR 5%"]], y=[r["EAA"]],
                    mode="markers+text",
                    name=lbl,
                    marker=dict(
                        symbol=crop_symbols[r["Plodina"]],
                        color=soil_colors[r["Půda"]],
                        size=size,
                        line=dict(color="#000", width=1),
                    ),
                    text=[lbl], textposition="top center",
                    textfont=dict(size=10),
                    hovertemplate=(
                        f"<b>{r['Plodina']} × {r['Půda']}</b><br>"
                        f"EAA: {r['EAA']:,.0f} €/rok<br>"
                        f"CVaR 5%: {r['CVaR 5%']:,.0f} €<br>"
                        f"VaR 5%: {r['VaR 5%']:,.0f} €<br>"
                        f"P(NPV>0): {r['P(NPV>0)']:.1%}<br>"
                        f"<extra></extra>".replace(",", " ")
                    ),
                    showlegend=False,
                ))
            # Median guides
            mn_eaa  = np.median([r["EAA"]     for r in cmp_res])
            md_cvar = np.median([r["CVaR 5%"] for r in cmp_res])
            fig_cv.add_hline(y=mn_eaa, line=dict(dash="dot", color="#888", width=1),
                              annotation_text=T["compare_median_eaa"],
                              annotation_position="left")
            fig_cv.add_vline(x=md_cvar, line=dict(dash="dot", color="#888", width=1),
                              annotation_text=T["compare_median_cvar"],
                              annotation_position="top")
            # "Ideální" oblast — pravý horní kvadrant (vysoké EAA + mírnější CVaR)
            x_max_v = max(r["CVaR 5%"] for r in cmp_res) * 1.05 if max(r["CVaR 5%"] for r in cmp_res) > 0 else max(r["CVaR 5%"] for r in cmp_res) * 0.95
            y_max_e = max(r["EAA"] for r in cmp_res) * 1.05
            fig_cv.add_shape(type="rect", x0=md_cvar, x1=x_max_v, y0=mn_eaa, y1=y_max_e,
                              fillcolor="rgba(76,175,80,0.06)", line=dict(width=0),
                              layer="below")
            fig_cv.update_layout(
                xaxis_title=COMPARE_T["scatter_cvar_x"],
                yaxis_title=COMPARE_T["scatter_cvar_y"],
                height=560, margin=dict(t=30, b=50, l=10, r=10),
                hovermode="closest",
            )
            st.plotly_chart(fig_cv, use_container_width=True, key="compare_scatter_cvar")
            st.markdown(legend_md)

            # ---------- 3) Souhrnná tabulka ----------
            st.divider()
            st.markdown("### " + COMPARE_T["table_t"])
            df_cmp = pd.DataFrame(cmp_res)
            df_disp = df_cmp.drop(columns=["Y_min", "Y_max"], errors="ignore").copy()
            for col in ["mean NPV", "σ ročního CF", "VaR 5%", "CVaR 5%", "EAA"]:
                if col in df_disp.columns:
                    df_disp[col] = df_disp[col].apply(
                        lambda v: f"{v:,.0f} €".replace(",", " "))
            for col in ["P(NPV>0)", "Payback prob."]:
                if col in df_disp.columns:
                    df_disp[col] = df_disp[col].apply(lambda v: f"{v:.1%}")
            st.dataframe(
                df_disp.style.apply(_highlight_crop, axis=1),
                use_container_width=True, hide_index=True,
            )

            # Excel export srovnání
            try:
                from io import BytesIO
                bio = BytesIO()
                with pd.ExcelWriter(bio, engine="openpyxl") as wr:
                    df_cmp.to_excel(wr, sheet_name=T["compare_excel_sheet"], index=False)
                bio.seek(0)
                st.download_button(
                    label=COMPARE_T["dl_btn"],
                    data=bio.getvalue(),
                    file_name=f"BioFarm_srovnani_{detected_zone_label[:10].replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_compare",
                )
            except Exception:
                pass

# ----- TAB 6: DIVERZIFIKACE (vždy aktivní – live z formuláře) -------------
DIV_T = {
    "cs": {
        "header":    "Diverzifikační optimalizátor",
        "desc":      ("Najde optimální poměr Miscanthus : SRC vrba pro zvolený "
                      "ukazatel rizika. Používá **aktuální** hodnoty z formuláře výše "
                      "(klimatická zóna, kvalita půdy, plocha, ceny, dotace, korelace, drift)."),
        "metric":    "Co optimalizovat?",
        "opts":      {"profit": "Maximalizovat průměrný NPV",
                       "eaa":    "Maximalizovat EAA (roční ekvivalent)",
                       "var":    "Maximalizovat VaR 5 % (minimalizovat ztrátu)",
                       "cvar":   "Maximalizovat CVaR 5 % (minimalizovat ztrátu)",
                       "std":    "Minimalizovat směrodatnou odchylku CF (nejstabilnější)"},
        "btn":       "Najít optimální poměr",
        "spinner":   "Hledám optimum napříč 51 poměry…",
        "warn_crops":"Pro diverzifikaci zaškrtněte v sekci 2 **obě plodiny** (Miscanthus + SRC).",
        "inputs_h":  "Vstupy použité pro výpočet",
        "opt_kpi":   "Optimální poměr",
        "metric_val":"Hodnota metriky v optimu",
        "vs_pure_m": "vs. čistě Miscanthus",
        "vs_pure_s": "vs. čistě SRC",
        "field":     "Vizualizace pole (rozdělení plochy)",
        "frontier":  "Křivka napříč všemi poměry (efficient frontier)",
        "x_axis":    "% Miscanthus v mixu",
    },
    "en": {
        "header":    "Diversification Optimizer",
        "desc":      ("Finds the optimal Miscanthus : SRC willow ratio for the chosen "
                      "risk metric. Uses **current** form values above "
                      "(climate zone, soil, area, costs, subsidy, correlation, drift)."),
        "metric":    "What to optimize?",
        "opts":      {"profit": "Maximize mean NPV",
                       "eaa":    "Maximize EAA (annual equivalent)",
                       "var":    "Maximize VaR 5 % (minimize loss)",
                       "cvar":   "Maximize CVaR 5 % (minimize loss)",
                       "std":    "Minimize CF standard deviation (most stable)"},
        "btn":       "Find optimal ratio",
        "spinner":   "Searching across 51 ratios…",
        "warn_crops":"To run diversification, tick **both crops** in section 2 (Miscanthus + SRC).",
        "inputs_h":  "Inputs used for the calculation",
        "opt_kpi":   "Optimal ratio",
        "metric_val":"Metric value at optimum",
        "vs_pure_m": "vs. pure Miscanthus",
        "vs_pure_s": "vs. pure SRC",
        "field":     "Field visualization (area split)",
        "frontier":  "Curve across all ratios (efficient frontier)",
        "x_axis":    "% Miscanthus in mix",
    },
}[st.session_state["lang"]]

with tab_div:
    st.markdown(f"### {DIV_T['header']}")
    st.markdown(DIV_T["desc"])

    div_ready = ("Miscanthus" in editable_costs) and ("SRC Vrba" in editable_costs)

    if not div_ready:
        st.warning(DIV_T["warn_crops"])
    else:
        try:
            div_misc_y = YIELD_DATA[detected_zone_key][soil_key]["M_giganteus"]
            div_src_y  = YIELD_DATA[detected_zone_key][soil_key]["SRC"]
        except KeyError:
            div_misc_y, div_src_y = [5, 10], [5, 10]

        div_misc_params = editable_costs["Miscanthus"]
        div_src_params  = editable_costs["SRC Vrba"]

        # ---------- Hezčí 2-sloupcový info-box vstupů ----------
        with st.expander("📋 " + DIV_T["inputs_h"], expanded=False):
            zone_lbl = T["zones"][detected_zone_key]
            soil_lbl = T["soil_opts"][SOIL_KEYS.index(soil_key)]

            # 3 skupiny vstupů: lokace+pole / parametry / plodiny
            section_titles = (T["div_loc_section"], T["div_param_section"],
                              T["div_crop_section"])
            yr_unit = T["div_yr_unit"]
            yr_per = T["div_yr_per"]

            def _row(label, value, icon=""):
                ico = f"<span style='display:inline-block;width:22px'>{icon}</span>" if icon else ""
                return (f"<div style='display:flex;justify-content:space-between;"
                        f"padding:8px 12px;border-bottom:1px solid #EEF0EA'>"
                        f"<span style='color:#555'>{ico}{label}</span>"
                        f"<span style='font-weight:600;color:#1B2733'>{value}</span></div>")

            def _section(title, rows):
                body = "".join(_row(l, v, i) for l, v, i in rows)
                return (f"<div style='background:#F7F8F4;border-radius:10px;"
                        f"border:1px solid #E0E2D9;overflow:hidden;margin-bottom:12px'>"
                        f"<div style='background:#33691E;color:white;padding:8px 12px;"
                        f"font-weight:600;font-size:13px;letter-spacing:0.4px'>"
                        f"{title}</div>{body}</div>")

            loc_rows = [
                (T["div_loc_zone"], zone_lbl, "🌍"),
                (T["div_loc_soil"], soil_lbl, "🪨"),
                (T["div_loc_area"], f"{plocha_ha:.0f} ha", "📐"),
            ]
            param_rows = [
                (T["div_param_discount"], f"{discount_pct:.1f} %", "💰"),
                (T["div_param_subsidy"], f"{subsidy_pct} %", "🏛️"),
                (T["div_param_rent"],
                 (f"{pachtovne} €/ha/{yr_unit}" if pachtovne > 0
                  else T["div_param_rent_owned"]),
                 "🏡"),
                (T["div_param_rho"], f"{rho_input:+.2f}", "🔗"),
                (T["div_param_drift"], f"{drift_pct:+.1f} {yr_per}", "📈"),
                (T["div_param_costesc"], f"{cost_esc_pct:+.1f} {yr_per}", "📊"),
            ]
            # Sadební materiál = hustota × cena
            misc_sadba = div_misc_params["hustota_vysadby"] * div_misc_params["cena_sadby_ks"]
            src_sadba  = div_src_params["hustota_vysadby"]  * div_src_params["cena_sadby_ks"]
            crop_rows = [
                (T["div_crop_misc_yield"], f"{div_misc_y[0]}–{div_misc_y[1]} t/ha", ""),
                (T["div_crop_misc_life"], f"{int(div_misc_params['zivotnost'])} {yr_unit}", ""),
                (T["div_crop_misc_seed"], f"{misc_sadba:,.0f} €/ha".replace(",", " "), ""),
                (T["div_crop_misc_price"], f"{div_misc_params['prodejni_cena_start']:.0f} €/t", ""),
                (T["div_crop_src_yield"], f"{div_src_y[0]}–{div_src_y[1]} t/ha", ""),
                (T["div_crop_src_life"], f"{int(div_src_params['zivotnost'])} {yr_unit}", ""),
                (T["div_crop_src_seed"], f"{src_sadba:,.0f} €/ha".replace(",", " "), ""),
                (T["div_crop_src_price"], f"{div_src_params['prodejni_cena_start']:.0f} €/t", ""),
            ]

            info_col1, info_col2 = st.columns([1, 1])
            with info_col1:
                st.markdown(_section(section_titles[0], loc_rows), unsafe_allow_html=True)
                st.markdown(_section(section_titles[1], param_rows), unsafe_allow_html=True)
            with info_col2:
                st.markdown(_section(section_titles[2], crop_rows), unsafe_allow_html=True)

        # ---------- Multi-select metrik (bez radio) ----------
        st.markdown(f"**{DIV_T['metric']}** "
                    f"<span style='color:#888;font-size:13px'>"
                    f"{T['div_metric_subtitle']}"
                    f"</span>",
                    unsafe_allow_html=True)
        sel_cols = st.columns(len(DIVERSIFY_METRICS))
        div_selected_metrics = []
        for i, mk in enumerate(DIVERSIFY_METRICS):
            with sel_cols[i]:
                # Defaultně NPV vybraný
                if st.checkbox(DIV_T["opts"][mk], value=(mk == "profit"),
                                key=f"div_chk_{mk}"):
                    div_selected_metrics.append(mk)

        if not div_selected_metrics:
            st.info(T["div_pick_metric"])

        if st.button(DIV_T["btn"], type="primary", key="btn_diversify",
                     disabled=(not div_selected_metrics)):
            with st.spinner(DIV_T["spinner"]):
                div_n_sim = n_sim
                div_res = run_diversification(
                    misc_params=div_misc_params, misc_y_bounds=div_misc_y,
                    src_params=div_src_params,   src_y_bounds=div_src_y,
                    total_area_ha=plocha_ha, n_sim=div_n_sim,
                    subsidy_perc=subsidy_pct / 100.0, discount_rate=discount_pct / 100.0,
                    rho=rho_input, drift=drift_rate,
                    cost_escalation=cost_escalation_rate,
                    pachtovne=pachtovne,
                    soil_quality=soil_key,
                )

            # ---------- Pro každou vybranou metriku zobrazíme blok pod sebou ----------
            for mk_idx, div_metric_key in enumerate(div_selected_metrics):
                opt_idx, opt_val = find_optimum(div_res, div_metric_key)
                opt_pct_m  = div_res["pct_misc"][opt_idx]
                opt_pct_s  = 100 - opt_pct_m
                pure_m_val = div_res[div_metric_key][-1]
                pure_s_val = div_res[div_metric_key][0]

                # Hlavička metriky – modrý gradient (sladěný s frontier křivkou)
                st.markdown(
                    f"<div style='margin-top:18px;background:linear-gradient(90deg,#0D47A1,#1976D2);"
                    f"color:white;padding:10px 16px;border-radius:8px;font-weight:600;font-size:1.1rem'>"
                    f"📊 {DIV_T['opts'][div_metric_key]}"
                    f"</div>", unsafe_allow_html=True)

                dk1, dk2, dk3, dk4 = st.columns(4)
                dk1.metric(DIV_T["opt_kpi"], f"{opt_pct_m} % M / {opt_pct_s} % SRC")
                dk2.metric(DIV_T["metric_val"], fmt_eur_cs(opt_val))
                dk3.metric(DIV_T["vs_pure_m"],
                           f"{'+' if (opt_val-pure_m_val)>=0 else '−'}{fmt_eur_cs(abs(opt_val-pure_m_val))}",
                           delta_color="normal")
                dk4.metric(DIV_T["vs_pure_s"],
                           f"{'+' if (opt_val-pure_s_val)>=0 else '−'}{fmt_eur_cs(abs(opt_val-pure_s_val))}",
                           delta_color="normal")

                # Vizualizace pole – pod sebou s frontier grafem
                area_M = plocha_ha * opt_pct_m / 100.0
                area_S = plocha_ha * opt_pct_s / 100.0
                fig_field = go.Figure()
                fig_field.add_trace(go.Bar(
                    y=["🌾"], x=[area_M], orientation="h",
                    name="Miscanthus",
                    marker=dict(color=COLOR_MISC, line=dict(color=COLOR_MISC_DARK, width=2)),
                    text=f"<b>{opt_pct_m} %</b><br>{area_M:.1f} ha", textposition="inside",
                    textfont=dict(color="white", size=15),
                ))
                fig_field.add_trace(go.Bar(
                    y=["🌾"], x=[area_S], orientation="h",
                    name=T["crop_src_chart_name"],
                    marker=dict(color=COLOR_SRC, line=dict(color=COLOR_SRC_DARK, width=2)),
                    text=f"<b>{opt_pct_s} %</b><br>{area_S:.1f} ha", textposition="inside",
                    textfont=dict(color="white", size=15),
                ))
                fig_field.update_layout(
                    barmode="stack", height=200,
                    title=DIV_T["field"], title_x=0,
                    xaxis=dict(title=T["div_field_area"], range=[0, plocha_ha]),
                    yaxis=dict(showticklabels=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
                    margin=dict(t=50, b=40, l=10, r=10),
                )
                st.plotly_chart(fig_field, use_container_width=True,
                                 key=f"div_field_{div_metric_key}_{mk_idx}")

                fig_front = go.Figure()
                fig_front.add_trace(go.Scatter(
                    x=div_res["pct_misc"], y=div_res[div_metric_key],
                    mode="lines+markers", name=DIV_T["opts"][div_metric_key],
                    line=dict(color="#1565C0", width=2.5),
                    marker=dict(size=6, color="#1565C0"),
                ))
                fig_front.add_trace(go.Scatter(
                    x=[opt_pct_m], y=[opt_val],
                    mode="markers", name="Optimum",
                    marker=dict(size=18, color="#D6483D", symbol="star",
                                line=dict(color="white", width=2)),
                ))
                fig_front.update_layout(
                    title=DIV_T["frontier"], title_x=0,
                    xaxis_title=DIV_T["x_axis"],
                    yaxis_title="€",
                    height=380,
                    xaxis=dict(dtick=10, range=[-2, 102]),
                    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
                    margin=dict(t=50, b=80),
                )
                st.plotly_chart(fig_front, use_container_width=True,
                                 key=f"div_frontier_{div_metric_key}_{mk_idx}")

                if mk_idx < len(div_selected_metrics) - 1:
                    st.divider()