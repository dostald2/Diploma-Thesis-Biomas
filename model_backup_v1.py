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
    "Miscanthus": {
        "zalozeni": 2575, "sadba_podil": 2075, "udrzba": 189,
        "sklizen_per_tuna": 25, "prodejni_cena_start": 80,
        "riziko_fail": 0.05, "zivotnost": 25,
    },
    "SRC Vrba": {
        "zalozeni": 2600, "sadba_podil": 1800, "udrzba": 280,
        "sklizen_per_tuna": 28, "prodejni_cena_start": 75,
        "riziko_fail": 0.05, "zivotnost": 30,
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
    # --- 5 nákladových / výnosových ---
    "prodejni_cena_start": {"cs": "Prodejní cena (EUR/t)",       "en": "Selling price (EUR/t)",
                            "cs_short": "Prodejní cena",          "en_short": "Selling price"},
    "yield_potential":     {"cs": "Výnosový potenciál (t/ha)",   "en": "Yield potential (t/ha)",
                            "cs_short": "Výnos",                  "en_short": "Yield"},
    "zalozeni":            {"cs": "Náklady na založení (EUR/ha)","en": "Establishment cost (EUR/ha)",
                            "cs_short": "Založení",               "en_short": "Establishment"},
    "udrzba":              {"cs": "Roční údržba (EUR/ha)",       "en": "Annual maintenance (EUR/ha)",
                            "cs_short": "Údržba",                 "en_short": "Maintenance"},
    "sklizen_per_tuna":    {"cs": "Náklady na sklizeň (EUR/t)",  "en": "Harvest cost (EUR/t)",
                            "cs_short": "Sklizeň",                "en_short": "Harvest cost"},
    # --- 5 rizikových / strukturálních ---
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
    "decarbon_drift":      {"cs": "Dekarbonizační drift (%/rok)", "en": "Decarbonization drift (%/yr)",
                            "cs_short": "Drift",                  "en_short": "Drift"},
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


def run_param_sensitivity(param_key, base_params, y_bounds, crop_type, n_sim,
                          years, subsidy_perc, area_ha, discount_rate,
                          weather_prob=0.05, src_tech="Direct Chip",
                          rho=0.0, drift=0.010):
    """
    Spustí simulaci pro jeden parametr v 11 krocích.
    Pro většinu parametrů: ±10 % relativně. Pro 'decarbon_drift': absolutní −5 % až +5 %.
    Vrací dict s klíči "pct", "profit", "eaa", "std", "var", "cvar",
    "payback_prob", "payback_yr" (každý list 11 hodnot).
    """
    profits, eaas, stds, vars_, cvars = [], [], [], [], []
    payback_probs, payback_years = [], []

    is_drift = (param_key == "decarbon_drift")
    steps_iter = SA_DRIFT_STEPS if is_drift else SA_STEPS

    for step in steps_iter:
        factor = 1.0 + step
        p = dict(base_params)
        yb = list(y_bounds)
        sim_years = years
        sub = subsidy_perc
        r = discount_rate
        wp = weather_prob
        d = drift

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
            # 'step' je už absolutní hodnota driftu (−0.05 až +0.05), ne odchylka
            d = float(step)
        elif param_key in p:
            p[param_key] = base_params[param_key] * factor

        if crop_type == "misc":
            cf, _, _ = simulate_miscanthus(n_sim, sim_years, p, yb, sub, wp,
                                            rho=rho, drift=d)
        else:
            cf, _, _ = simulate_src(n_sim, sim_years, p, yb, src_tech, sub, wp,
                                     rho=rho, drift=d)

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
        "is_absolute_axis": is_drift,
    }


SA_LINE_COLORS = [
    "#1f77b4",   # modrá  – Prodejní cena
    "#ff7f0e",   # oranžová – Výnosový potenciál
    "#2ca02c",   # zelená – Náklady na založení
    "#d62728",   # červená – Roční údržba
    "#9467bd",   # fialová – Náklady na sklizeň
]


def make_sensitivity_line_chart(all_sa, metric_key, title, yaxis_label,
                                param_labels_short, lang_code):
    """
    Spojnicový graf: X = odchylka parametru (%) nebo absolutní hodnota (drift), Y = metrika.
    Pokud některý parametr má is_absolute_axis=True, vykreslí se na sekundární ose X nahoře.
    """
    # Rozdělíme parametry: relativní odchylka (dolní osa) vs absolutní (horní osa)
    rel_keys = [k for k, v in all_sa.items() if not v.get("is_absolute_axis")]
    abs_keys = [k for k, v in all_sa.items() if v.get("is_absolute_axis")]

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
            name=param_labels_short[pk] + (" (abs %)" if lang_code == "cs" else " (abs %)"),
            line=dict(color=SA_LINE_COLORS[color_idx % len(SA_LINE_COLORS)],
                      width=2.5, dash="dot"),
            marker=dict(size=5, symbol="diamond"),
            xaxis="x2",
        ))
        color_idx += 1

    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

    layout_kwargs = dict(
        title=dict(text=title, font=dict(size=15), x=0.5, xanchor="center"),
        yaxis_title=yaxis_label,
        xaxis=dict(
            title="Odchylka parametru (%)" if lang_code == "cs" else "Parameter deviation (%)",
            dtick=2, range=[-11, 11],
        ),
        legend=dict(
            orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        height=480,
        margin=dict(t=70, b=110),
    )
    if abs_keys:
        layout_kwargs["xaxis2"] = dict(
            title="Drift (%/rok – absolutně)" if lang_code == "cs" else "Drift (%/yr – absolute)",
            overlaying="x", side="top",
            dtick=1, range=[-5.5, 5.5],
        )
    fig.update_layout(**layout_kwargs)
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


def gompertz_growth(t_arr, y_max=1.0, b=6.0, c=0.95):
    # b=6.0, c=0.95 kalibrováno vizuálně dle Janota et al. 2023:
    #   rok 1 nízký, rok 3–4 výrazný nástup, rok 5–6 plný výnos, plateau do roku 14
    growth  = np.exp(-b * np.exp(-c * t_arr))
    # Pokles od roku 14 pro Miscanthus: 4 %/rok
    # Clifton-Brown et al. (2024): irská 16letá data ukazují pokles od roku 10–12,
    # rate ~5–6 %/rok. Hodnota 4 %/rok od roku 14 je kompromis mezi dobře řízenou
    # plantáží a průměrnými podmínkami. Dolní limit 55 % = stabilizace dle dánských dat.
    decline = np.where(t_arr > 14, 1.0 - 0.04*(t_arr - 14), 1.0)
    return growth * np.clip(decline, 0.55, 1.0)


def src_yield_curve(t_arr):
    """
    Relativní roční produkce suché hmoty SRC vrby v průběhu životnosti plantáže.

    Model: rise → plateau (bez systematického poklesu).
    Empirická data (Stolarski 2019, 12 let; Castellano Albors 2025, globální dataset)
    ukazují, že po počátečním nárůstu se výnosy SRC ustálí na plateau bez
    degenerativního trendu – pokles v jednotlivých letech je způsoben počasím,
    ne stárnutím plantáže.

    Tvar: Gompertz nástup do plateau na 1.0.
      - Gompertz: b=3.0, c=0.40 → rok 3 ~37 %, rok 5 ~69 %, rok 8 ~92 %,
        rok 11 ~98 %, pak plateau ≈ 1.0
      - Mírný pokles od roku 25: rate=0.015/rok (konzervativní odhad pro
        velmi staré plantáže, kde může začít degenerace kořenového systému)
      - Rok 30 ≈ 93 % maxima (vs. původních 35 %)
    """
    t = np.asarray(t_arr, dtype=float)
    growth = np.exp(-3.0 * np.exp(-0.40 * t))
    # Mírný pokles pouze u velmi starých plantáží (25+ let)
    late_decline = np.where(t > 25, np.exp(-0.015 * (t - 25)), 1.0)
    return growth * late_decline


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


def simulate_miscanthus(n_sim, years, params, yield_bounds, subsidy_perc,
                        weather_prob=0.05, rho=0.0, drift=0.010):
    y_max_sim      = np.clip(np.random.normal(np.mean(yield_bounds),
                             (yield_bounds[1]-yield_bounds[0])/4, n_sim), 0, 60)
    gompertz_curve = gompertz_growth(np.arange(1, years+1, dtype=float))
    wu, pu         = generate_correlated_shocks(years, n_sim, rho=rho)
    # scale kalibrován na meziroční rozdíl ~4 t/ha dle vědců (2025)
    # Na průměru ~11.5 t/ha → relativní scale = 4 / (2 * 11.5) ≈ 0.17
    wm             = np.clip(norm.ppf(wu, loc=1.0, scale=0.17), 0.5, 1.5)
    # Stresový rok: 1 z 20 (5 %), výnos klesne na 50 % (×0.50) dle vědců
    wm[np.random.rand(years, n_sim) < weather_prob] *= 0.50
    yields         = gompertz_curve[:, np.newaxis] * y_max_sim[np.newaxis, :] * wm
    # Ornstein-Uhlenbeck mean-reverting cenový model (Miscanthus)
    # mu = startovní cena (slouží i jako dlouhodobý průměr trhu)
    psc = norm.ppf(pu)
    prices = ou_price_process(
        years, n_sim,
        p0=params["prodejni_cena_start"],
        mu=params["prodejni_cena_start"],  # dlouhodobý průměr = startovní cena
        theta=0.25, sigma=15.0, drift=drift,
        correlated_noise=psc,
        clip_lo=45, clip_hi=160,
    )
    cf = yields*prices - params["udrzba"] - yields*params["sklizen_per_tuna"]
    capex = np.random.normal(params["zalozeni"], 200, n_sim)

    # Selhání plantáže: 5 % pravděpodobnost (1 z 20) dle vědců 2025
    fail = np.random.binomial(1, params["riziko_fail"], n_sim)  # 0 nebo 1

    # Pro simulace kde nastalo selhání: 50 % se obnoví, 50 % se nevyplatí
    renew = np.random.binomial(1, 0.5, n_sim)  # 0=nevyplatí, 1=obnova

    # Obnova (fail=1, renew=1): extra +60 % CAPEX, výnosy pokračují normálně
    replant_cost = fail * renew * 0.60 * capex

    # Bez obnovy (fail=1, renew=0): nulové výnosy a nulové provozní náklady
    # → vynulujeme cf pro roky 1+ u těchto simulací
    abandoned = (fail == 1) & (renew == 0)  # boolean maska
    cf[:, abandoned] = 0.0  # všechny roky = 0 (žádný příjem ani náklady)

    # Rok 0: CAPEX + případná obnova, sníženo o dotaci
    cf[0, :] -= (capex + replant_cost) * (1 - subsidy_perc)

    return cf, yields, prices


def simulate_src(n_sim, years, params, yield_bounds, tech_type, subsidy_perc,
                 weather_prob=0.05, rho=0.0, drift=0.010):
    # Maximální roční výnos pro každou simulaci (různé farmy)
    y_max_sim = np.clip(
        np.random.normal(np.mean(yield_bounds), (yield_bounds[1]-yield_bounds[0])/4, n_sim),
        0, 40
    )

    # Výnosová křivka dle Janota et al. 2023 (česká data, Fig. 7)
    # Tvar: nástup → plateau (roky 12–20) → pokles od roku 20
    t_arr  = np.arange(1, years + 1, dtype=float)
    curve  = src_yield_curve(t_arr)   # shape (years,)

    # Roční produkce = křivka × y_max × weather
    wu, pu = generate_correlated_shocks(years, n_sim, rho=rho)
    # scale kalibrováno na meziroční rozdíl ~2.5 t/ha (vědci 2025)
    wm = np.clip(norm.ppf(wu, loc=1.0, scale=0.13), 0.5, 1.5)
    # Stresový rok: 5 % pravděpodobnost, výnos na 50 %
    wm[np.random.rand(years, n_sim) < weather_prob] *= 0.50

    # Roční přírůstek biomasy (t/ha) pro každý rok a simulaci
    annual_growth = curve[:, np.newaxis] * y_max_sim[np.newaxis, :] * wm

    # Akumulace a sklizeň
    harvested = np.zeros((years, n_sim))
    accum = np.zeros(n_sim)
    # První sklizeň rok 5, pak každé 3 roky: 5, 8, 11, ..., 29 (dle vědců 2025)
    harvest_years = set(range(5, years + 1, 3))
    for t in range(years):
        accum += annual_growth[t, :]
        if (t + 1) in harvest_years:
            harvested[t, :] = accum
            accum = np.zeros(n_sim)
    # Ornstein-Uhlenbeck mean-reverting cenový model (SRC)
    psc = norm.ppf(pu)
    prices = ou_price_process(
        years, n_sim,
        p0=params["prodejni_cena_start"],
        mu=params["prodejni_cena_start"],
        theta=0.25, sigma=12.0, drift=drift,  # nižší volatilita než Miscanthus (menší trh)
        correlated_noise=psc,
        clip_lo=35, clip_hi=150,
    )
    cf = np.zeros((years, n_sim)) - params["udrzba"]
    for t in range(years):
        if (t + 1) in harvest_years:
            v = harvested[t, :]
            cf[t, :] += v*prices[t, :] - v*params["sklizen_per_tuna"]
    capex = np.random.normal(params["zalozeni"], 150, n_sim)

    # Selhání plantáže: 5 % pravděpodobnost (1 z 20) dle vědců 2025
    fail = np.random.binomial(1, params["riziko_fail"], n_sim)

    # Pro simulace kde nastalo selhání: 50 % se obnoví, 50 % se nevyplatí
    renew = np.random.binomial(1, 0.5, n_sim)

    # Obnova (fail=1, renew=1): extra +60 % CAPEX, sklizeň pokračuje normálně
    replant_cost = fail * renew * 0.60 * capex

    # Bez obnovy (fail=1, renew=0): nulové výnosy a nulové provozní náklady
    abandoned = (fail == 1) & (renew == 0)
    cf[:, abandoned] = 0.0

    # Rok 0: CAPEX + případná obnova, sníženo o dotaci
    cf[0, :] -= (capex + replant_cost) * (1 - subsidy_perc)

    return cf, harvested, prices


def calculate_sensitivity_matrix(crop_type, years, base_params, y_bounds,
                                  src_tech, area_ha, n_sim, subsidy_perc,
                                  rho=0.0, drift=0.010):
    steps = 10
    xr, yr = np.linspace(0, 0.2, steps), np.linspace(0, 0.2, steps)
    zm = []
    for wp in yr:
        row = []
        for fp in xr:
            p = {**base_params, "riziko_fail": fp}
            if crop_type == "misc":
                cf, _, _ = simulate_miscanthus(n_sim, years, p, y_bounds, subsidy_perc,
                                                wp, rho=rho, drift=drift)
            else:
                cf, _, _ = simulate_src(n_sim, years, p, y_bounds, src_tech, subsidy_perc,
                                         wp, rho=rho, drift=drift)
            row.append(np.mean(np.std(cf * area_ha, axis=0)))
        zm.append(row)
    return xr, yr, np.array(zm)


# ===========================================================================
# DIVERZIFIKAČNÍ OPTIMALIZÁTOR – optimální mix Miscanthus + SRC
# ===========================================================================
DIVERSIFY_METRICS = ["profit", "eaa", "var", "cvar"]   # vše maximalizujeme
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
                         step_pct=DIVERSIFY_STEP_PCT):
    """
    Pro 51 poměrů Miscanthus:SRC (krok 2 %) spočítá distribuci kombinovaného
    portfolia a vrátí všechny metriky pro každý poměr. Mix sdílí společný
    realizovaný shock vektor (jedno pole = jedno počasí), proto nasazujeme
    společný numpy seed před každou dvojicí simulací.
    Plocha se rozdělí: area_M = total_area × pct_M, area_S = total_area × (1-pct_M).
    """
    misc_years = int(misc_params["zivotnost"])
    src_years  = int(src_params["zivotnost"])

    pct_grid = np.arange(0, 100 + step_pct, step_pct)   # 0, 2, …, 100
    out = {"pct_misc": pct_grid.tolist(),
           "profit": [], "eaa": [], "var": [], "cvar": [], "std": []}

    for pct in pct_grid:
        area_M = total_area_ha * (pct / 100.0)
        area_S = total_area_ha * (1.0 - pct / 100.0)

        # Společný seed → obě plodiny "vidí" stejnou meteorologickou realizaci
        seed = int(1_000_003 + pct)
        np.random.seed(seed)
        if area_M > 0:
            cf_m, _, _ = simulate_miscanthus(n_sim, misc_years, misc_params,
                                             misc_y_bounds, subsidy_perc,
                                             weather_prob, rho=rho, drift=drift)
            npv_m, eaa_m = _scenario_metrics(cf_m * area_M, misc_years, discount_rate)
        else:
            npv_m = np.zeros(n_sim); eaa_m = np.zeros(n_sim); cf_m = np.zeros((misc_years, n_sim))

        np.random.seed(seed + 1)
        if area_S > 0:
            cf_s, _, _ = simulate_src(n_sim, src_years, src_params,
                                       src_y_bounds, "Direct Chip", subsidy_perc,
                                       weather_prob, rho=rho, drift=drift)
            npv_s, eaa_s = _scenario_metrics(cf_s * area_S, src_years, discount_rate)
        else:
            npv_s = np.zeros(n_sim); eaa_s = np.zeros(n_sim); cf_s = np.zeros((src_years, n_sim))

        # Agregace na úroveň portfolia (sčítáme NPV/EAA per scénář)
        port_npv = npv_m + npv_s
        port_eaa = eaa_m + eaa_s

        var5  = float(np.percentile(port_npv, 5))
        mask  = port_npv <= var5
        cvar5 = float(port_npv[mask].mean()) if mask.any() else var5

        # Roční odchylka kombinovaného CF (orientačně, používáme delší horizont)
        max_y  = max(misc_years, src_years)
        cf_pad = np.zeros((max_y, n_sim))
        cf_pad[:misc_years, :] += cf_m * area_M
        cf_pad[:src_years,  :] += cf_s * area_S
        std_yr = float(np.mean(np.std(cf_pad, axis=0)))

        out["profit"].append(float(np.mean(port_npv)))
        out["eaa"].append(float(np.mean(port_eaa)))
        out["var"].append(var5)
        out["cvar"].append(cvar5)
        out["std"].append(std_yr)

    return out


def find_optimum(div_results, metric_key):
    """Najde index s maximální hodnotou metriky (pro VaR/CVaR = nejméně záporné)."""
    arr = np.array(div_results[metric_key])
    idx = int(np.argmax(arr))
    return idx, float(arr[idx])


# ===========================================================================
# HTML TABULKA – zaručené centrování bez závislosti na Streamlit CSS
# ===========================================================================
def _fmt_eur(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:,.0f}&nbsp;€"

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
        "Pravděpodobnost návratnosti", "VaR (5%)", "CVaR (5%)", "Prům. roční odchylka",
    ]
    headers = [crop] + [rc[k] for k in col_order]

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
.block-container { padding-top: 2.5rem !important; }

/* Tlačítka vlajek – menší, bez nadbytečného odsazení */
[data-testid="stButton"] button {
    font-size: 1.5rem !important;
    padding: 2px 8px !important;
    min-width: 0 !important;
    line-height: 1.3 !important;
    border-radius: 6px !important;
}

/* Tlačítka v col_flag vedle sebe přes flex */
[data-testid="column"]:last-child [data-testid="stVerticalBlock"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 6px !important;
    justify-content: flex-end !important;
    align-items: center !important;
    padding-top: 22px !important;
}

/* ---- Centrování st.data_editor (AG Grid) ---- */
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
st.set_page_config(layout="wide", page_title="BioFarm Simulator")
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
# SEKCE 1 – MAPA
# ===========================================================================
st.header(T["sec1_header"])
st.markdown(T["sec1_desc"])

fmap = folium.Map(location=[49.8, 15.5], zoom_start=8)
fmap.add_child(folium.LatLngPopup())
map_data = st_folium(fmap, height=600, use_container_width=True)

detected_zone_key = "Střední Evropa / Mírné pásmo"
real_rain, real_temp = 600.0, 10.0

# Uchováme BPEJ výsledek v session_state aby přežil rerun
if "bpej_result"      not in st.session_state: st.session_state["bpej_result"] = None
if "bpej_soil_key"    not in st.session_state: st.session_state["bpej_soil_key"] = None
if "last_clicked_pos" not in st.session_state: st.session_state["last_clicked_pos"] = None

BS = BPEJ_STRINGS[st.session_state["lang"]]

if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    current_pos = (round(lat, 5), round(lon, 5))

    # Spusť lookup jen při nové souřadnici
    if current_pos != st.session_state["last_clicked_pos"]:
        st.session_state["last_clicked_pos"] = current_pos
        st.session_state["bpej_result"]      = None
        st.session_state["bpej_soil_key"]    = None

    st.success(f"{T['coord_success']} {lat:.4f}, {lon:.4f}")

    # Klima data
    with st.spinner(T["spinner_climate"]):
        tv, rv = get_climate_data(lat, lon)
        if tv is not None:
            real_temp, real_rain = tv, rv
            detected_zone_key = determine_zone(lat, real_temp, real_rain)

    # BPEJ lookup (jen pokud ještě nemáme výsledek pro tuto souřadnici)
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

# Připrav BPEJ metriku
bpej_res     = st.session_state.get("bpej_result")
lang_code    = st.session_state["lang"]
bpej_src     = bpej_res.get("source") if bpej_res else None
bpej_fert    = bpej_res.get("fertility") if bpej_res else None
bpej_code    = bpej_res.get("bpej_code") if bpej_res else None

if bpej_src == "BPEJ/vumop" and bpej_fert:
    bpej_fmt     = f"{bpej_code[0]}.{bpej_code[1:3]}.{bpej_code[3:]}" if bpej_code else ""
    bpej_value   = f"{bpej_fert}  ({bpej_fmt})" if bpej_fmt else bpej_fert
    bpej_caption = BS["bpej_metric_cap_ok"]
    bpej_delta   = None
    # Mapování BPEJ 3 tříd → nové 4 kategorie půdy
    _bpej_to_soil = {"Velmi úrodná": "Optimální", "Úrodná": "Průměrná", "Neúrodná": "Neúrodná"}
    _mapped = _bpej_to_soil.get(bpej_fert)
    if _mapped and _mapped in SOIL_KEYS:
        st.session_state["bpej_soil_key"] = _mapped
elif bpej_src == "mimo_CR":
    bpej_value   = BS["bpej_metric_na"]
    bpej_caption = BS["bpej_metric_cap_out"]
    bpej_delta   = None
elif bpej_src == "chyba":
    err_msg = (bpej_res.get("error") or "")
    # Rozliš nezem. půdu od technické chyby
    if "nezem" in err_msg.lower() or "nenalezeno" in err_msg.lower():
        bpej_value   = BS["bpej_metric_noag"]
        bpej_caption = BS["bpej_metric_cap_no"]
    else:
        bpej_value   = BS["bpej_metric_err"]
        bpej_caption = BS["bpej_metric_cap_err"]
    bpej_delta   = None
else:
    # Ještě nenačteno (před prvním kliknutím)
    bpej_value   = "—"
    bpej_caption = "Klikněte do mapy" if lang_code == "cs" else "Click on the map"
    bpej_delta   = None

# 4 metriky v řadě – stejný formát pro všechny
ci1, ci2, ci3, ci4 = st.columns(4)
ci1.metric(T["metric_temp"], f"{real_temp:.1f} °C")
ci1.caption(T["metric_temp_cap"])
ci2.metric(T["metric_rain"], f"{real_rain:.0f} mm")
ci2.caption(T["metric_rain_cap"])
ci3.markdown(f"""
<div style="font-size:12px;color:#888;font-weight:400;margin-bottom:2px">{BS["climate_zone_label"]}</div>
<div style="font-size:16px;font-weight:600;line-height:1.3">{detected_zone_label}</div>
<div style="font-size:12px;color:#aaa;margin-top:3px">{BS["climate_zone_cap"]}</div>
""", unsafe_allow_html=True)
ci4.markdown(f"""
<div style="font-size:12px;color:#888;font-weight:400;margin-bottom:2px">{BS["bpej_metric_label"]}</div>
<div style="font-size:16px;font-weight:600;line-height:1.3">{bpej_value}</div>
<div style="font-size:12px;color:#aaa;margin-top:3px">{bpej_caption}</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===========================================================================
# SEKCE 2 – KONFIGURACE
# ===========================================================================
st.header(T["sec2_header"])

cp1, cp2 = st.columns(2)
with cp1:
    # Výchozí index – buď z BPEJ auto-detekce, nebo 1 (Úrodná)
    _bpej_key = st.session_state.get("bpej_soil_key")
    _default_soil_idx = (SOIL_KEYS.index(_bpej_key)
                         if _bpej_key and _bpej_key in SOIL_KEYS else 2)

    soil_idx  = st.selectbox(T["soil_quality"], range(4),
                              format_func=lambda i: T["soil_opts"][i],
                              index=_default_soil_idx)
    soil_key  = SOIL_KEYS[soil_idx]
    plocha_ha = st.number_input(T["area_label"], min_value=1.0, value=10.0, step=1.0)

with cp2:
    st.write(T["crops_label"])
    chk_gig = st.checkbox("Miscanthus giganteus", value=True)
    chk_src = st.checkbox("SRC Vrba (Willow)")
    plodiny = []
    if chk_gig: plodiny.append("M_giganteus")
    if chk_src: plodiny.append("SRC Vrba")

n_sim       = int(st.number_input(T["nsim_label"], min_value=1, max_value=50000,
                                   value=5000, step=1000))
subsidy_pct  = st.slider(T["subsidy_label"], 0, 100, 0, 5)
discount_pct = st.number_input(
    T.get("discount_label", "Diskontní sazba / Discount rate (%)"),
    min_value=0.0, max_value=20.0, value=0.0, step=0.1, format="%.1f",
    help=T.get("discount_help",
               "Reálná diskontní sazba pro výpočet NPV. 0 % = bez diskontování (nominální součet)."),
)

# Pokročilé stochastické parametry: korelace počasí↔cena a dekarbonizační drift
adv_c1, adv_c2 = st.columns(2)
with adv_c1:
    rho_input = st.number_input(
        T.get("rho_label", "Korelace počasí ↔ cena (ρ)"),
        min_value=-1.0, max_value=1.0, value=0.0, step=0.05, format="%.2f",
        help=T.get("rho_help",
                   "Záporná hodnota = natural hedge (vysoký výnos → nižší cena). "
                   "0 = nezávislé. Default 0 (revize 2026)."),
    )
with adv_c2:
    drift_pct = st.number_input(
        T.get("drift_label", "Dekarbonizační prémie (drift, %/rok)"),
        min_value=-5.0, max_value=5.0, value=1.0, step=0.1, format="%.1f",
        help=T.get("drift_help",
                   "Roční reálný růst dlouhodobé ceny biomasy "
                   "(odráží růst cen emisních povolenek a dekarbonizační politiky)."),
    )
drift_rate = drift_pct / 100.0

st.markdown("---")

# ===========================================================================
# SEKCE 3 – NÁKLADY
# ===========================================================================
st.header(T["sec3_header"])
st.markdown(f"<p style='text-align:center'>{T['sec3_desc']}</p>", unsafe_allow_html=True)

cost_keys_map = T["cost_keys"]
rev_cost_keys = {v: k for k, v in cost_keys_map.items()}
editable_costs = {}
show_misc = "M_giganteus" in plodiny
show_src  = "SRC Vrba"    in plodiny

if show_misc and show_src:
    cc1, cc2 = st.columns(2)
else:
    cc1 = cc2 = st.container()

if show_misc:
    with cc1:
        st.markdown("<h3 style='text-align:center'>Miscanthus giganteus</h3>",
                    unsafe_allow_html=True)
        dfm = pd.DataFrame.from_dict(DEFAULT_COSTS["Miscanthus"], orient="index",
                                      columns=[T["cost_col_label"]])
        dfm.index = dfm.index.map(cost_keys_map)
        em = st.data_editor(dfm, use_container_width=True, key="misc_edit")
        em.index = em.index.map(rev_cost_keys)
        editable_costs["Miscanthus"] = em[T["cost_col_label"]].to_dict()

if show_src:
    with cc2:
        st.markdown("<h3 style='text-align:center'>SRC Vrba (Willow)</h3>",
                    unsafe_allow_html=True)
        dfs = pd.DataFrame.from_dict(DEFAULT_COSTS["SRC Vrba"], orient="index",
                                      columns=[T["cost_col_label"]])
        dfs.index = dfs.index.map(cost_keys_map)
        es = st.data_editor(dfs, use_container_width=True, key="src_edit")
        es.index = es.index.map(rev_cost_keys)
        editable_costs["SRC Vrba"] = es[T["cost_col_label"]].to_dict()

st.markdown("---")

# ===========================================================================
# SEKCE 3.5 – CITLIVOSTNÍ ANALÝZA (výběr parametrů)
# ===========================================================================
SA = SENSITIVITY_STRINGS[st.session_state["lang"]]
st.header(SA["sa_header"])

lang_sa = st.session_state["lang"]
sa_all_keys = list(SENSITIVITY_PARAMS.keys())
sa_options = {k: v[lang_sa] for k, v in SENSITIVITY_PARAMS.items()}

# 2 řady po 5 checkboxech, žádný předvolený, max 5 najednou
sa_desc_max = (
    "Vyberte **max. 5** parametrů, jejichž vliv chcete analyzovat (±10 %)."
    if lang_sa == "cs" else
    "Select **up to 5** parameters whose impact you want to analyze (±10 %)."
)
st.markdown(sa_desc_max)

sa_row1_keys = sa_all_keys[:5]
sa_row2_keys = sa_all_keys[5:]
sa_selected = []

sa_cols1 = st.columns(5)
for col, pk in zip(sa_cols1, sa_row1_keys):
    with col:
        if st.checkbox(sa_options[pk], value=False, key=f"sa_chk_{pk}"):
            sa_selected.append(pk)

sa_cols2 = st.columns(5)
for col, pk in zip(sa_cols2, sa_row2_keys):
    with col:
        if st.checkbox(sa_options[pk], value=False, key=f"sa_chk_{pk}"):
            sa_selected.append(pk)

if len(sa_selected) > SA_MAX_SELECTED:
    warn_msg = (f"⚠️ Vybráno {len(sa_selected)} parametrů – maximum je {SA_MAX_SELECTED}. "
                "Budou použity pouze první vybrané."
                if lang_sa == "cs" else
                f"⚠️ {len(sa_selected)} parameters selected – max is {SA_MAX_SELECTED}. "
                "Only the first selected will be used.")
    st.warning(warn_msg)
    sa_selected = sa_selected[:SA_MAX_SELECTED]

st.markdown("---")

# ===========================================================================
# SEKCE 4 – SIMULACE
# ===========================================================================
if st.button(T["run_button"], type="primary", use_container_width=True):
    st.write(T["results_header"])
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
            cf, yields, prices = simulate_miscanthus(
                n_sim, int(params["zivotnost"]), params, y_bounds, subsidy_pct/100.0,
                rho=rho_input, drift=drift_rate)
            results[plodina] = {"cf": cf, "yields": yields, "prices": prices,
                                "years": int(params["zivotnost"]),
                                "params": params, "y_bounds": y_bounds, "type": "misc"}
        elif plodina == "SRC Vrba":
            params = editable_costs["SRC Vrba"]
            cf, yields, prices = simulate_src(
                n_sim, int(params["zivotnost"]), params, y_bounds, "Direct Chip", subsidy_pct/100.0,
                rho=rho_input, drift=drift_rate)
            results[plodina] = {"cf": cf, "yields": yields, "prices": prices,
                                "years": int(params["zivotnost"]),
                                "params": params, "y_bounds": y_bounds, "type": "src"}

        bar.progress((i+1)/len(plodiny))

    # -----------------------------------------------------------------------
    # GRAFY
    # -----------------------------------------------------------------------
    for plodina, data in results.items():
        st.markdown(f"## 🌿 {plodina}")

        cf_total = data["cf"] * plocha_ha
        yrs      = np.arange(1, data["years"]+1)
        mean_cf  = np.mean(cf_total,  axis=1)
        p5_cf    = np.percentile(cf_total,  5, axis=1)
        p95_cf   = np.percentile(cf_total, 95, axis=1)

        fig_cf = go.Figure()
        fig_cf.add_trace(go.Scatter(x=yrs, y=mean_cf, name=T["cf_mean"],
                                    line=dict(color="green", width=3)))
        fig_cf.add_trace(go.Scatter(x=yrs, y=p95_cf, name=T["cf_opt"],
                                    line=dict(width=0), showlegend=False))
        fig_cf.add_trace(go.Scatter(x=yrs, y=p5_cf,  name=T["cf_pes"],
                                    fill="tonexty", fillcolor="rgba(0,100,80,0.2)",
                                    line=dict(width=0), showlegend=False))
        fig_cf.update_layout(title=T["cf_title"].format(ha=plocha_ha),
                             xaxis_title=T["cf_xaxis"], yaxis_title=T["cf_yaxis"])
        st.plotly_chart(fig_cf, use_container_width=True, key=f"cf_{plodina}")

        # NPV s uživatelsky nastavenou diskontní sazbou (0 % = prostý součet)
        r = discount_pct / 100.0
        if r > 0:
            discount_factors = (1 + r) ** -np.arange(data["years"])
            total_profits = np.sum(cf_total * discount_factors[:, np.newaxis], axis=0)
        else:
            total_profits = np.sum(cf_total, axis=0)
        var_5          = np.percentile(total_profits, 5)
        cvar_5         = total_profits[total_profits <= var_5].mean()
        avg_yearly_std = np.mean(np.std(cf_total, axis=0))
        cum_cf         = np.cumsum(cf_total, axis=0)
        pb_years       = np.argmax(cum_cf > 0, axis=0)
        succ_pb        = pb_years[pb_years > 0]
        payback_prob   = len(succ_pb)/n_sim if n_sim > 0 else 0
        avg_payback    = float(np.mean(succ_pb)+1) if len(succ_pb) > 0 else None
        # Equivalent Annual Annuity – férové srovnání plodin s různou životností
        eaa_per_sim    = equivalent_annual_annuity(total_profits, data["years"], r)
        eaa_mean       = float(np.mean(eaa_per_sim))

        summary_data.append({
            "Plodina":                     plodina,
            "Průměrný Zisk":               np.mean(total_profits),
            "EAA (EUR/rok)":               eaa_mean,
            "Potenciál (95%)":             np.percentile(total_profits, 95),
            "Doba návratnosti (roky)":     avg_payback if avg_payback else float("nan"),
            "Pravděpodobnost návratnosti": payback_prob,
            "VaR (5%)":                    var_5,
            "CVaR (5%)":                   cvar_5,
            "Prům. roční odchylka":        avg_yearly_std,
        })

        ch1, ch2 = st.columns(2)
        with ch1:
            fh = go.Figure(data=[go.Histogram(x=total_profits, nbinsx=50,
                                               name=T["hist_name"], marker_color="#1f77b4")])
            fh.update_layout(title=T["hist_title"],
                             xaxis_title=T["hist_xaxis"], yaxis_title=T["hist_yaxis"])
            st.plotly_chart(fh, use_container_width=True, key=f"hist_{plodina}")
        with ch2:
            fp = go.Figure(data=[go.Histogram(x=succ_pb+1, name=T["payback_name"],
                                               marker_color="#2ca02c")])
            fp.update_layout(title=T["payback_title"],
                             xaxis_title=T["payback_xaxis"], yaxis_title=T["payback_yaxis"])
            st.plotly_chart(fp, use_container_width=True, key=f"payback_{plodina}")

        st.subheader(T["kpi_header"])
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(T["kpi_avg_profit"], f"{np.mean(total_profits):,.0f} €")
        k2.metric(T.get("kpi_eaa", "Roční ekvivalent (EAA)"), f"{eaa_mean:,.0f} €/rok")
        k3.metric(T["kpi_potential"],  f"{np.percentile(total_profits, 95):,.0f} €")
        k4.metric(T["kpi_payback"],
                  f"{avg_payback:.2f} {T['kpi_payback_unit']}" if avg_payback else "N/A")
        k5.metric(T["kpi_payback_prob"],
                  f"{payback_prob:.1%}" if payback_prob >= 0.01 else f"{payback_prob:.4%}")

        st.subheader(T["risk_header"])
        r1, r2, r3 = st.columns(3)
        r1.metric(T["risk_std"],  f"{avg_yearly_std:,.0f} €")
        r2.metric(T["risk_var"],  f"{var_5:,.0f} €",
                  delta_color="inverse", help=T["risk_var_help"])
        r3.metric(T["risk_cvar"], f"{cvar_5:,.0f} €",
                  delta_color="inverse", help=T["risk_cvar_help"])

        st.subheader(T["yield_header"])
        my   = np.mean(data["yields"],  axis=1)
        p5y  = np.percentile(data["yields"],  5, axis=1)
        p95y = np.percentile(data["yields"], 95, axis=1)
        fy   = go.Figure()
        fy.add_trace(go.Scatter(x=yrs, y=my,   name=T["yield_mean"], line=dict(color="#ff7f0e")))
        fy.add_trace(go.Scatter(x=yrs, y=p95y, name=T["cf_opt"],     line=dict(width=0), showlegend=False))
        fy.add_trace(go.Scatter(x=yrs, y=p5y,  name=T["cf_pes"],
                                fill="tonexty", fillcolor="rgba(255,127,14,0.2)",
                                line=dict(width=0), showlegend=False))
        if data["type"] == "misc":
            t_ref = np.arange(1, data["years"]+1, dtype=float)
            fy.add_trace(go.Scatter(x=yrs, y=gompertz_growth(t_ref)*np.mean(data["y_bounds"]),
                                    name=T["gompertz_label"],
                                    line=dict(color="gray", dash="dash", width=1)))
        fy.update_layout(title=T["yield_title"].format(crop=plodina),
                         xaxis_title=T["yield_xaxis"], yaxis_title=T["yield_yaxis"])
        st.plotly_chart(fy, use_container_width=True, key=f"yield_{plodina}")

        if data["type"] == "src":
            st.subheader(T["learn_header"])
            # Sklizňové roky: 5, 8, 11, 14, 17, 20, 23, 26, 29
            src_harvest_yrs = [5 + i*3 for i in range(9)]
            # Relativní výnos v každém sklizňovém roce dle bell curve
            curve_vals = src_yield_curve(np.array(src_harvest_yrs, dtype=float))
            peak_val   = curve_vals.max()
            rm_pct     = (curve_vals / peak_val * 100).tolist()
            rl = [T["learn_rot_label"].format(r=r+1, yr=src_harvest_yrs[r]) for r in range(9)]
            peak_idx = int(np.argmax(curve_vals))
            colors = []
            for i, v in enumerate(rm_pct):
                if i == peak_idx:
                    colors.append("#2ca02c")   # peak – tmavě zelená
                elif v >= 80:
                    colors.append("#74c476")   # dobrý výnos – světle zelená
                else:
                    colors.append("#d62728")   # pokles – červená
            fl = go.Figure(data=[go.Bar(
                x=rl, y=[round(v, 1) for v in rm_pct],
                marker_color=colors,
                text=[f"{v:.0f} %" for v in rm_pct],
                textposition="outside")])
            fl.update_layout(
                title=T["learn_title"],
                xaxis_title=T["learn_xaxis"],
                yaxis_title=T["learn_yaxis"],
                yaxis=dict(range=[0, 115]))
            st.plotly_chart(fl, use_container_width=True, key=f"learn_{plodina}")

        st.subheader(T["sens_header"])
        st.markdown(T["sens_desc"])
        with st.spinner(T["sens_spinner"].format(crop=plodina)):
            xv, yv, zm = calculate_sensitivity_matrix(
                data["type"], data["years"], data["params"], data["y_bounds"],
                "Direct Chip" if data["type"] == "src" else None,
                plocha_ha, n_sim, subsidy_pct/100.0,
                rho=rho_input, drift=drift_rate)
            f3d = go.Figure(data=[go.Surface(
                z=zm, x=xv, y=yv, colorscale="Viridis",
                colorbar=dict(title=T["sens_colorbar"]))])
            f3d.update_layout(
                title=T["sens_title"],
                scene=dict(xaxis_title=T["sens_xaxis"],
                           yaxis_title=T["sens_yaxis"],
                           zaxis_title=T["sens_zaxis"]),
                width=800, height=600, margin=dict(l=65, r=50, b=65, t=90))
            st.plotly_chart(f3d, use_container_width=True, key=f"3d_{plodina}")

        st.subheader(T["price_header"])
        mp   = np.mean(data["prices"],  axis=1)
        p5p  = np.percentile(data["prices"],  5, axis=1)
        p95p = np.percentile(data["prices"], 95, axis=1)
        fpr  = go.Figure()
        fpr.add_trace(go.Scatter(x=yrs, y=mp,   name=T["price_mean"], line=dict(color="purple")))
        fpr.add_trace(go.Scatter(x=yrs, y=p95p, name=T["cf_opt"],     line=dict(width=0), showlegend=False))
        fpr.add_trace(go.Scatter(x=yrs, y=p5p,  name=T["cf_pes"],
                                 fill="tonexty", fillcolor="rgba(128,0,128,0.2)",
                                 line=dict(width=0), showlegend=False))
        fpr.update_layout(title=T["price_title"].format(crop=plodina),
                          xaxis_title=T["price_xaxis"], yaxis_title=T["price_yaxis"])
        st.plotly_chart(fpr, use_container_width=True, key=f"price_{plodina}")

        st.divider()

    # -----------------------------------------------------------------------
    # REKAPITULACE – čistá HTML tabulka
    # -----------------------------------------------------------------------
    if len(summary_data) > 1:
        st.header(T["recap_header"])
        st.markdown(render_recap_table(summary_data, T), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # CITLIVOSTNÍ ANALÝZA – tornado grafy
    # -----------------------------------------------------------------------
    if sa_selected and results:
        st.header(SA["sa_header_recap"])
        param_labels_full  = {k: v[lang_sa] for k, v in SENSITIVITY_PARAMS.items()}
        short_key = f"{lang_sa}_short"
        param_labels_short = {k: v[short_key] for k, v in SENSITIVITY_PARAMS.items()}
        # Snížený počet simulací pro citlivostní analýzu (rychlost)
        sa_n_sim = min(n_sim, 2000)

        for plodina, data in results.items():
            st.markdown(f"### {plodina}")
            all_sa = {}    # param_key → {"pct": [...], "profit": [...], ...}
            sa_bar = st.progress(0)
            for idx, pk in enumerate(sa_selected):
                with st.spinner(SA["sa_running"].format(
                        crop=plodina, param=param_labels_full[pk])):
                    all_sa[pk] = run_param_sensitivity(
                        param_key=pk,
                        base_params=data["params"],
                        y_bounds=data["y_bounds"],
                        crop_type=data["type"],
                        n_sim=sa_n_sim,
                        years=data["years"],
                        subsidy_perc=subsidy_pct / 100.0,
                        area_ha=plocha_ha,
                        discount_rate=discount_pct / 100.0,
                        weather_prob=0.05,
                        src_tech="Direct Chip" if data["type"] == "src" else None,
                        rho=rho_input, drift=drift_rate,
                    )
                sa_bar.progress((idx + 1) / len(sa_selected))

            # 8 spojnicových grafů: Zisk, EAA, StdDev, VaR, CVaR, P(návratnost), Doba návratnosti
            y_labels = {
                "cs": {"profit": "Zisk (€)", "eaa": "EAA (€/rok)",
                       "std": "Směr. odchylka (€)",
                       "var": "VaR 5 % (€)", "cvar": "CVaR 5 % (€)",
                       "payback_prob": "Pravděpodobnost (%)",
                       "payback_yr": "Roky"},
                "en": {"profit": "Profit (€)", "eaa": "EAA (€/yr)",
                       "std": "Std. Dev. (€)",
                       "var": "VaR 5 % (€)", "cvar": "CVaR 5 % (€)",
                       "payback_prob": "Probability (%)",
                       "payback_yr": "Years"},
            }[lang_sa]
            sa_title_eaa = "Citlivost EAA (±10 %)" if lang_sa == "cs" else "EAA Sensitivity (±10 %)"

            tc1, tc2 = st.columns(2)
            with tc1:
                st.plotly_chart(make_sensitivity_line_chart(
                    all_sa, "profit", SA["sa_title_profit"],
                    y_labels["profit"], param_labels_short, lang_sa),
                    use_container_width=True, key=f"sa_profit_{plodina}")
            with tc2:
                st.plotly_chart(make_sensitivity_line_chart(
                    all_sa, "eaa", sa_title_eaa,
                    y_labels["eaa"], param_labels_short, lang_sa),
                    use_container_width=True, key=f"sa_eaa_{plodina}")

            tc3, tc4 = st.columns(2)
            with tc3:
                st.plotly_chart(make_sensitivity_line_chart(
                    all_sa, "std", SA["sa_title_std"],
                    y_labels["std"], param_labels_short, lang_sa),
                    use_container_width=True, key=f"sa_std_{plodina}")
            with tc4:
                st.plotly_chart(make_sensitivity_line_chart(
                    all_sa, "var", SA["sa_title_var"],
                    y_labels["var"], param_labels_short, lang_sa),
                    use_container_width=True, key=f"sa_var_{plodina}")

            tc5, tc6 = st.columns(2)
            with tc5:
                st.plotly_chart(make_sensitivity_line_chart(
                    all_sa, "cvar", SA["sa_title_cvar"],
                    y_labels["cvar"], param_labels_short, lang_sa),
                    use_container_width=True, key=f"sa_cvar_{plodina}")
            with tc6:
                st.plotly_chart(make_sensitivity_line_chart(
                    all_sa, "payback_prob", SA["sa_title_pb_prob"],
                    y_labels["payback_prob"], param_labels_short, lang_sa),
                    use_container_width=True, key=f"sa_pbprob_{plodina}")

            tc7, _ = st.columns(2)
            with tc7:
                st.plotly_chart(make_sensitivity_line_chart(
                    all_sa, "payback_yr", SA["sa_title_pb_yr"],
                    y_labels["payback_yr"], param_labels_short, lang_sa),
                    use_container_width=True, key=f"sa_pbyr_{plodina}")

            st.divider()


# ===========================================================================
# SEKCE 5 – DIVERZIFIKAČNÍ OPTIMALIZÁTOR (čte LIVE hodnoty z formuláře)
# ===========================================================================
st.markdown("---")
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
                       "cvar":   "Maximalizovat CVaR 5 % (minimalizovat ztrátu)"},
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
                       "cvar":   "Maximize CVaR 5 % (minimize loss)"},
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

st.header(DIV_T["header"])
st.markdown(DIV_T["desc"])

# Předpoklad: obě plodiny musí být zaškrtnuté → editable_costs obsahuje obě položky
div_ready = ("Miscanthus" in editable_costs) and ("SRC Vrba" in editable_costs)

if not div_ready:
    st.warning(DIV_T["warn_crops"])
else:
    # LIVE výpočet vstupů přímo z formuláře (každý rerun = aktuální hodnoty)
    try:
        div_misc_y = YIELD_DATA[detected_zone_key][soil_key]["M_giganteus"]
        div_src_y  = YIELD_DATA[detected_zone_key][soil_key]["SRC"]
    except KeyError:
        div_misc_y, div_src_y = [5, 10], [5, 10]

    div_misc_params = editable_costs["Miscanthus"]
    div_src_params  = editable_costs["SRC Vrba"]

    # Info-box: ukáže přesně co se používá – aby uživatel viděl, že vše sedí
    with st.expander(DIV_T["inputs_h"], expanded=False):
        info_lang = st.session_state["lang"]
        labels = {
            "cs": {"zone": "Klimatická zóna", "soil": "Kvalita půdy",
                   "area": "Celková plocha", "yield_m": "Výnosy Miscanthus (t/ha)",
                   "yield_s": "Výnosy SRC (t/ha)", "sub": "Dotace na založení",
                   "disc": "Diskontní sazba", "rho": "Korelace ρ (počasí↔cena)",
                   "drift": "Dekarbonizační drift",
                   "life_m": "Životnost Miscanthus", "life_s": "Životnost SRC",
                   "price_m": "Počáteční cena Miscanthus", "price_s": "Počáteční cena SRC",
                   "yr": "let"},
            "en": {"zone": "Climate zone", "soil": "Soil quality",
                   "area": "Total area", "yield_m": "Miscanthus yield (t/ha)",
                   "yield_s": "SRC yield (t/ha)", "sub": "Establishment subsidy",
                   "disc": "Discount rate", "rho": "Correlation ρ (weather↔price)",
                   "drift": "Decarbonization drift",
                   "life_m": "Miscanthus lifetime", "life_s": "SRC lifetime",
                   "price_m": "Miscanthus starting price", "price_s": "SRC starting price",
                   "yr": "yrs"},
        }[info_lang]
        zone_lbl = T["zones"][detected_zone_key]
        soil_lbl = T["soil_opts"][SOIL_KEYS.index(soil_key)]
        rows = [
            (labels["zone"],   zone_lbl),
            (labels["soil"],   soil_lbl),
            (labels["area"],   f"{plocha_ha:.0f} ha"),
            (labels["yield_m"],f"{div_misc_y[0]}–{div_misc_y[1]}"),
            (labels["yield_s"],f"{div_src_y[0]}–{div_src_y[1]}"),
            (labels["sub"],    f"{subsidy_pct} %"),
            (labels["disc"],   f"{discount_pct:.1f} %"),
            (labels["rho"],    f"{rho_input:+.2f}"),
            (labels["drift"],  f"{drift_pct:+.1f} %/rok" if info_lang=="cs" else f"{drift_pct:+.1f} %/yr"),
            (labels["life_m"], f"{int(div_misc_params['zivotnost'])} {labels['yr']}"),
            (labels["life_s"], f"{int(div_src_params['zivotnost'])} {labels['yr']}"),
            (labels["price_m"],f"{div_misc_params['prodejni_cena_start']:.0f} €/t"),
            (labels["price_s"],f"{div_src_params['prodejni_cena_start']:.0f} €/t"),
        ]
        info_html = "<table style='font-size:13px'>"
        for k, v in rows:
            info_html += f"<tr><td style='padding:3px 16px 3px 0;color:#666'>{k}</td><td style='padding:3px 0;font-weight:600'>{v}</td></tr>"
        info_html += "</table>"
        st.markdown(info_html, unsafe_allow_html=True)

    div_metric_key = st.radio(
        DIV_T["metric"],
        options=DIVERSIFY_METRICS,
        format_func=lambda k: DIV_T["opts"][k],
        horizontal=True, key="div_metric_radio",
    )

    if st.button(DIV_T["btn"], type="primary", key="btn_diversify"):
        with st.spinner(DIV_T["spinner"]):
            div_n_sim = min(n_sim, 2000)
            div_res = run_diversification(
                misc_params=div_misc_params, misc_y_bounds=div_misc_y,
                src_params=div_src_params,   src_y_bounds=div_src_y,
                total_area_ha=plocha_ha, n_sim=div_n_sim,
                subsidy_perc=subsidy_pct / 100.0, discount_rate=discount_pct / 100.0,
                rho=rho_input, drift=drift_rate,
            )

        opt_idx, opt_val = find_optimum(div_res, div_metric_key)
        opt_pct_m  = div_res["pct_misc"][opt_idx]
        opt_pct_s  = 100 - opt_pct_m
        pure_m_val = div_res[div_metric_key][-1]   # 100 % M
        pure_s_val = div_res[div_metric_key][0]    # 0 % M = 100 % SRC

        # KPI
        dk1, dk2, dk3, dk4 = st.columns(4)
        dk1.metric(DIV_T["opt_kpi"], f"{opt_pct_m} % M / {opt_pct_s} % SRC")
        dk2.metric(DIV_T["metric_val"], f"{opt_val:,.0f} €")
        dk3.metric(DIV_T["vs_pure_m"],
                   f"{(opt_val - pure_m_val):+,.0f} €",
                   delta_color="normal")
        dk4.metric(DIV_T["vs_pure_s"],
                   f"{(opt_val - pure_s_val):+,.0f} €",
                   delta_color="normal")

        # Vizualizace pole – stacked horizontal bar
        st.subheader(DIV_T["field"])
        area_M = plocha_ha * opt_pct_m / 100.0
        area_S = plocha_ha * opt_pct_s / 100.0
        fig_field = go.Figure()
        fig_field.add_trace(go.Bar(
            y=["🌾"], x=[area_M], orientation="h",
            name="Miscanthus", marker=dict(color="#7CB342", line=dict(color="#33691E", width=2)),
            text=f"<b>{opt_pct_m} %</b><br>{area_M:.1f} ha", textposition="inside",
            textfont=dict(color="white", size=16),
        ))
        fig_field.add_trace(go.Bar(
            y=["🌾"], x=[area_S], orientation="h",
            name="SRC vrba", marker=dict(color="#8D6E63", line=dict(color="#3E2723", width=2)),
            text=f"<b>{opt_pct_s} %</b><br>{area_S:.1f} ha", textposition="inside",
            textfont=dict(color="white", size=16),
        ))
        fig_field.update_layout(
            barmode="stack", height=180,
            xaxis=dict(title="Plocha (ha)", range=[0, plocha_ha]),
            yaxis=dict(showticklabels=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=40, b=40, l=10, r=10),
        )
        st.plotly_chart(fig_field, use_container_width=True, key="div_field")

        # Frontier – křivka napříč všemi poměry
        st.subheader(DIV_T["frontier"])
        fig_front = go.Figure()
        fig_front.add_trace(go.Scatter(
            x=div_res["pct_misc"], y=div_res[div_metric_key],
            mode="lines+markers", name=DIV_T["opts"][div_metric_key],
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=6),
        ))
        # Zvýrazni optimum
        fig_front.add_trace(go.Scatter(
            x=[opt_pct_m], y=[opt_val],
            mode="markers", name="Optimum",
            marker=dict(size=18, color="#d62728", symbol="star",
                        line=dict(color="white", width=2)),
        ))
        fig_front.update_layout(
            xaxis_title=DIV_T["x_axis"],
            yaxis_title=DIV_T["opts"][div_metric_key],
            height=420,
            xaxis=dict(dtick=10, range=[-2, 102]),
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
            margin=dict(t=30, b=80),
        )
        st.plotly_chart(fig_front, use_container_width=True, key="div_frontier")