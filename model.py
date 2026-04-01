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
        "zalozeni": 2600, "sadba_podil": 1800, "udrzba": 475,
        "sklizen_per_tuna": 36, "prodejni_cena_start": 75,
        "riziko_fail": 0.05, "zivotnost": 30,
    },
}

# Výnosové rozsahy [min, max] t/ha sušiny dle vědců (2025)
# Kategorie půdy: Optimální / Průměrná / Neúrodná / Nevhodná
# Nevhodná: 0–4 t/ha (obě plodiny), Neúrodná: 4–7, Průměrná: Misc 7–14 / SRC 7–12, Optimální: Misc 14–16 / SRC 12–15
YIELD_DATA = {
    "Tropické a Subtropické": {
        "Optimální": {"M_giganteus": [14, 16], "SRC": [12, 15]},
        "Průměrná":  {"M_giganteus": [7,  14], "SRC": [7,  12]},
        "Neúrodná":  {"M_giganteus": [4,   7], "SRC": [4,   7]},
        "Nevhodná":  {"M_giganteus": [0,   4], "SRC": [0,   4]},
    },
    "Jižní Evropa / Středomoří": {
        "Optimální": {"M_giganteus": [14, 16], "SRC": [12, 15]},
        "Průměrná":  {"M_giganteus": [7,  14], "SRC": [7,  12]},
        "Neúrodná":  {"M_giganteus": [4,   7], "SRC": [4,   7]},
        "Nevhodná":  {"M_giganteus": [0,   4], "SRC": [0,   4]},
    },
    "Střední Evropa / Mírné pásmo": {
        "Optimální": {"M_giganteus": [14, 16], "SRC": [12, 15]},
        "Průměrná":  {"M_giganteus": [7,  14], "SRC": [7,  12]},
        "Neúrodná":  {"M_giganteus": [4,   7], "SRC": [4,   7]},
        "Nevhodná":  {"M_giganteus": [0,   4], "SRC": [0,   4]},
    },
    "Severní Evropa / Chladné pásmo": {
        "Optimální": {"M_giganteus": [14, 16], "SRC": [12, 15]},
        "Průměrná":  {"M_giganteus": [7,  14], "SRC": [7,  12]},
        "Neúrodná":  {"M_giganteus": [4,   7], "SRC": [4,   7]},
        "Nevhodná":  {"M_giganteus": [0,   4], "SRC": [0,   4]},
    },
    "Suché / Marginální půdy": {
        "Optimální": {"M_giganteus": [14, 16], "SRC": [12, 15]},
        "Průměrná":  {"M_giganteus": [7,  14], "SRC": [7,  12]},
        "Neúrodná":  {"M_giganteus": [4,   7], "SRC": [4,   7]},
        "Nevhodná":  {"M_giganteus": [0,   4], "SRC": [0,   4]},
    },
}

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
    #   rok 1 nízký, rok 3–4 výrazný nástup, rok 5–6 plný výnos, plateau do roku 18
    growth  = np.exp(-b * np.exp(-c * t_arr))
    # Pokles po roku 18 pro Miscanthus (dle vědců 2025)
    decline = np.where(t_arr > 18, 1.0 - 0.025*(t_arr - 18), 1.0)
    return growth * np.clip(decline, 0.60, 1.0)


def src_yield_curve(t_arr):
    """
    Relativní roční produkce suché hmoty SRC vrby v průběhu životnosti plantáže.
    Kalibrováno vizuálně dle Janota et al. (Energy Reports 2023, Fig. 7) – česká polní data.

    Tvar: bell curve (zvonek) – nástup do roku 11, pak exponenciální pokles.
      - Gompertz nástup: b=3.0, c=0.40 → rok 5 ~69 %, rok 8 ~92 %, rok 11 = peak
      - Exponenciální pokles od roku 11: rate=0.06/rok
      - Výsledek: rok 16 ~77 %, rok 22 ~54 %, rok 29 ~35 %
    """
    t = np.asarray(t_arr, dtype=float)
    growth  = np.exp(-3.0 * np.exp(-0.40 * t))
    decline = np.exp(-0.06 * np.maximum(0.0, t - 11))
    return growth * decline


def generate_correlated_shocks(years, n_sim, rho=-0.35):
    L     = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z_cor = L @ np.random.standard_normal((2, years * n_sim))
    u     = norm.cdf(z_cor)
    return u[0].reshape(years, n_sim), u[1].reshape(years, n_sim)


def src_rotation_multiplier(rotation_index):
    # Dle vědců 2025: pokles až po 20 letech = rotace 7 (rok 21)
    # Rotace 0=rok3, 1=rok6, ..., 6=rok21
    # Plateau 1.12 udrženo do rotace 6, pak degenerace
    learning = [1.00, 1.08, 1.12, 1.12, 1.12, 1.12, 1.12]  # rotace 0–6 (roky 3–21)
    if rotation_index < len(learning):
        return learning[rotation_index]
    # Degenerace od rotace 7 (rok 24+): −4 % za rotaci
    degeneration = rotation_index - (len(learning) - 1)
    return max(learning[-1] * (0.96 ** degeneration), 0.75)


def simulate_miscanthus(n_sim, years, params, yield_bounds, subsidy_perc, weather_prob=0.05):
    y_max_sim      = np.clip(np.random.normal(np.mean(yield_bounds),
                             (yield_bounds[1]-yield_bounds[0])/4, n_sim), 0, 60)
    gompertz_curve = gompertz_growth(np.arange(1, years+1, dtype=float))
    wu, pu         = generate_correlated_shocks(years, n_sim)
    # scale kalibrován na meziroční rozdíl ~4 t/ha dle vědců (2025)
    # Na průměru ~11.5 t/ha → relativní scale = 4 / (2 * 11.5) ≈ 0.17
    wm             = np.clip(norm.ppf(wu, loc=1.0, scale=0.17), 0.5, 1.5)
    # Stresový rok: 1 z 20 (5 %), výnos klesne na 50 % (×0.50) dle vědců
    wm[np.random.rand(years, n_sim) < weather_prob] *= 0.50
    yields         = gompertz_curve[:, np.newaxis] * y_max_sim[np.newaxis, :] * wm
    psc            = norm.ppf(pu)
    prices         = np.zeros((years, n_sim))
    prices[0, :]   = params["prodejni_cena_start"]
    for t in range(1, years):
        shock = 0.70*psc[t, :] + 0.30*np.random.standard_normal(n_sim)
        prices[t, :] = prices[t-1, :] * np.exp(0.20*shock)
    prices = np.clip(prices, 45, 160)
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


def simulate_src(n_sim, years, params, yield_bounds, tech_type, subsidy_perc, weather_prob=0.05):
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
    wu, pu = generate_correlated_shocks(years, n_sim)
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
    psc    = norm.ppf(pu)
    prices = np.zeros((years, n_sim))
    prices[0, :] = params["prodejni_cena_start"]
    for t in range(1, years):
        shock = 0.70*psc[t, :] + 0.30*np.random.standard_normal(n_sim)
        prices[t, :] = prices[t-1, :] * np.exp(0.20*shock)
    prices = np.clip(prices, 35, 150)
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
                                  src_tech, area_ha, n_sim, subsidy_perc):
    steps = 10
    xr, yr = np.linspace(0, 0.2, steps), np.linspace(0, 0.2, steps)
    zm = []
    for wp in yr:
        row = []
        for fp in xr:
            p = {**base_params, "riziko_fail": fp}
            if crop_type == "misc":
                cf, _, _ = simulate_miscanthus(n_sim, years, p, y_bounds, subsidy_perc, wp)
            else:
                cf, _, _ = simulate_src(n_sim, years, p, y_bounds, src_tech, subsidy_perc, wp)
            row.append(np.mean(np.std(cf * area_ha, axis=0)))
        zm.append(row)
    return xr, yr, np.array(zm)


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
        "Průměrný Zisk", "Potenciál (95%)", "Doba návratnosti (roky)",
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
subsidy_pct = st.slider(T["subsidy_label"], 0, 100, 0, 5)

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
                n_sim, int(params["zivotnost"]), params, y_bounds, subsidy_pct/100.0)
            results[plodina] = {"cf": cf, "yields": yields, "prices": prices,
                                "years": int(params["zivotnost"]),
                                "params": params, "y_bounds": y_bounds, "type": "misc"}
        elif plodina == "SRC Vrba":
            params = editable_costs["SRC Vrba"]
            cf, yields, prices = simulate_src(
                n_sim, int(params["zivotnost"]), params, y_bounds, "Direct Chip", subsidy_pct/100.0)
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

        total_profits  = np.sum(cf_total, axis=0)
        var_5          = np.percentile(total_profits, 5)
        cvar_5         = total_profits[total_profits <= var_5].mean()
        avg_yearly_std = np.mean(np.std(cf_total, axis=0))
        cum_cf         = np.cumsum(cf_total, axis=0)
        pb_years       = np.argmax(cum_cf > 0, axis=0)
        succ_pb        = pb_years[pb_years > 0]
        payback_prob   = len(succ_pb)/n_sim if n_sim > 0 else 0
        avg_payback    = float(np.mean(succ_pb)+1) if len(succ_pb) > 0 else None

        summary_data.append({
            "Plodina":                     plodina,
            "Průměrný Zisk":               np.mean(total_profits),
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
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(T["kpi_avg_profit"], f"{np.mean(total_profits):,.0f} €")
        k2.metric(T["kpi_potential"],  f"{np.percentile(total_profits, 95):,.0f} €")
        k3.metric(T["kpi_payback"],
                  f"{avg_payback:.2f} {T['kpi_payback_unit']}" if avg_payback else "N/A")
        k4.metric(T["kpi_payback_prob"],
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
                plocha_ha, n_sim, subsidy_pct/100.0)
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