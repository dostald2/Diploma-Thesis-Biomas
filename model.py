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
SOIL_KEYS = ["Velmi úrodná", "Úrodná", "Neúrodná"]

# ===========================================================================
# DATA
# ===========================================================================
DEFAULT_COSTS = {
    "Miscanthus": {
        "zalozeni": 2575, "sadba_podil": 2075, "udrzba": 189,
        "sklizen_per_tuna": 25, "prodejni_cena_start": 80,
        "riziko_fail": 0.20, "zivotnost": 20,
    },
    "SRC Vrba": {
        "zalozeni": 2600, "sadba_podil": 1800, "udrzba": 475,
        "sklizen_per_tuna": 36, "prodejni_cena_start": 75,
        "riziko_fail": 0.20, "zivotnost": 24,
    },
}

YIELD_DATA = {
    "Tropické a Subtropické": {
        "Velmi úrodná": {"M_giganteus": [28, 46], "SRC": [12, 20]},
        "Úrodná":       {"M_giganteus": [24, 38], "SRC": [11, 17]},
        "Neúrodná":     {"M_giganteus": [18, 30], "SRC": [8,  14]},
    },
    "Jižní Evropa / Středomoří": {
        "Velmi úrodná": {"M_giganteus": [15, 24], "SRC": [10, 11]},
        "Úrodná":       {"M_giganteus": [12, 18], "SRC": [8,  10]},
        "Neúrodná":     {"M_giganteus": [5,  10], "SRC": [5,   9]},
    },
    "Střední Evropa / Mírné pásmo": {
        "Velmi úrodná": {"M_giganteus": [10, 18], "SRC": [8, 12]},
        "Úrodná":       {"M_giganteus": [9,  14], "SRC": [7, 10]},
        "Neúrodná":     {"M_giganteus": [4,   7], "SRC": [5,  9]},
    },
    "Severní Evropa / Chladné pásmo": {
        "Velmi úrodná": {"M_giganteus": [5, 10], "SRC": [7, 10]},
        "Úrodná":       {"M_giganteus": [4,  8], "SRC": [6,  9]},
        "Neúrodná":     {"M_giganteus": [3,  6], "SRC": [4,  7]},
    },
    "Suché / Marginální půdy": {
        "Velmi úrodná": {"M_giganteus": [4, 10], "SRC": [5, 9]},
        "Úrodná":       {"M_giganteus": [3,  8], "SRC": [4, 8]},
        "Neúrodná":     {"M_giganteus": [2,  5], "SRC": [3, 6]},
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


def gompertz_growth(t_arr, y_max=1.0, b=3.5, c=0.55):
    growth  = np.exp(-b * np.exp(-c * t_arr))
    decline = np.where(t_arr > 12, 1.0 - 0.025*(t_arr - 12), 1.0)
    return growth * np.clip(decline, 0.60, 1.0)


def generate_correlated_shocks(years, n_sim, rho=-0.35):
    L     = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z_cor = L @ np.random.standard_normal((2, years * n_sim))
    u     = norm.cdf(z_cor)
    return u[0].reshape(years, n_sim), u[1].reshape(years, n_sim)


def src_rotation_multiplier(rotation_index):
    learning = [1.00, 1.08, 1.12, 1.12, 1.12]
    if rotation_index < len(learning):
        return learning[rotation_index]
    return max(learning[-1] * (0.96 ** (rotation_index - len(learning) + 1)), 0.75)


def simulate_miscanthus(n_sim, years, params, yield_bounds, subsidy_perc, weather_prob=0.2):
    y_max_sim      = np.clip(np.random.normal(np.mean(yield_bounds),
                             (yield_bounds[1]-yield_bounds[0])/4, n_sim), 0, 60)
    gompertz_curve = gompertz_growth(np.arange(1, years+1, dtype=float))
    wu, pu         = generate_correlated_shocks(years, n_sim)
    wm             = np.clip(norm.ppf(wu, loc=1.0, scale=0.22), 0.3, 1.6)
    wm[np.random.rand(years, n_sim) < weather_prob] *= 0.70
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
    fail  = np.random.binomial(1, params["riziko_fail"], n_sim)
    cf[0, :] -= (capex + fail*0.5*capex) * (1 - subsidy_perc)
    return cf, yields, prices


def simulate_src(n_sim, years, params, yield_bounds, tech_type, subsidy_perc, weather_prob=0.2):
    base = np.maximum(np.random.normal(np.mean(yield_bounds),
                      (yield_bounds[1]-yield_bounds[0])/4, (years, n_sim)), 0)
    wu, pu = generate_correlated_shocks(years, n_sim)
    wm = np.clip(norm.ppf(wu, loc=1.0, scale=0.22), 0.3, 1.6)
    wm[np.random.rand(years, n_sim) < weather_prob] *= 0.70
    growth    = base * wm
    harvested = np.zeros((years, n_sim))
    accum, rot = np.zeros(n_sim), 0
    for t in range(years):
        accum += growth[t, :]
        if (t+1) % 3 == 0:
            harvested[t, :] = accum * src_rotation_multiplier(rot)
            accum = np.zeros(n_sim); rot += 1
    psc    = norm.ppf(pu)
    prices = np.zeros((years, n_sim))
    prices[0, :] = params["prodejni_cena_start"]
    for t in range(1, years):
        shock = 0.70*psc[t, :] + 0.30*np.random.standard_normal(n_sim)
        prices[t, :] = prices[t-1, :] * np.exp(0.20*shock)
    prices = np.clip(prices, 35, 150)
    cf = np.zeros((years, n_sim)) - params["udrzba"]
    for t in range(years):
        if (t+1) % 3 == 0:
            v = harvested[t, :]
            cf[t, :] += v*prices[t, :] - v*params["sklizen_per_tuna"]
    capex = np.random.normal(params["zalozeni"], 150, n_sim)
    fail  = np.random.binomial(1, params["riziko_fail"], n_sim)
    cf[0, :] -= (capex + fail*0.5*capex) * (1 - subsidy_perc)
    return cf, harvested, prices


def calculate_sensitivity_matrix(crop_type, years, base_params, y_bounds,
                                  src_tech, area_ha, n_sim, subsidy_perc):
    steps = 10
    xr, yr = np.linspace(0, 0.4, steps), np.linspace(0, 0.4, steps)
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
# GLOBÁLNÍ CSS
# ===========================================================================
GLOBAL_CSS = """
<style>
.block-container { padding-top: 0.8rem !important; }
</style>
"""

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

col_ttl, _, col_flag = st.columns([7, 1, 2])

with col_ttl:
    st.title(T["app_title"])
    st.markdown(T["app_subtitle"])

with col_flag:
    # Vycentruj vlajky doprava a dolů k nadpisu
    st.markdown("<div style='padding-top:28px;display:flex;justify-content:flex-end;gap:8px;'>",
                unsafe_allow_html=True)
    fc1, fc2 = st.columns(2)
    with fc1:
        cs_active = st.session_state["lang"] == "cs"
        if st.button("🇨🇿", key="btn_cs",
                     help="Čeština",
                     type="primary" if cs_active else "secondary"):
            st.session_state["lang"] = "cs"
            st.rerun()
    with fc2:
        en_active = st.session_state["lang"] == "en"
        if st.button("🇬🇧", key="btn_en",
                     help="English",
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

fmap = folium.Map(location=[49.8, 15.5], zoom_start=4)
fmap.add_child(folium.LatLngPopup())
map_data = st_folium(fmap, height=600, use_container_width=True)

detected_zone_key = "Střední Evropa / Mírné pásmo"
real_rain, real_temp = 600.0, 10.0

if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.success(f"{T['coord_success']} {lat:.4f}, {lon:.4f}")
    with st.spinner(T["spinner_climate"]):
        tv, rv = get_climate_data(lat, lon)
        if tv is not None:
            real_temp, real_rain = tv, rv
            detected_zone_key = determine_zone(lat, real_temp, real_rain)

detected_zone_label = T["zones"][detected_zone_key]

ci1, ci2, ci3 = st.columns(3)
ci1.metric(T["metric_temp"], f"{real_temp:.1f} °C")
ci1.caption(T["metric_temp_cap"])
ci2.metric(T["metric_rain"], f"{real_rain:.0f} mm")
ci2.caption(T["metric_rain_cap"])
ci3.info(f"{T['climate_zone']}: **{detected_zone_label}**")

st.markdown("---")

# ===========================================================================
# SEKCE 2 – KONFIGURACE
# ===========================================================================
st.header(T["sec2_header"])

cp1, cp2 = st.columns(2)
with cp1:
    soil_idx  = st.selectbox(T["soil_quality"], range(3),
                              format_func=lambda i: T["soil_opts"][i])
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
            rm = [src_rotation_multiplier(r) for r in range(8)]
            rl = [T["learn_rot_label"].format(r=r+1, yr=(r+1)*3) for r in range(8)]
            fl = go.Figure(data=[go.Bar(
                x=rl, y=[m*100 for m in rm],
                marker_color=["#2ca02c" if m >= 1.0 else "#d62728" for m in rm])])
            fl.update_layout(title=T["learn_title"], xaxis_title=T["learn_xaxis"],
                             yaxis_title=T["learn_yaxis"], yaxis=dict(range=[70, 120]))
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