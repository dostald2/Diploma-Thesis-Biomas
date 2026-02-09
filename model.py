import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium
import requests

# --- 1. DATA A KONFIGURACE ---

DEFAULT_COSTS = {
    "Miscanthus": {
        "zalozeni": 2575,
        "sadba_podil": 2075,
        "udrzba": 189,
        "sklizen_per_tuna": 25,
        "prodejni_cena_start": 80,
        "riziko_fail": 0.20,
        "zivotnost": 20
    },
    "SRC Vrba": {
        "zalozeni": 2600,
        "sadba_podil": 1800,
        "udrzba": 475,
        "sklizen_per_tuna": 36,
        "prodejni_cena_start": 60,
        "riziko_fail": 0.20,
        "zivotnost": 24
    }
}

COST_TRANSLATIONS = {
    "zalozeni": "Založení",
    "sadba_podil": "Podíl sadby",
    "udrzba": "Údržba",
    "sklizen_per_tuna": "Sklizeň (€/tuna)",
    "prodejni_cena_start": "Počáteční cena (€/tuna)",
    "riziko_fail": "Riziko neúspěchu (%)",
    "zivotnost": "Životnost (roky)",
}

YIELD_DATA = {
    "Tropické a Subtropické": {
        "Velmi úrodná": {"M_giganteus": [24, 46], "M_sinensis": [20, 40], "SRC": [12, 20]},
        "Úrodná": {"M_giganteus": [22, 42], "M_sinensis": [18, 35], "SRC": [11, 19]},
        "Neúrodná": {"M_giganteus": [18, 35], "M_sinensis": [15, 30], "SRC": [10, 15]}
    },
    "Jižní Evropa / Středomoří": {
        "Velmi úrodná": {"M_giganteus": [15, 21], "M_sinensis": [10, 14], "SRC": [10, 11]},
        "Úrodná": {"M_giganteus": [12, 18], "M_sinensis": [9, 12], "SRC": [8, 10]},
        "Neúrodná": {"M_giganteus": [5, 10], "M_sinensis": [7, 9], "SRC": [5, 9]}
    },
    "Střední Evropa / Mírné pásmo": {
        "Velmi úrodná": {"M_giganteus": [10, 15], "M_sinensis": [8, 11], "SRC": [8, 12]},
        "Úrodná": {"M_giganteus": [9, 13], "M_sinensis": [7, 10], "SRC": [7, 10]},
        "Neúrodná": {"M_giganteus": [4, 7], "M_sinensis": [6, 8], "SRC": [5, 9]}
    },
    "Severní Evropa / Chladné pásmo": {
        "Velmi úrodná": {"M_giganteus": [9, 11], "M_sinensis": [9, 11], "SRC": [7, 10]},
        "Úrodná": {"M_giganteus": [7, 9], "M_sinensis": [7, 9], "SRC": [6, 9]},
        "Neúrodná": {"M_giganteus": [5, 7], "M_sinensis": [5, 7], "SRC": [4, 7]}
    },
    "Suché / Marginální půdy": {
        "Velmi úrodná": {"M_giganteus": [4, 7], "M_sinensis": [6, 8], "SRC": [5, 9]},
        "Úrodná": {"M_giganteus": [3, 6], "M_sinensis": [5, 7], "SRC": [4, 8]},
        "Neúrodná": {"M_giganteus": [2, 5], "M_sinensis": [4, 6], "SRC": [3, 6]}
    }
}


# --- 2. POMOCNÉ FUNKCE (API & SIMULACE) ---

@st.cache_data
def get_climate_data(lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2014-01-01",
        "end_date": "2023-12-31",
        "daily": ["temperature_2m_mean", "precipitation_sum"],
        "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        daily_temps = [x for x in data['daily']['temperature_2m_mean'] if x is not None]
        daily_rain = [x for x in data['daily']['precipitation_sum'] if x is not None]

        avg_temp = sum(daily_temps) / len(daily_temps)
        avg_annual_precip = (sum(daily_rain) / len(daily_rain)) * 365

        return avg_temp, avg_annual_precip
    except:
        return None, None


def determine_zone(lat, avg_temp, avg_rain):
    if avg_rain < 450:
        return "Suché / Marginální půdy"
    if abs(lat) < 23.5 or avg_temp > 22:
        return "Tropické a Subtropické"
    if avg_temp > 13:
        return "Jižní Evropa / Středomoří"
    elif avg_temp > 8:
        return "Střední Evropa / Mírné pásmo"
    else:
        return "Severní Evropa / Chladné pásmo"


def sigmoid_growth(t, y_max, k=0.8, t0=2.5):
    return y_max / (1 + np.exp(-k * (t - t0)))


# --- UPRAVENÉ SIMULAČNÍ FUNKCE (přidán parametr weather_prob) ---

def simulate_miscanthus(n_sim, years, params, yield_bounds, subsidy_perc, weather_prob=0.2):
    y_mean = np.mean(yield_bounds)
    y_std = (yield_bounds[1] - yield_bounds[0]) / 4
    y_max_sim = np.clip(np.random.normal(y_mean, y_std, n_sim), 0, 50)

    yields = np.zeros((years, n_sim))
    weather = np.random.normal(1, 0.25, (years, n_sim))

    drought_mask = np.random.rand(years, n_sim) < weather_prob
    weather[drought_mask] *= 0.7

    for t in range(1, years + 1):
        yields[t - 1, :] = y_max_sim * sigmoid_growth(t, 1) * weather[t - 1, :]

    prices = np.zeros((years, n_sim))
    prices[0, :] = params['prodejni_cena_start']
    for t in range(1, years):
        shock = np.random.normal(0, 1, n_sim)
        prices[t, :] = prices[t - 1, :] * np.exp((0.02 - 0.5 * 0.2 ** 2) + 0.2 * shock)
    prices = np.clip(prices, 48, 134)

    revenue = yields * prices
    harvest_costs = yields * params['sklizen_per_tuna']
    cash_flow = revenue - params['udrzba'] - harvest_costs

    failures = np.random.binomial(1, params['riziko_fail'], n_sim)
    initial_capex = np.random.normal(params['zalozeni'], 200, n_sim)
    replant_cost = failures * (0.5 * initial_capex)
    
    # Aplikace dotace
    total_initial_cost = initial_capex + replant_cost
    cash_flow[0, :] -= total_initial_cost * (1 - subsidy_perc)


    return cash_flow, yields, prices


def simulate_src(n_sim, years, params, yield_bounds, tech_type, subsidy_perc, weather_prob=0.2):
    y_mean = np.mean(yield_bounds)
    y_std = (yield_bounds[1] - yield_bounds[0]) / 4
    yearly_growth = np.maximum(np.random.normal(y_mean, 2.5, (years, n_sim)), 0)

    weather = np.random.normal(1, 0.25, (years, n_sim))
    drought_mask = np.random.rand(years, n_sim) < weather_prob
    weather[drought_mask] *= 0.7
    yearly_growth *= weather

    harvested_biomass = np.zeros((years, n_sim))
    accumulated = np.zeros(n_sim)

    for t in range(years):
        accumulated += yearly_growth[t, :]
        if (t + 1) % 3 == 0:
            harvested_biomass[t, :] = accumulated
            accumulated = np.zeros(n_sim)

    prices = np.zeros((years, n_sim))
    prices[0, :] = params['prodejni_cena_start']
    for t in range(1, years):
        shock = np.random.normal(0, 1, n_sim)
        prices[t, :] = prices[t - 1, :] * np.exp((0.02 - 0.5 * 0.2 ** 2) + 0.2 * shock)
    prices = np.clip(prices, 40, 120)

    cash_flow = np.zeros((years, n_sim)) - params['udrzba']

    for t in range(years):
        if (t + 1) % 3 == 0:
            vol = harvested_biomass[t, :]
            rev = vol * prices[t, :]
            harvest_cost = vol * params['sklizen_per_tuna']
            cash_flow[t, :] += rev - harvest_cost

    failures = np.random.binomial(1, params['riziko_fail'], n_sim)
    initial_capex = np.random.normal(params['zalozeni'], 150, n_sim)
    replant_cost = failures * (0.5 * initial_capex)

    # Aplikace dotace
    total_initial_cost = initial_capex + replant_cost
    cash_flow[0, :] -= total_initial_cost * (1 - subsidy_perc)

    return cash_flow, harvested_biomass, prices


# --- UPRAVENÁ FUNKCE PRO VÝPOČET MATICE SENSITIVITY ---
def calculate_sensitivity_matrix(plodina_type, years, base_params, y_bounds, src_tech, area_ha, n_sim, subsidy_perc):
    steps = 10
    x_range = np.linspace(0.0, 0.4, steps)
    y_range = np.linspace(0.0, 0.4, steps)
    z_matrix = []

    for weather_p in y_range:
        row = []
        for fail_p in x_range:
            temp_params = base_params.copy()
            temp_params['riziko_fail'] = fail_p

            if plodina_type == "misc":
                cf, _, _ = simulate_miscanthus(n_sim, years, temp_params, y_bounds, subsidy_perc, weather_prob=weather_p)
            else:
                cf, _, _ = simulate_src(n_sim, years, temp_params, y_bounds, src_tech, subsidy_perc, weather_prob=weather_p)

            yearly_stds = np.std(cf * area_ha, axis=0)
            avg_yearly_std = np.mean(yearly_stds)
            row.append(avg_yearly_std)
        z_matrix.append(row)

    return x_range, y_range, np.array(z_matrix)


# --- 3. UI STRUKTURA ---

st.set_page_config(layout="wide", page_title="BioFarm Simulator")

st.title("🌍 BioFarm Simulator: Monte Carlo")
st.markdown("Interaktivní nástroj pro predikci výnosů biomasy na základě lokality a klimatických dat.")
st.markdown("---")

# --- SEKCE 1: MAPA ---
st.header("1. Výběr Lokality")
st.markdown("Klikněte do mapy pro automatické určení klimatického pásu a srážek.")

m = folium.Map(location=[49.8, 15.5], zoom_start=4)
m.add_child(folium.LatLngPopup())
map_data = st_folium(m, height=600, use_container_width=True)

detected_zone = "Střední Evropa / Mírné pásmo"
real_rain = 600
real_temp = 10

if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.success(f"📍 Vybrána souřadnice: {lat:.4f}, {lon:.4f}")

    with st.spinner('Stahuji data o klimatu...'):
        t, r = get_climate_data(lat, lon)
        if t is not None:
            real_temp, real_rain = t, r
            detected_zone = determine_zone(lat, real_temp, real_rain)

col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("Detekovaná Teplota", f"{real_temp:.1f} °C")
col_info1.caption("Průměr za 10 let")
col_info2.metric("Roční srážky", f"{real_rain:.0f} mm")
col_info2.caption("Rozhoduje o suchu")
col_info3.info(f"Klimatický pás: **{detected_zone}**")

st.markdown("---")

# --- SEKCE 2: PARAMETRY PŮDY A PLODIN ---
st.header("2. Konfigurace Farmy")

col_params1, col_params2 = st.columns(2)

with col_params1:
    lokalita = detected_zone
    puda = st.selectbox("Kvalita půdy:", ["Velmi úrodná", "Úrodná", "Neúrodná"])
    plocha_ha = st.number_input("Plocha (ha):", min_value=1.0, value=10.0, step=1.0)

with col_params2:
    st.write("Vyber plodiny pro simulaci:")
    m_giganteus = st.checkbox("M_giganteus", value=True)
    m_sinensis = st.checkbox("M_sinensis")
    src_vrba = st.checkbox("SRC Vrba")
    plodiny = []
    if m_giganteus:
        plodiny.append("M_giganteus")
    if m_sinensis:
        plodiny.append("M_sinensis")
    if src_vrba:
        plodiny.append("SRC Vrba")

n_sim = st.number_input("Počet simulací (Monte Carlo):", min_value=1000, max_value=50000, value=5000, step=1000)
subsidy_percentage = st.slider("Dotace na založení (%)", 0, 100, 0, 5)

st.markdown("---")

# --- SEKCE 3: NÁKLADY ---
st.header("3. Detailní Náklady (€/ha)")
st.markdown("<p style='text-align: center;'>Hodnoty jsou předvyplněné dle standardů, ale můžete je upravit.</p>",
            unsafe_allow_html=True)

st.markdown("""
<style>
    .stDataFrame div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] div[data-testid="stDataFrameContainer"] div[role="grid"] div[role="row"] div[role="gridcell"] {
        text-align: center !important;
        display: flex;
        justify-content: center;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

editable_costs = {}
show_miscanthus = any(p in ["M_giganteus", "M_sinensis"] for p in plodiny)
show_src = "SRC Vrba" in plodiny

if show_miscanthus and show_src:
    col_cost1, col_cost2 = st.columns(2)
else:
    col_cost1 = st.container()
    col_cost2 = st.container()

if show_miscanthus:
    with col_cost1:
        st.markdown("<h3 style='text-align: center;'>Miscanthus</h3>", unsafe_allow_html=True)
        df_misc = pd.DataFrame.from_dict(DEFAULT_COSTS["Miscanthus"], orient='index', columns=["Hodnota (€)"])
        df_misc.index = df_misc.index.map(COST_TRANSLATIONS)
        edited_misc = st.data_editor(df_misc, use_container_width=True, key="misc_edit")
        original_keys = {v: k for k, v in COST_TRANSLATIONS.items()}
        edited_misc.index = edited_misc.index.map(original_keys)
        editable_costs["Miscanthus"] = edited_misc["Hodnota (€)"].to_dict()

if show_src:
    with col_cost2:
        st.markdown("<h3 style='text-align: center;'>SRC Vrba</h3>", unsafe_allow_html=True)
        df_src = pd.DataFrame.from_dict(DEFAULT_COSTS["SRC Vrba"], orient='index', columns=["Hodnota (€)"])
        df_src.index = df_src.index.map(COST_TRANSLATIONS)
        edited_src = st.data_editor(df_src, use_container_width=True, key="src_edit")
        original_keys = {v: k for k, v in COST_TRANSLATIONS.items()}
        edited_src.index = edited_src.index.map(original_keys)
        editable_costs["SRC Vrba"] = edited_src["Hodnota (€)"].to_dict()

st.markdown("---")

# --- SEKCE 4: SIMULACE ---
if st.button("🚀 Spustit Simulaci", type="primary", use_container_width=True):

    st.write("### 📊 Výsledky Simulace")
    results = {}
    summary_data = []

    my_bar = st.progress(0)

    for i, plodina in enumerate(plodiny):
        try:
            yield_key = "SRC" if plodina == "SRC Vrba" else "M_giganteus" if "giganteus" in plodina else "M_sinensis"
            y_bounds = YIELD_DATA[lokalita][puda][yield_key]
        except KeyError:
            y_bounds = [5, 10]
            st.warning(f"Pozor: Data pro {plodina} v pásu {lokalita} chybí, použita výchozí [5, 10].")

        if plodina in ["M_giganteus", "M_sinensis"]:
            params = editable_costs["Miscanthus"]
            cf, yields, prices = simulate_miscanthus(n_sim, int(params['zivotnost']), params, y_bounds, subsidy_percentage / 100.0)
            results[plodina] = {"cf": cf, "yields": yields, "years": int(params['zivotnost']), "params": params, "y_bounds": y_bounds,
                                "type": "misc", "prices": prices}

        elif plodina == "SRC Vrba":
            params = editable_costs["SRC Vrba"]
            cf, yields, prices = simulate_src(n_sim, int(params['zivotnost']), params, y_bounds, "Direct Chip", subsidy_percentage / 100.0)
            results[plodina] = {"cf": cf, "yields": yields, "years": int(params['zivotnost']), "params": params, "y_bounds": y_bounds,
                                "type": "src", "prices": prices}

        my_bar.progress((i + 1) / len(plodiny))

    # Vykreslení grafů
    for plodina, data in results.items():
        st.markdown(f"## 🌿 {plodina}")

        cf_total = data["cf"] * plocha_ha
        years = np.arange(1, data["years"] + 1)
        mean_cf = np.mean(cf_total, axis=1)
        p5_cf = np.percentile(cf_total, 5, axis=1)
        p95_cf = np.percentile(cf_total, 95, axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=mean_cf, name='Průměrný roční zisk', line=dict(color='green', width=3)))
        fig.add_trace(go.Scatter(x=years, y=p95_cf, name='Optimistický (95%)', line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(x=years, y=p5_cf, name='Pesimistický (5%)', fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                       line=dict(width=0), showlegend=False))
        fig.update_layout(title=f"Cash Flow ({plocha_ha} ha)", xaxis_title="Rok", yaxis_title="Zisk (€)")
        st.plotly_chart(fig, use_container_width=True, key=f"cash_flow_{plodina}")

        total_profits = np.sum(cf_total, axis=0)
        var_5 = np.percentile(total_profits, 5)
        cvar_5 = total_profits[total_profits <= var_5].mean()
        avg_yearly_std = np.mean(np.std(cf_total, axis=0))

        # Analýza návratnosti
        cumulative_cf = np.cumsum(cf_total, axis=0)
        payback_years = np.argmax(cumulative_cf > 0, axis=0)
        successful_payback = payback_years[payback_years > 0]
        payback_prob = len(successful_payback) / n_sim if n_sim > 0 else 0
        avg_payback_year = np.mean(successful_payback) + 1 if len(successful_payback) > 0 else "N/A"

        # Uložení dat pro rekapitulaci
        summary_data.append({
            "Plodina": plodina,
            "Průměrný Zisk": np.mean(total_profits),
            "Potenciál (95%)": np.percentile(total_profits, 95),
            "Doba návratnosti (roky)": avg_payback_year,
            "Pravděpodobnost návratnosti": payback_prob,
            "VaR (5%)": var_5,
            "CVaR (5%)": cvar_5,
            "Prům. roční odchylka": avg_yearly_std
        })

        # Histogramy
        col_hist1, col_hist2 = st.columns(2)
        with col_hist1:
            fig_hist = go.Figure(data=[go.Histogram(x=total_profits, nbinsx=50, name='Rozdělení zisku', marker_color='#1f77b4')]) # Modrá
            fig_hist.update_layout(title="Rozdělení Celkového Zisku", xaxis_title="Celkový zisk (€)", yaxis_title="Počet simulací")
            st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{plodina}")
        with col_hist2:
            fig_payback = go.Figure(data=[go.Histogram(x=successful_payback + 1, name='Rok návratnosti', marker_color='#2ca02c')]) # Zelená
            fig_payback.update_layout(title="Rozdělení Doby Návratnosti", xaxis_title="Rok návratnosti investice", yaxis_title="Počet simulací")
            st.plotly_chart(fig_payback, use_container_width=True, key=f"payback_{plodina}")


        st.subheader("Klíčové ukazatele výkonu (KPIs)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Průměrný Celkový Zisk", f"{np.mean(total_profits):,.0f} €")
        col2.metric("Potenciál (95% percentil)", f"{np.percentile(total_profits, 95):,.0f} €")
        col3.metric("Prům. doba návratnosti (úspěšné)", f"{avg_payback_year:.2f} let" if isinstance(avg_payback_year, (int, float)) else "N/A")
        col4.metric("Pravděpodobnost návratnosti", f"{payback_prob:.1%}" if payback_prob >= 0.01 else f"{payback_prob:.4%}")

        st.subheader("Ukazatele Rizika")
        col1_risk, col2_risk, col3_risk = st.columns(3)
        col1_risk.metric("Prům. roční odchylka", f"{avg_yearly_std:,.0f} €")
        col2_risk.metric("Value at Risk (VaR 5%)", f"{var_5:,.0f} €", delta_color="inverse", help="Maximální očekávaná ztráta, která by neměla být překročena s 95% pravděpodobností.")
        col3_risk.metric("Conditional VaR (CVaR 5%)", f"{cvar_5:,.0f} €", delta_color="inverse", help="Průměrná ztráta v 5% nejhorších scénářů.")

        # Graf produkce
        st.subheader("🌱 Průměrná roční sklizeň")
        mean_yields = np.mean(data["yields"], axis=1)
        p5_yields = np.percentile(data["yields"], 5, axis=1)
        p95_yields = np.percentile(data["yields"], 95, axis=1)
        fig_yield = go.Figure()
        fig_yield.add_trace(go.Scatter(x=years, y=mean_yields, name='Průměrná sklizeň', line=dict(color='#ff7f0e')))
        fig_yield.add_trace(go.Scatter(x=years, y=p95_yields, name='Optimistický (95%)', line=dict(width=0), showlegend=False))
        fig_yield.add_trace(go.Scatter(x=years, y=p5_yields, name='Pesimistický (5%)', fill='tonexty', fillcolor='rgba(255, 127, 14, 0.2)', line=dict(width=0), showlegend=False))
        fig_yield.update_layout(title=f"Průměrná sklizeň pro {plodina}", xaxis_title="Rok", yaxis_title="Sklizeň (tuna/ha)")
        st.plotly_chart(fig_yield, use_container_width=True, key=f"yield_{plodina}")

        st.subheader("🎲 Analýza Citlivosti Rizika")
        st.markdown(
            "Graf ukazuje, jak se mění průměrná roční odchylka zisku v závislosti na pravděpodobnosti selhání sadby a špatného počasí.")

        with st.spinner(f"Počítám matici rizik pro {plodina}..."):
            x_vals, y_vals, z_matrix = calculate_sensitivity_matrix(
                data["type"],
                data["years"],
                data["params"],
                data["y_bounds"],
                "Direct Chip" if data["type"] == "src" else None,
                plocha_ha,
                n_sim,
                subsidy_percentage / 100.0
            )

            fig_3d = go.Figure(data=[go.Surface(
                z=z_matrix,
                x=x_vals,
                y=y_vals,
                colorscale='Viridis',
                colorbar=dict(title='Prům. roční odchylka (€)')
            )])

            fig_3d.update_layout(
                title=f'Průměrná roční odchylka v závislosti na rizicích',
                scene=dict(
                    xaxis_title='Pravděpodobnost selhání (%)',
                    yaxis_title='Pravděpodobnost sucha (%)',
                    zaxis_title='Prům. roční odchylka (€)',
                ),
                width=800,
                height=600,
                margin=dict(l=65, r=50, b=65, t=90)
            )
            
            st.plotly_chart(fig_3d, use_container_width=True, key=f"3d_{plodina}")

        # C. Graf vývoje ceny
        st.subheader("📈 Průměrný vývoj ceny komodity")
        mean_prices = np.mean(data["prices"], axis=1)
        p5_prices = np.percentile(data["prices"], 5, axis=1)
        p95_prices = np.percentile(data["prices"], 95, axis=1)
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=years, y=mean_prices, name='Průměrná cena', line=dict(color='purple')))
        fig_price.add_trace(go.Scatter(x=years, y=p95_prices, name='Optimistický (95%)', line=dict(width=0), showlegend=False))
        fig_price.add_trace(go.Scatter(x=years, y=p5_prices, name='Pesimistický (5%)', fill='tonexty', fillcolor='rgba(128, 0, 128, 0.2)', line=dict(width=0), showlegend=False))
        fig_price.update_layout(title=f"Průměrný vývoj ceny pro {plodina}", xaxis_title="Rok", yaxis_title="Cena (€/tuna)")
        st.plotly_chart(fig_price, use_container_width=True, key=f"price_{plodina}")

        st.divider()

    # Rekapitulace
    if len(summary_data) > 1:
        st.header("🏆 Rekapitulace a Porovnání")
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.set_index("Plodina")
        st.dataframe(summary_df.style.format({
            "Průměrný Zisk": "{:,.0f} €",
            "Potenciál (95%)": "{:,.0f} €",
            "Doba návratnosti (roky)": "{:.2f}",
            "Pravděpodobnost návratnosti": "{:.1%}",
            "VaR (5%)": "{:,.0f} €",
            "CVaR (5%)": "{:,.0f} €",
            "Prům. roční odchylka": "{:,.0f} €"
        }))