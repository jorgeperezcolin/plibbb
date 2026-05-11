"""
PLI 3B — Predictive Location Intelligence para Tiendas 3B
Demo ejecutiva en Streamlit para Business Data Scientists.

Ejecutar:
    streamlit run app.py
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="PLI 3B | Predictive Location Intelligence",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0B1F3A"
SECONDARY = "#163A5F"
GREEN = "#159A5B"
YELLOW = "#F2B705"
ORANGE = "#F26A21"
RED = "#C7352D"
BLUE = "#2A6FDB"
LIGHT = "#F5F7FA"
GRAY = "#667085"

DECISION_COLORS = {
    "Abrir": GREEN,
    "Posponer": ORANGE,
    "Descartar": RED,
    "Reubicar": BLUE,
}

RISK_COLORS = {
    "Bajo": GREEN,
    "Medio": YELLOW,
    "Alto": ORANGE,
    "Crítico": RED,
}


CUSTOM_CSS = f"""
<style>
    .main {{
        background: linear-gradient(180deg, #ffffff 0%, {LIGHT} 100%);
    }}
    .block-container {{
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {PRIMARY} 0%, #071426 100%);
    }}
    [data-testid="stSidebar"] * {{
        color: #ffffff;
    }}
    h1, h2, h3 {{
        color: {PRIMARY};
        letter-spacing: -0.02em;
    }}
    .hero-card {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
        padding: 26px 30px;
        border-radius: 22px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 20px 45px rgba(11,31,58,0.18);
    }}
    .hero-card h1 {{
        color: white;
        margin: 0;
        font-size: 2.15rem;
    }}
    .hero-card p {{
        color: #DDE7F3;
        font-size: 1.02rem;
        max-width: 980px;
    }}
    .concept-card {{
        background: white;
        border: 1px solid #E5EAF1;
        border-radius: 18px;
        padding: 18px;
        min-height: 150px;
        box-shadow: 0 10px 28px rgba(16,24,40,0.06);
    }}
    .metric-card {{
        background: white;
        border: 1px solid #E7ECF3;
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 12px 28px rgba(16,24,40,0.06);
        height: 100%;
    }}
    .metric-label {{
        color: {GRAY};
        font-size: 0.82rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
    }}
    .metric-value {{
        color: {PRIMARY};
        font-size: 1.82rem;
        font-weight: 800;
        margin-top: 4px;
    }}
    .metric-note {{
        color: {GRAY};
        font-size: 0.82rem;
        margin-top: 4px;
    }}
    .alert-card {{
        background: #FFF7ED;
        border-left: 5px solid {ORANGE};
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
        color: #7A3A08;
        font-weight: 600;
    }}
    .decision-box {{
        border-radius: 18px;
        padding: 18px;
        background: white;
        border: 1px solid #E5EAF1;
        box-shadow: 0 10px 28px rgba(16,24,40,0.06);
    }}
    .small-muted {{
        color: {GRAY};
        font-size: 0.9rem;
    }}
    .footer-note {{
        color: {GRAY};
        font-size: 0.82rem;
        margin-top: 24px;
        border-top: 1px solid #E5EAF1;
        padding-top: 12px;
    }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Datos sintéticos
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def make_sites(seed: int = 42, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = {
        "Valle de México": (19.4326, -99.1332, "Core"),
        "Centro-Bajío": (20.5888, -100.3899, "Core"),
        "Puebla-Veracruz": (19.0414, -98.2063, "Expansión reciente"),
        "Occidente": (20.6597, -103.3496, "Expansión reciente"),
        "Norte": (25.6866, -100.3161, "Nueva geografía"),
        "Sureste": (20.9674, -89.5926, "Nueva geografía"),
        "Pacífico": (24.8091, -107.3940, "Nueva geografía"),
    }
    cedis_by_region = {
        "Valle de México": "CEDIS Tultitlán",
        "Centro-Bajío": "CEDIS Querétaro",
        "Puebla-Veracruz": "CEDIS Puebla",
        "Occidente": "CEDIS Guadalajara",
        "Norte": "CEDIS Monterrey",
        "Sureste": "CEDIS Mérida",
        "Pacífico": "CEDIS Culiacán",
    }

    rows = []
    for i in range(n):
        region = rng.choice(list(regions.keys()), p=[0.21, 0.18, 0.14, 0.14, 0.13, 0.11, 0.09])
        lat0, lon0, geo_type = regions[region]
        lat = lat0 + rng.normal(0, 0.45)
        lon = lon0 + rng.normal(0, 0.55)
        score = int(np.clip(rng.normal(72 if geo_type == "Core" else 64, 13), 25, 97))
        cannibal = float(np.clip(rng.normal(0.18 if geo_type == "Nueva geografía" else 0.28, 0.12), 0.02, 0.68))
        cedis_distance = float(np.clip(rng.normal(42 if geo_type == "Core" else 110, 40), 8, 240))
        capex = float(np.clip(rng.normal(5.5, 0.95), 4.1, 8.6))
        payback = float(np.clip(37 - score * 0.18 + cannibal * 9 + cedis_distance * 0.025 + rng.normal(0, 2.1), 15, 48))
        probability = float(np.clip(1 / (1 + math.exp((payback - 26) / 4.5)), 0.04, 0.96))
        demand_new = float(np.clip(1 - cannibal + rng.normal(0, 0.06), 0.22, 0.98))
        ebitda = float(np.clip((score / 100) * 5.3 - cannibal * 1.4 - capex * 0.12 + rng.normal(0, 0.25), 0.8, 5.8))

        if score >= 76 and payback <= 26 and cannibal < 0.38:
            decision = "Abrir"
        elif score < 50 or payback > 36:
            decision = "Descartar"
        elif cannibal >= 0.42:
            decision = "Reubicar"
        else:
            decision = "Posponer"

        if payback <= 24 and cannibal < 0.25:
            risk = "Bajo"
        elif payback <= 29 and cannibal < 0.38:
            risk = "Medio"
        elif payback <= 36:
            risk = "Alto"
        else:
            risk = "Crítico"

        rows.append(
            {
                "site_id": f"T3B-{1000+i}",
                "sitio": f"Sitio candidato {i+1:03d}",
                "region": region,
                "tipo_geografia": geo_type,
                "lat": lat,
                "lon": lon,
                "score_posar": score,
                "decision": decision,
                "riesgo": risk,
                "cedis": cedis_by_region[region],
                "payback_meses": round(payback, 1),
                "prob_payback_26": round(probability, 2),
                "canibalizacion": round(cannibal, 2),
                "demanda_nueva": round(demand_new, 2),
                "demanda_transferida": round(1 - demand_new, 2),
                "distancia_cedis_km": round(cedis_distance, 1),
                "capex_mdp": round(capex, 2),
                "ebitda_mdp": round(ebitda, 2),
                "color_decision": DECISION_COLORS[decision],
                "color_riesgo": RISK_COLORS[risk],
                "aperturas_sugeridas": int(np.clip(rng.normal(80 if decision == "Abrir" else 35, 18), 10, 140)),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def make_cohorts(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    cohorts = ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4", "2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4", "2026-Q1"]
    regions = ["Core", "Expansión reciente", "Nueva geografía"]
    for cohort in cohorts:
        for geo in regions:
            for month in [6, 12, 18, 24]:
                base = 15 if geo == "Core" else 11 if geo == "Expansión reciente" else 7
                forecast = base + month * 0.19 + rng.normal(0, 1.0)
                real = forecast + rng.normal(1.0 if geo == "Core" else -0.8 if geo == "Expansión reciente" else -2.2, 1.5)
                expected_payback = 24 if geo == "Core" else 28 if geo == "Expansión reciente" else 33
                actual_payback = expected_payback + rng.normal(-1 if real > forecast else 3, 2.2)
                rows.append(
                    {
                        "cohorte": cohort,
                        "geografia": geo,
                        "edad_meses": month,
                        "sss_forecast": round(forecast, 1),
                        "sss_real": round(real, 1),
                        "desviacion_sss": round(real - forecast, 1),
                        "payback_esperado": round(expected_payback, 1),
                        "payback_real": round(actual_payback, 1),
                    }
                )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def make_regional_plan(seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = ["Valle de México", "Centro-Bajío", "Puebla-Veracruz", "Occidente", "Norte", "Sureste", "Pacífico"]
    rows = []
    for idx, region in enumerate(regions, start=1):
        capacity = int(np.clip(rng.normal(78 if idx < 5 else 56, 15), 32, 96))
        density = int(np.clip(rng.normal(70 if idx < 5 else 45, 16), 20, 94))
        demand = int(np.clip(rng.normal(82 if idx < 5 else 66, 12), 38, 96))
        logistics_risk = max(0, 100 - capacity + rng.normal(0, 5))
        priority = 0.42 * demand + 0.35 * density + 0.23 * capacity - 0.20 * logistics_risk
        rows.append(
            {
                "region": region,
                "prioridad_base": round(priority, 1),
                "capacidad_cedis_pct": capacity,
                "densidad_minima_pct": density,
                "atractivo_demanda_pct": demand,
                "riesgo_logistico_pct": round(logistics_risk, 1),
                "aperturas_base": int(np.clip(priority * 1.1, 35, 115)),
                "trimestre_recomendado": f"2026-Q{min(4, max(1, math.ceil(idx/2)))}",
            }
        )
    return pd.DataFrame(rows).sort_values("prioridad_base", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def make_decision_register(sites: pd.DataFrame, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    responsables = ["DRI Norte", "DRI Centro", "DRI Occidente", "DRI Sureste", "DRI Desarrollo Corp."]
    evidencias = ["Score PoSAR + forecast", "Visita campo + isócrona", "Comité CEDIS", "Benchmark cohorte", "Stress test financiero"]
    resultados = ["En seguimiento", "Aprobado", "Rechazado", "Recalibrar modelo", "Revisar inmueble"]
    base_date = date(2026, 5, 1)
    sample = sites.sample(55, random_state=seed).copy()
    rows = []
    for _, row in sample.iterrows():
        exception = rng.choice(["No", "Sí"], p=[0.82, 0.18])
        committee_decision = row["decision"] if exception == "No" else rng.choice(["Abrir", "Posponer", "Reubicar"])
        rows.append(
            {
                "sitio": row["sitio"],
                "region": row["region"],
                "responsable_dri": rng.choice(responsables),
                "fecha_decision": base_date - timedelta(days=int(rng.integers(0, 120))),
                "recomendacion_modelo": row["decision"],
                "decision_comite": committee_decision,
                "excepcion_aprobada": exception,
                "evidencia_utilizada": rng.choice(evidencias),
                "resultado_posterior": rng.choice(resultados),
                "score_posar": row["score_posar"],
                "payback_meses": row["payback_meses"],
            }
        )
    return pd.DataFrame(rows)


sites = make_sites()
cohorts = make_cohorts()
regional_plan = make_regional_plan()
decision_register = make_decision_register(sites)


# -----------------------------------------------------------------------------
# Componentes visuales
# -----------------------------------------------------------------------------

def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def executive_alert(text: str) -> None:
    st.markdown(f"<div class='alert-card'>⚠️ {text}</div>", unsafe_allow_html=True)


def section_title(title: str, caption: str | None = None) -> None:
    st.subheader(title)
    if caption:
        st.markdown(f"<div class='small-muted'>{caption}</div>", unsafe_allow_html=True)


def fig_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color=PRIMARY),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def score_gauge(score: int, title: str = "Score PoSAR") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 42}},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": PRIMARY},
                "steps": [
                    {"range": [0, 50], "color": "#FDECEC"},
                    {"range": [50, 70], "color": "#FFF6D7"},
                    {"range": [70, 85], "color": "#E7F6EE"},
                    {"range": [85, 100], "color": "#D7ECFF"},
                ],
                "threshold": {"line": {"color": RED, "width": 4}, "thickness": 0.75, "value": 76},
            },
        )
    )
    return fig_layout(fig, 300)


def forecast_for_site(row: pd.Series) -> pd.DataFrame:
    months = np.arange(1, 25)
    start = 2.2 + row.score_posar / 48
    maturity = 1 - np.exp(-months / 8)
    cannibal_drag = row.canibalizacion * 0.75
    sales = (start + 2.7 * maturity - cannibal_drag) * (1 + np.sin(months / 3) * 0.025)
    low = sales * 0.88
    high = sales * 1.12
    return pd.DataFrame({"mes": months, "ventas_mdp": sales, "banda_baja": low, "banda_alta": high})


# -----------------------------------------------------------------------------
# Páginas
# -----------------------------------------------------------------------------

def page_command_center() -> None:
    hero(
        "PLI 3B — Executive Command Center",
        "No es un mapa. Es un sistema operativo de decisión territorial para proteger CapEx, payback, SSS por cohorte y EBITDA de las próximas aperturas.",
    )

    total_sites = len(sites)
    open_sites = int((sites.decision == "Abrir").sum())
    postponed = int((sites.decision == "Posponer").sum())
    discarded = int((sites.decision == "Descartar").sum())
    avg_payback = sites.payback_meses.mean()
    high_risk = int(sites.riesgo.isin(["Alto", "Crítico"]).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("CapEx anual", "Ps.3,300M", "590–630 aperturas")
    with c2:
        metric_card("CapEx en riesgo", "Ps.165–330M", "5–10% subóptimo")
    with c3:
        metric_card("Sitios evaluados", f"{total_sites}", "pipeline sintético")
    with c4:
        metric_card("Payback esperado", f"{avg_payback:.1f} meses", "objetivo ≤26 meses")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card("Recomendadas", f"{open_sites}", "abrir")
    with c6:
        metric_card("Pospuestas", f"{postponed}", "requieren densidad")
    with c7:
        metric_card("Descartadas", f"{discarded}", "riesgo económico")
    with c8:
        metric_card("Riesgo alto/crítico", f"{high_risk}", "por región y cohorte")

    st.markdown("---")
    left, right = st.columns([1.25, 1])
    with left:
        section_title("CapEx en riesgo por región", "Estimación sintética de CapEx comprometido en sitios con riesgo alto o crítico.")
        risk_df = sites.assign(capex_riesgo=lambda d: np.where(d.riesgo.isin(["Alto", "Crítico"]), d.capex_mdp, 0))
        risk_region = risk_df.groupby("region", as_index=False)["capex_riesgo"].sum().sort_values("capex_riesgo", ascending=False)
        fig = px.bar(risk_region, x="region", y="capex_riesgo", text_auto=".1f", labels={"capex_riesgo": "CapEx en riesgo (Ps.M)", "region": "Región"})
        fig.update_traces(marker_color=ORANGE)
        st.plotly_chart(fig_layout(fig), use_container_width=True)

    with right:
        section_title("Distribución de decisiones", "De decisiones aisladas a un portafolio gobernado.")
        decision_df = sites.decision.value_counts().reset_index()
        decision_df.columns = ["decision", "sitios"]
        fig = px.pie(decision_df, names="decision", values="sitios", hole=0.58, color="decision", color_discrete_map=DECISION_COLORS)
        st.plotly_chart(fig_layout(fig), use_container_width=True)

    st.markdown("### Alertas ejecutivas")
    alert_cols = st.columns(3)
    with alert_cols[0]:
        executive_alert("Región Norte presenta densidad logística insuficiente para apertura masiva simultánea.")
    with alert_cols[1]:
        executive_alert("12 sitios tienen canibalización estimada superior a 42% en isócrona peatonal.")
    with alert_cols[2]:
        executive_alert("Cohortes de nueva geografía muestran maduración inferior al forecast a 12 meses.")

    st.markdown("### Arquitectura conceptual")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="concept-card"><h3>PLI</h3><p>Predictive Location Intelligence evalúa sitios, regiones, canibalización, ventas, payback y riesgo territorial.</p></div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="concept-card"><h3>PoSAR</h3><p>Representación analítica del sitio: microzona, isócrona, competencia, demanda, CEDIS y riesgo explicable.</p></div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="concept-card"><h3>DOS</h3><p>Decision Operating System que conecta modelo, comité, accountability, P&L, trazabilidad y aprendizaje.</p></div>
        """, unsafe_allow_html=True)


def page_map() -> None:
    hero("Mapa de Decisión Territorial", "Cada punto representa una decisión de capital: abrir, posponer, reubicar o descartar.")

    with st.sidebar:
        st.markdown("### Filtros del mapa")
        selected_regions = st.multiselect("Región", sorted(sites.region.unique()), default=sorted(sites.region.unique()))
        selected_risk = st.multiselect("Nivel de riesgo", ["Bajo", "Medio", "Alto", "Crítico"], default=["Bajo", "Medio", "Alto", "Crítico"])
        selected_decisions = st.multiselect("Decisión recomendada", list(DECISION_COLORS.keys()), default=list(DECISION_COLORS.keys()))
        score_range = st.slider("Rango de Score PoSAR", 0, 100, (45, 100))

    filtered = sites[
        sites.region.isin(selected_regions)
        & sites.riesgo.isin(selected_risk)
        & sites.decision.isin(selected_decisions)
        & sites.score_posar.between(score_range[0], score_range[1])
    ]

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Sitios filtrados", str(len(filtered)), "pipeline visible")
    with c2:
        metric_card("Score promedio", f"{filtered.score_posar.mean() if len(filtered) else 0:.1f}", "PoSAR")
    with c3:
        metric_card("CapEx filtrado", f"Ps.{filtered.capex_mdp.sum():.1f}M", "estimado")

    if filtered.empty:
        st.warning("No hay sitios con los filtros seleccionados.")
        return

    fig = px.scatter_mapbox(
        filtered,
        lat="lat",
        lon="lon",
        color="decision",
        size="capex_mdp",
        hover_name="sitio",
        hover_data={
            "region": True,
            "score_posar": True,
            "riesgo": True,
            "cedis": True,
            "payback_meses": True,
            "capex_mdp": True,
            "lat": False,
            "lon": False,
        },
        color_discrete_map=DECISION_COLORS,
        zoom=4.3,
        center={"lat": 22.5, "lon": -101.5},
        height=620,
    )
    fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    section_title("Pipeline territorial", "Tabla ejecutiva para revisar sitios filtrados.")
    st.dataframe(
        filtered[["sitio", "region", "score_posar", "decision", "riesgo", "cedis", "payback_meses", "capex_mdp", "canibalizacion"]].sort_values("score_posar", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def page_scorecard() -> None:
    hero("Site Candidate Scorecard", "PoSAR convierte cada sitio candidato en un objeto analítico defendible ante comité.")

    site_name = st.selectbox("Selecciona sitio candidato", sites.sort_values("score_posar", ascending=False).sitio.tolist())
    row = sites.loc[sites.sitio == site_name].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Recomendación", row.decision, f"Riesgo {row.riesgo}")
    with c2:
        metric_card("Payback", f"{row.payback_meses:.1f} meses", "objetivo ≤26 meses")
    with c3:
        metric_card("Prob. payback ≤26", f"{row.prob_payback_26:.0%}", "intervalo sintético")
    with c4:
        metric_card("CapEx", f"Ps.{row.capex_mdp:.2f}M", row.cedis)

    left, right = st.columns([0.9, 1.3])
    with left:
        st.plotly_chart(score_gauge(int(row.score_posar)), use_container_width=True)
        st.markdown(
            f"""
            <div class="decision-box">
                <h3>Justificación para comité</h3>
                <p><b>{row.sitio}</b> en <b>{row.region}</b> obtiene una recomendación de <b>{row.decision}</b> por su Score PoSAR de <b>{row.score_posar}/100</b>, payback esperado de <b>{row.payback_meses:.1f} meses</b> y canibalización estimada de <b>{row.canibalizacion:.0%}</b>.</p>
                <p>La decisión debe revisarse contra capacidad logística del <b>{row.cedis}</b>, densidad peatonal de microzona y maduración esperada de cohortes comparables.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        forecast = forecast_for_site(row)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast.mes, y=forecast.banda_alta, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast.mes, y=forecast.banda_baja, mode="lines", fill="tonexty", line=dict(width=0), name="Intervalo confianza"))
        fig.add_trace(go.Scatter(x=forecast.mes, y=forecast.ventas_mdp, mode="lines+markers", name="Forecast ventas"))
        fig.update_layout(title="Forecast de ventas mensuales a 24 meses", xaxis_title="Mes", yaxis_title="Ventas (Ps.M)")
        st.plotly_chart(fig_layout(fig), use_container_width=True)

        drivers = pd.DataFrame(
            {
                "driver": ["Densidad peatonal", "Perfil socioeconómico", "Competencia", "Canibalización", "Distancia CEDIS", "Formato inmueble"],
                "impacto": [18, 14, -8, -int(row.canibalizacion * 30), -int(row.distancia_cedis_km / 10), 9 if row.capex_mdp < 5.7 else -4],
            }
        )
        fig2 = px.bar(drivers, x="impacto", y="driver", orientation="h", title="Drivers explicables del score")
        fig2.update_traces(marker_color=np.where(drivers.impacto >= 0, GREEN, RED))
        st.plotly_chart(fig_layout(fig2, 300), use_container_width=True)

    section_title("Detalle analítico PoSAR")
    st.dataframe(
        pd.DataFrame(
            [
                ["Demanda nueva", f"{row.demanda_nueva:.0%}"],
                ["Demanda transferida", f"{row.demanda_transferida:.0%}"],
                ["Riesgo de canibalización", f"{row.canibalizacion:.0%}"],
                ["Distancia al CEDIS", f"{row.distancia_cedis_km:.1f} km"],
                ["EBITDA esperado", f"Ps.{row.ebitda_mdp:.2f}M"],
                ["Tipo de geografía", row.tipo_geografia],
            ],
            columns=["Variable", "Valor"],
        ),
        use_container_width=True,
        hide_index=True,
    )


def page_cohorts() -> None:
    hero("Cohort Performance Monitor", "El sistema aprende con cada apertura: forecast → decisión → apertura → desempeño real → recalibración.")

    geo = st.selectbox("Geografía", ["Todas"] + sorted(cohorts.geografia.unique().tolist()))
    df = cohorts.copy() if geo == "Todas" else cohorts[cohorts.geografia == geo].copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Tiendas <3 años", "45%", "base de maduración")
    with c2:
        metric_card("Desviación SSS prom.", f"{df.desviacion_sss.mean():+.1f} pp", "real vs forecast")
    with c3:
        metric_card("Payback real prom.", f"{df.payback_real.mean():.1f} meses", "por cohorte")

    line_df = df.groupby(["edad_meses", "geografia"], as_index=False)[["sss_forecast", "sss_real"]].mean()
    line_melt = line_df.melt(id_vars=["edad_meses", "geografia"], value_vars=["sss_forecast", "sss_real"], var_name="serie", value_name="sss")
    fig = px.line(line_melt, x="edad_meses", y="sss", color="serie", line_dash="geografia", markers=True, title="SSS forecast vs. real por edad de tienda")
    st.plotly_chart(fig_layout(fig), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        cohort_summary = df.groupby("cohorte", as_index=False)["desviacion_sss"].mean()
        fig2 = px.bar(cohort_summary, x="cohorte", y="desviacion_sss", title="Desviación promedio por cohorte")
        fig2.update_traces(marker_color=np.where(cohort_summary.desviacion_sss >= 0, GREEN, RED))
        st.plotly_chart(fig_layout(fig2), use_container_width=True)

    with right:
        payback = df.groupby("geografia", as_index=False)[["payback_esperado", "payback_real"]].mean()
        payback_melt = payback.melt(id_vars="geografia", var_name="serie", value_name="meses")
        fig3 = px.bar(payback_melt, x="geografia", y="meses", color="serie", barmode="group", title="Payback real vs. esperado")
        st.plotly_chart(fig_layout(fig3), use_container_width=True)

    section_title("Desviaciones que alimentan el learning loop")
    table = df.sort_values(["desviacion_sss", "payback_real"]).head(20)
    st.dataframe(table, use_container_width=True, hide_index=True)


def page_regional_optimizer() -> None:
    hero("Regional Sequencing Optimizer", "Secuenciar regiones evita que la velocidad de aperturas presione margen, logística y maduración de cohortes.")

    scenario = st.selectbox("Escenario", ["Conservador", "Base", "Agresivo"])
    factor = {"Conservador": 0.78, "Base": 1.0, "Agresivo": 1.26}[scenario]
    df = regional_plan.copy()
    df["aperturas_escenario"] = (df.aperturas_base * factor).round(0).astype(int)
    df["impacto_ebitda_mdp"] = (df.aperturas_escenario * (df.prioridad_base / 100) * 0.85 - df.riesgo_logistico_pct * 0.08).round(1)
    df["capex_requerido_mdp"] = (df.aperturas_escenario * 5.5).round(1)
    df["maduracion_estimada_meses"] = np.clip(38 - df.prioridad_base * 0.14 + df.riesgo_logistico_pct * 0.05, 20, 42).round(1)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Aperturas escenario", f"{df.aperturas_escenario.sum()}", scenario)
    with c2:
        metric_card("CapEx requerido", f"Ps.{df.capex_requerido_mdp.sum():.0f}M", "portfolio regional")
    with c3:
        metric_card("EBITDA esperado", f"Ps.{df.impacto_ebitda_mdp.sum():.1f}M", "impacto anualizado")

    if scenario == "Agresivo":
        executive_alert("Escenario agresivo incrementa el riesgo de repetir presión de margen por aperturas simultáneas sin densidad logística mínima.")

    left, right = st.columns([1.2, 1])
    with left:
        section_title("Roadmap priorizado de entrada regional")
        st.dataframe(
            df[["region", "trimestre_recomendado", "prioridad_base", "aperturas_escenario", "capacidad_cedis_pct", "densidad_minima_pct", "capex_requerido_mdp", "maduracion_estimada_meses"]],
            use_container_width=True,
            hide_index=True,
        )
    with right:
        fig = px.bar(df.sort_values("impacto_ebitda_mdp", ascending=True), x="impacto_ebitda_mdp", y="region", orientation="h", title="Impacto EBITDA por región")
        fig.update_traces(marker_color=GREEN)
        st.plotly_chart(fig_layout(fig), use_container_width=True)

    section_title("Capacidad logística vs. densidad mínima")
    fig2 = px.scatter(
        df,
        x="capacidad_cedis_pct",
        y="densidad_minima_pct",
        size="aperturas_escenario",
        color="riesgo_logistico_pct",
        hover_name="region",
        color_continuous_scale="OrRd",
        title="Balance entre CEDIS, densidad y ritmo de apertura",
    )
    fig2.add_hline(y=60, line_dash="dash", line_color=GRAY)
    fig2.add_vline(x=60, line_dash="dash", line_color=GRAY)
    st.plotly_chart(fig_layout(fig2), use_container_width=True)


def page_governance() -> None:
    hero("Decision Register / Governance", "Trazabilidad y accountability: cada excepción queda documentada, cada decisión se puede auditar.")

    with st.sidebar:
        st.markdown("### Filtros de gobierno")
        regions = st.multiselect("Región", sorted(decision_register.region.unique()), default=sorted(decision_register.region.unique()))
        dris = st.multiselect("Responsable", sorted(decision_register.responsable_dri.unique()), default=sorted(decision_register.responsable_dri.unique()))
        decisions = st.multiselect("Decisión", sorted(decision_register.decision_comite.unique()), default=sorted(decision_register.decision_comite.unique()))
        exception = st.multiselect("Excepción", ["No", "Sí"], default=["No", "Sí"])

    df = decision_register[
        decision_register.region.isin(regions)
        & decision_register.responsable_dri.isin(dris)
        & decision_register.decision_comite.isin(decisions)
        & decision_register.excepcion_aprobada.isin(exception)
    ]

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Decisiones registradas", str(len(df)), "con evidencia")
    with c2:
        metric_card("Excepciones", f"{(df.excepcion_aprobada == 'Sí').sum()}", "aprobadas")
    with c3:
        match_rate = (df.recomendacion_modelo == df.decision_comite).mean() if len(df) else 0
        metric_card("Alineación modelo-comité", f"{match_rate:.0%}", "sin excepción")

    st.dataframe(
        df.sort_values("fecha_decision", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x="decision_comite", color="excepcion_aprobada", barmode="group", title="Decisiones por comité y excepción")
        st.plotly_chart(fig_layout(fig), use_container_width=True)
    with right:
        fig2 = px.box(df, x="decision_comite", y="payback_meses", color="decision_comite", title="Payback por decisión tomada", color_discrete_map=DECISION_COLORS)
        st.plotly_chart(fig_layout(fig2), use_container_width=True)


def page_investment_committee() -> None:
    hero("Comité de Inversión — Vista Ejecutiva", "Cada apertura es una decisión de capital: priorizar, aprobar, posponer o pedir más evidencia.")

    if "committee_actions" not in st.session_state:
        st.session_state.committee_actions = {}

    shortlist = sites.sort_values(["score_posar", "prob_payback_26"], ascending=False).head(12).copy()
    shortlist["riesgo_financiero"] = np.where(shortlist.riesgo.isin(["Alto", "Crítico"]), "Alto", np.where(shortlist.riesgo == "Medio", "Medio", "Bajo"))
    shortlist["valor_prioridad"] = (shortlist.score_posar * shortlist.prob_payback_26 * (1 - shortlist.canibalizacion)).round(1)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Shortlist comité", str(len(shortlist)), "sitios priorizados")
    with c2:
        metric_card("CapEx requerido", f"Ps.{shortlist.capex_mdp.sum():.1f}M", "shortlist")
    with c3:
        metric_card("EBITDA esperado", f"Ps.{shortlist.ebitda_mdp.sum():.1f}M", "anualizado")

    st.markdown("### Lista priorizada")
    st.dataframe(
        shortlist[["sitio", "region", "capex_mdp", "ebitda_mdp", "riesgo_financiero", "payback_meses", "score_posar", "decision", "valor_prioridad"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Simulación de decisión del comité")
    selected = st.selectbox("Sitio para decisión", shortlist.sitio.tolist())
    selected_row = shortlist[shortlist.sitio == selected].iloc[0]

    st.markdown(
        f"""
        <div class="decision-box">
            <h3>{selected_row.sitio} · {selected_row.region}</h3>
            <p><b>Recomendación:</b> {selected_row.decision} · <b>Score PoSAR:</b> {selected_row.score_posar}/100 · <b>Payback:</b> {selected_row.payback_meses:.1f} meses · <b>CapEx:</b> Ps.{selected_row.capex_mdp:.2f}M · <b>EBITDA esperado:</b> Ps.{selected_row.ebitda_mdp:.2f}M</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2, b3, b4 = st.columns(4)
    actions = ["Aprobar", "Posponer", "Solicitar más evidencia", "Rechazar"]
    for col, action in zip([b1, b2, b3, b4], actions):
        with col:
            if st.button(action, key=f"{selected}-{action}", use_container_width=True):
                st.session_state.committee_actions[selected] = action
                st.success(f"Decisión registrada: {action}")

    if st.session_state.committee_actions:
        st.markdown("### Decisiones tomadas en esta sesión")
        actions_df = pd.DataFrame(
            [{"sitio": k, "acción_comité": v} for k, v in st.session_state.committee_actions.items()]
        )
        st.dataframe(actions_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="footer-note">
        Mensaje clave: PLI reduce CapEx inmovilizado en sitios subóptimos y permite defender la tesis de 14,000–20,000 tiendas con evidencia, no solo con narrativa.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Navegación
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("# PLI 3B")
    st.markdown("**Predictive Location Intelligence**")
    st.markdown("Business Data Scientists")
    st.markdown("---")
    page = st.radio(
        "Navegación",
        [
            "1. Executive Command Center",
            "2. Mapa de Decisión Territorial",
            "3. Site Candidate Scorecard",
            "4. Cohort Performance Monitor",
            "5. Regional Sequencing Optimizer",
            "6. Decision Register / Governance",
            "7. Comité de Inversión",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Demo con datos sintéticos. Sin APIs externas.")

if page.startswith("1."):
    page_command_center()
elif page.startswith("2."):
    page_map()
elif page.startswith("3."):
    page_scorecard()
elif page.startswith("4."):
    page_cohorts()
elif page.startswith("5."):
    page_regional_optimizer()
elif page.startswith("6."):
    page_governance()
else:
    page_investment_committee()
