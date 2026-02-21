import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import CoolProp.CoolProp as CP
import streamlit.components.v1 as components
import pandas as pd

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="AD-HTC Gas Cycle | Energhx",
    layout="wide",
    page_icon="🟣",
    initial_sidebar_state="expanded"
)

# ==========================================
# Custom CSS
# ==========================================
st.markdown("""
<style>
  /* Dark theme */
  .stApp { background-color: #0a0a14; }
  [data-testid="stSidebar"] { background: linear-gradient(180deg, #120824 0%, #0d1a2e 100%); border-right: 1px solid #3d1a6e; }
  [data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] span { color: #dcc6f0 !important; }

  /* Section label pills */
  .sec-pill {
    display:inline-block; background:linear-gradient(90deg,#5b1fa8,#8e44ad);
    color:#fff; padding:5px 14px; border-radius:20px; font-size:11px;
    font-weight:700; letter-spacing:0.8px; text-transform:uppercase;
    margin:10px 0 6px 0; width:100%; text-align:center;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background:linear-gradient(135deg,#150a2e,#1e1045);
    border:1px solid #5b1fa8; border-radius:12px; padding:14px;
    box-shadow:0 4px 20px rgba(91,31,168,0.35);
  }
  [data-testid="stMetricValue"] { color:#c39bd3 !important; font-size:1.5rem !important; font-weight:800 !important; }
  [data-testid="stMetricLabel"] { color:#9b59b6 !important; font-weight:600 !important; font-size:0.78rem !important; }
  [data-testid="stMetricDelta"] { font-size:0.75rem !important; }

  /* Tabs */
  [data-baseweb="tab-list"] { background:#120824; border-radius:10px; padding:4px; }
  [data-baseweb="tab"] { color:#c39bd3 !important; border-radius:8px; font-weight:600; }
  [aria-selected="true"] { background:linear-gradient(90deg,#5b1fa8,#8e44ad) !important; color:#fff !important; }

  /* Inputs */
  [data-testid="stNumberInput"] input {
    background:#1a0d35 !important; color:#e0cfff !important;
    border:1px solid #5b1fa8 !important; border-radius:6px !important;
  }
  .stSlider > div > div > div { background:#5b1fa8 !important; }

  /* Headers */
  h1,h2,h3 { color:#c39bd3 !important; }
  .stMarkdown h3 { border-bottom:1px solid #3d1a6e; padding-bottom:6px; }

  /* Analyze button */
  [data-testid="baseButton-primary"] {
    background:linear-gradient(90deg,#5b1fa8,#9b59b6) !important;
    border:none !important; font-weight:800 !important;
    letter-spacing:1.2px !important; font-size:15px !important;
    box-shadow:0 4px 18px rgba(91,31,168,0.55) !important;
    transition:all 0.2s !important;
  }

  /* Info/error boxes */
  [data-testid="stInfo"] { background:#150a2e; border-left:4px solid #5b1fa8; }
  [data-testid="stAlert"] { border-radius:10px; }

  /* Dataframe */
  [data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }

  /* Caption */
  .stCaption { color:#7a5fa0 !important; }

  hr { border-color:#2d1660 !important; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Thermodynamic Functions
# ==========================================

def calculate_steam_cycle(P_cond_kPa, P_boiler_kPa, T_boiler_C, pump_eff_pct, turb_eff_pct):
    """Full Rankine cycle calculation with saturation dome."""
    P1 = P_cond_kPa * 1000.0
    P2 = P_boiler_kPa * 1000.0
    T3 = T_boiler_C + 273.15
    eta_p = pump_eff_pct / 100.0
    eta_t = turb_eff_pct / 100.0

    # State 1: Saturated liquid at condenser pressure
    h1 = CP.PropsSI('H','P',P1,'Q',0,'Water')
    s1 = CP.PropsSI('S','P',P1,'Q',0,'Water')
    v1 = 1.0 / CP.PropsSI('D','P',P1,'Q',0,'Water')
    T1 = CP.PropsSI('T','P',P1,'Q',0,'Water') - 273.15

    # State 2: After pump (isentropic + efficiency)
    h2s = h1 + v1*(P2-P1)
    h2  = h1 + (h2s-h1)/eta_p
    s2  = CP.PropsSI('S','P',P2,'H',h2,'Water')
    T2  = CP.PropsSI('T','P',P2,'H',h2,'Water') - 273.15

    # State 3: Superheated steam at boiler outlet
    h3 = CP.PropsSI('H','P',P2,'T',T3,'Water')
    s3 = CP.PropsSI('S','P',P2,'T',T3,'Water')

    # State 4: After turbine
    h4s = CP.PropsSI('H','P',P1,'S',s3,'Water')
    h4  = h3 - (h3-h4s)*eta_t
    s4  = CP.PropsSI('S','P',P1,'H',h4,'Water')
    T4  = CP.PropsSI('T','P',P1,'H',h4,'Water') - 273.15

    # Saturation dome
    P_sat_arr = np.linspace(700, CP.PropsSI('Pcrit','Water')*0.9995, 300)
    hf = np.array([CP.PropsSI('H','P',p,'Q',0,'Water')/1000 for p in P_sat_arr])
    sf = np.array([CP.PropsSI('S','P',p,'Q',0,'Water')/1000 for p in P_sat_arr])
    hg = np.array([CP.PropsSI('H','P',p,'Q',1,'Water')/1000 for p in P_sat_arr])
    sg = np.array([CP.PropsSI('S','P',p,'Q',1,'Water')/1000 for p in P_sat_arr])

    h_cycle = np.array([h1,h2,h3,h4,h1]) / 1000.0
    s_cycle = np.array([s1,s2,s3,s4,s1]) / 1000.0
    T_states = [T1, T2, T3-273.15+273.15, T4]  # in °C
    T_states = [T1, T2, T_boiler_C, T4]

    pump_w   = (h2-h1)/1000
    boiler_q = (h3-h2)/1000
    turb_w   = (h3-h4)/1000
    cond_q   = (h4-h1)/1000
    net_w    = turb_w - pump_w
    eta      = net_w/boiler_q * 100

    states = {
        'labels': ['1 (Cond Out)','2 (Pump Out)','3 (Turb In)','4 (Turb Out)'],
        'T_C':    [T1, T2, T_boiler_C, T4],
        'P_kPa':  [P_cond_kPa, P_boiler_kPa, P_boiler_kPa, P_cond_kPa],
        'h_kJ':   [h1/1000, h2/1000, h3/1000, h4/1000],
        's_kJ':   [s1/1000, s2/1000, s3/1000, s4/1000],
    }
    return h_cycle, s_cycle, sf, hf, sg, hg, pump_w, boiler_q, turb_w, cond_q, net_w, eta, states


def calculate_gas_cycle_THdot(pr, T1_C, T3_C, eta_c, eta_t, m_dot=1.0):
    """
    Rigorous T-Hdot diagram using CoolProp Air properties.
    Returns process segments as (H_dot, T) arrays for each process.
    """
    T1 = T1_C + 273.15
    T3 = T3_C + 273.15
    P1 = 101325.0
    P2 = P1 * pr
    ec = eta_c / 100.0
    et = eta_t / 100.0

    h1 = CP.PropsSI('H','P',P1,'T',T1,'Air')
    s1 = CP.PropsSI('S','P',P1,'T',T1,'Air')

    # Isentropic compression
    h2s = CP.PropsSI('H','P',P2,'S',s1,'Air')
    h2  = h1 + (h2s-h1)/ec
    T2  = CP.PropsSI('T','P',P2,'H',h2,'Air')
    s2  = CP.PropsSI('S','P',P2,'H',h2,'Air')

    h3  = CP.PropsSI('H','P',P2,'T',T3,'Air')
    s3  = CP.PropsSI('S','P',P2,'T',T3,'Air')

    h4s = CP.PropsSI('H','P',P1,'S',s3,'Air')
    h4  = h3 - (h3-h4s)*et
    T4  = CP.PropsSI('T','P',P1,'H',h4,'Air')
    s4  = CP.PropsSI('S','P',P1,'H',h4,'Air')

    # ---- Build T-Hdot traces for each process ----
    # Process 1-2: Compression (N points along pressure path)
    N = 60
    P_12 = np.linspace(P1, P2, N)
    # Use actual path: polytropic with real gas props
    H_12, T_12 = [], []
    for P_i in P_12:
        # interpolate enthalpy linearly (actual irreversible path)
        frac = (np.log(P_i)-np.log(P1))/(np.log(P2)-np.log(P1))
        h_i = h1 + frac*(h2-h1)
        T_i = CP.PropsSI('T','P',P_i,'H',h_i,'Air') - 273.15
        H_12.append((h_i - h1)*m_dot/1000)
        T_12.append(T_i)

    # Process 2-3: Combustion/Heat addition at constant pressure P2
    N2 = 80
    T_23_arr = np.linspace(T2, T3, N2)
    H_23, T_23 = [], []
    for T_i in T_23_arr:
        h_i = CP.PropsSI('H','P',P2,'T',T_i,'Air')
        H_23.append((h_i - h1)*m_dot/1000)
        T_23.append(T_i - 273.15)

    # Process 3-4: Expansion (turbine)
    P_34 = np.linspace(P2, P1, N)
    H_34, T_34 = [], []
    for P_i in P_34:
        frac = (np.log(P2)-np.log(P_i))/(np.log(P2)-np.log(P1))
        h_i = h3 - frac*(h3-h4)
        T_i = CP.PropsSI('T','P',P_i,'H',h_i,'Air') - 273.15
        H_34.append((h_i - h1)*m_dot/1000)
        T_34.append(T_i)

    # Process 4-1: Heat rejection at P1
    T_41_arr = np.linspace(T4, T1, N)
    H_41, T_41 = [], []
    for T_i in T_41_arr:
        h_i = CP.PropsSI('H','P',P1,'T',T_i,'Air')
        H_41.append((h_i - h1)*m_dot/1000)
        T_41.append(T_i - 273.15)

    comp_w   = (h2-h1)*m_dot/1000
    heat_in  = (h3-h2)*m_dot/1000
    turb_w   = (h3-h4)*m_dot/1000
    net_w    = turb_w - comp_w
    bwr      = comp_w/turb_w*100
    eta      = net_w/heat_in*100

    T_states = [T1_C, T2-273.15, T3_C, T4-273.15]
    h_states = [0, (h2-h1)*m_dot/1000, (h3-h1)*m_dot/1000, (h4-h1)*m_dot/1000]
    state_labels = [
        f"1\n({T1_C:.0f}°C, {P1/1000:.0f} kPa)",
        f"2\n({T2-273.15:.0f}°C, {P2/1000:.0f} kPa)",
        f"3\n({T3_C:.0f}°C, {P2/1000:.0f} kPa)",
        f"4\n({T4-273.15:.0f}°C, {P1/1000:.0f} kPa)"
    ]

    states = {
        'labels': ['1 (Comp In)','2 (Comp Out)','3 (Turb In)','4 (Turb Out)'],
        'T_C':    [T1_C, T2-273.15, T3_C, T4-273.15],
        'P_kPa':  [P1/1000, P2/1000, P2/1000, P1/1000],
        'h_kJ':   [h1/1000, h2/1000, h3/1000, h4/1000],
        's_kJ':   [s1/1000, s2/1000, s3/1000, s4/1000],
    }

    traces = {
        '1-2 Compression':  (H_12, T_12, '#f39c12'),
        '2-3 Combustion':   (H_23, T_23, '#e74c3c'),
        '3-4 Expansion':    (H_34, T_34, '#9b59b6'),
        '4-1 Heat Rejection':(H_41, T_41,'#3498db'),
    }
    return traces, h_states, T_states, state_labels, comp_w, heat_in, turb_w, net_w, bwr, eta, states


def biomass_outputs(m_total, moist_pct, ad_eff_pct, htc_conv_pct):
    m_rich  = m_total * moist_pct/100
    m_lean  = m_total * (1 - moist_pct/100)
    m_bio   = m_total * ad_eff_pct/100
    m_char  = m_lean  * htc_conv_pct/100
    m_vol   = m_lean  * (1 - htc_conv_pct/100)
    return m_rich, m_lean, m_bio, m_char, m_vol


# ==========================================
# SIDEBAR — Input Control Panel
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown("*Configure all system parameters below.*")
    st.divider()

    # --- Biomass ---
    st.markdown('<div class="sec-pill">🌿 Biomass & Feed</div>', unsafe_allow_html=True)
    m_biomass   = st.number_input("Total Biomass Flow (kg/s)", 0.1, 500.0, 10.0, 0.5,
                                   help="Total biomass feedstock entering the homogenizer")
    moist_pct   = st.slider("Moisture Content (%)", 10, 90, 50, 1,
                             help="% moisture in feedstock → moisture-rich fraction goes to AD")
    htc_conv    = st.slider("HTC Conversion (%)", 30, 95, 70, 1,
                             help="Dry biomass fraction converted to hydrochar in HTC reactor")

    st.divider()

    # --- AD ---
    st.markdown('<div class="sec-pill">⚗️ Anaerobic Digestion</div>', unsafe_allow_html=True)
    ad_eff      = st.slider("AD Biogas Yield (%)", 20, 90, 60, 1)
    biogas_lhv  = st.number_input("Biogas LHV (MJ/kg)", 10.0, 55.0, 22.0, 0.5)

    st.divider()

    # --- HTC Steam Cycle ---
    st.markdown('<div class="sec-pill">💧 HTC Steam Cycle</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        p_cond   = st.number_input("Cond. P (kPa)",   5.0, 500.0, 10.0, 5.0,
                                    help="Condenser / reactor pressure (State 1)")
    with c2:
        p_boiler = st.number_input("Boiler P (kPa)", 500.0, 20000.0, 2000.0, 100.0)
    t_boiler     = st.number_input("Boiler Outlet T (°C)", 150.0, 700.0, 350.0, 5.0,
                                    help="Superheated steam temperature at turbine inlet (State 3)")
    c3, c4 = st.columns(2)
    with c3:
        pump_eff  = st.number_input("Pump η (%)",   50.0, 99.0, 85.0, 1.0)
    with c4:
        s_turb_eff= st.number_input("S.Turb η (%)", 50.0, 99.0, 85.0, 1.0)

    st.divider()

    # --- Gas Cycle ---
    st.markdown('<div class="sec-pill">💨 Gas Cycle (Brayton)</div>', unsafe_allow_html=True)
    pr_ratio     = st.number_input("Pressure Ratio (rp)", 2.0, 40.0, 12.0, 0.5)
    c5, c6 = st.columns(2)
    with c5:
        t_air_in = st.number_input("Air Inlet T (°C)", -20.0, 50.0, 25.0, 1.0)
    with c6:
        t_turb_in= st.number_input("Turb Inlet T (°C)", 600.0, 1600.0, 1100.0, 25.0)
    c7, c8 = st.columns(2)
    with c7:
        comp_eff  = st.number_input("Comp η (%)",  50.0, 99.0, 88.0, 1.0)
    with c8:
        g_turb_eff= st.number_input("GTurb η (%)", 50.0, 99.0, 90.0, 1.0)

    st.divider()
    m_dot_gas = st.number_input("Gas Mass Flow Rate (kg/s)", 0.1, 500.0, 1.0, 0.5,
                                 help="Air/gas mass flow through the Brayton cycle")
    st.divider()
    analyze_btn = st.button("🚀 Analyze System", type="primary", use_container_width=True)
    st.caption("© 2025 Energhx Research Group · University of Lagos")


# ==========================================
# MAIN HEADER
# ==========================================
col_title, col_logo = st.columns([5,1])
with col_title:
    st.markdown("# 🟣 AD-HTC Fuel-Enhanced Power Gas Cycle")
    st.markdown("**Energhx Research Group** · Faculty of Engineering, University of Lagos")
with col_logo:
    st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📊 Performance Dashboard",
    "🔄 Process Schematic",
    "📋 State Properties"
])


# ==========================================
# ANIMATED SCHEMATIC — TAB 2
# ==========================================
with tab2:
    st.markdown("### AD-HTC Fuel-Enhanced Power Gas Cycle — Animated Process Schematic")
    st.markdown("*Hover over state-point dots for thermodynamic properties after running analysis.*")

    # Build dynamic tooltip labels from session state
    _sp = st.session_state.get('steam_states', None)
    _gp = st.session_state.get('gas_states', None)

    def _fmt(d, key, i, unit, dec=1):
        try: return f"{d[key][i]:.{dec}f} {unit}"
        except: return "—"

    if _sp:
        s_tip = [
            f"T={_fmt(_sp,'T_C',i,'°C')} | P={_fmt(_sp,'P_kPa',i,'kPa',0)}<br>h={_fmt(_sp,'h_kJ',i,'kJ/kg')} | s={_fmt(_sp,'s_kJ',i,'kJ/kg·K',4)}"
            for i in range(4)]
    else:
        s_tip = ["Run analysis to see state properties"] * 4

    if _gp:
        g_tip = [
            f"T={_fmt(_gp,'T_C',i,'°C')} | P={_fmt(_gp,'P_kPa',i,'kPa',0)}<br>h={_fmt(_gp,'h_kJ',i,'kJ/kg')} | s={_fmt(_gp,'s_kJ',i,'kJ/kg·K',4)}"
            for i in range(4)]
    else:
        g_tip = ["Run analysis to see state properties"] * 4

    schematic_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:#07071a; overflow:hidden; }}

/* ── ANIMATIONS ── */
.anim       {{ stroke-dasharray:16 8; animation:dash 2.0s linear infinite; }}
.anim-slow  {{ stroke-dasharray:16 8; animation:dash 3.2s linear infinite; }}
.anim-fast  {{ stroke-dasharray:12 6; animation:dash 1.2s linear infinite; }}
.anim-shaft {{ stroke-dasharray:22 9; animation:dash 3.0s linear infinite; }}
@keyframes dash {{ to {{ stroke-dashoffset:-72; }} }}

/* ── TOOLTIP ── */
.tip {{
  position:absolute; display:none;
  background:#1a0a35; border:1px solid #6c2eb9;
  color:#dcc6f0; font:400 10px 'Segoe UI',Arial,sans-serif;
  padding:7px 10px; border-radius:7px; white-space:nowrap;
  pointer-events:none; z-index:99; line-height:1.6;
  box-shadow: 0 4px 18px rgba(108,46,185,0.5);
}}
.state-dot {{ cursor:pointer; }}
.state-dot:hover + .tip {{ display:block; }}

/* ── FOREIGNOBJECT tooltips via title elements ── */
svg text, svg circle, svg polygon, svg rect, svg line, svg polyline {{
  vector-effect: non-scaling-stroke;
}}
</style>
</head>
<body>
<svg xmlns="http://www.w3.org/2000/svg"
     width="100%" height="720" viewBox="0 0 1200 720"
     style="display:block; max-width:1200px; margin:auto;">
<defs>

<!-- ── GRADIENTS ── -->
<linearGradient id="gPurple" x1="0%" y1="0%" x2="100%" y2="100%">
  <stop offset="0%" stop-color="#1a0635"/><stop offset="100%" stop-color="#2e0d5e"/>
</linearGradient>
<linearGradient id="gBlue" x1="0%" y1="0%" x2="100%" y2="100%">
  <stop offset="0%" stop-color="#071830"/><stop offset="100%" stop-color="#0d2b50"/>
</linearGradient>
<linearGradient id="gGreen" x1="0%" y1="0%" x2="100%" y2="100%">
  <stop offset="0%" stop-color="#071f0a"/><stop offset="100%" stop-color="#0e3515"/>
</linearGradient>
<linearGradient id="gRed" x1="0%" y1="0%" x2="100%" y2="100%">
  <stop offset="0%" stop-color="#2a0505"/><stop offset="100%" stop-color="#4a0d0d"/>
</linearGradient>
<linearGradient id="gShaft" x1="0%" y1="0%" x2="100%" y2="0%">
  <stop offset="0%" stop-color="#8a6200"/><stop offset="50%" stop-color="#ffd700"/>
  <stop offset="100%" stop-color="#8a6200"/>
</linearGradient>

<!-- ── GLOW FILTERS ── -->
<filter id="fG"  x="-35%" y="-35%" width="170%" height="170%">
  <feGaussianBlur stdDeviation="4" result="b"/>
  <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
</filter>
<filter id="fGR" x="-50%" y="-50%" width="200%" height="200%">
  <feGaussianBlur stdDeviation="6" result="b"/>
  <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
</filter>
<filter id="fSh">
  <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#000" flood-opacity="0.55"/>
</filter>

<!-- ── ARROWHEADS ── -->
<marker id="mGn"  markerWidth="9" markerHeight="7" refX="7" refY="3.5" orient="auto"><polygon points="0,0 9,3.5 0,7" fill="#27ae60"/></marker>
<marker id="mTl"  markerWidth="9" markerHeight="7" refX="7" refY="3.5" orient="auto"><polygon points="0,0 9,3.5 0,7" fill="#1abc9c"/></marker>
<marker id="mOr"  markerWidth="9" markerHeight="7" refX="7" refY="3.5" orient="auto"><polygon points="0,0 9,3.5 0,7" fill="#e67e22"/></marker>
<marker id="mBl"  markerWidth="9" markerHeight="7" refX="7" refY="3.5" orient="auto"><polygon points="0,0 9,3.5 0,7" fill="#3498db"/></marker>
<marker id="mRd"  markerWidth="9" markerHeight="7" refX="7" refY="3.5" orient="auto"><polygon points="0,0 9,3.5 0,7" fill="#e74c3c"/></marker>
<marker id="mGr"  markerWidth="9" markerHeight="7" refX="7" refY="3.5" orient="auto"><polygon points="0,0 9,3.5 0,7" fill="#95a5a6"/></marker>
<marker id="mDGr" markerWidth="9" markerHeight="7" refX="7" refY="3.5" orient="auto"><polygon points="0,0 9,3.5 0,7" fill="#7f8c8d"/></marker>
<marker id="mPu"  markerWidth="9" markerHeight="7" refX="7" refY="3.5" orient="auto"><polygon points="0,0 9,3.5 0,7" fill="#8e44ad"/></marker>
<marker id="mO2"  markerWidth="9" markerHeight="7" refX="7" refY="3.5" orient="auto"><polygon points="0,0 9,3.5 0,7" fill="#e67e22"/></marker>

<!-- ── TEXT STYLES via CSS in SVG ── -->
<style>
  .tt  {{ font:700 13px/1.4 'Segoe UI',Arial,sans-serif; fill:#f0e6ff; text-anchor:middle; }}
  .ts  {{ font:400 9.5px   'Segoe UI',Arial,sans-serif; fill:#a98fd4; text-anchor:middle; }}
  .tsec{{ font:700 11.5px  'Segoe UI',Arial,sans-serif; fill:#8e44ad; text-anchor:middle; }}
  .tfl {{ font:400 9.5px   'Segoe UI',Arial,sans-serif; }}
  .tsh {{ font:600 9.5px   'Segoe UI',Arial,sans-serif; fill:#ffd700; text-anchor:middle; }}
  .tsn {{ font:700 10px    'Segoe UI',Arial,sans-serif; fill:#fff;   text-anchor:middle; dominant-baseline:middle; }}
  .tft {{ font:400 8.5px   'Segoe UI',Arial,sans-serif; fill:#4a1a8a; text-anchor:middle; }}
  .tleg{{ font:700 10px    'Segoe UI',Arial,sans-serif; fill:#c39bd3; text-anchor:middle; }}
  .tli {{ font:400 9px     'Segoe UI',Arial,sans-serif; fill:#ccc;   dominant-baseline:middle; }}
  .tpb {{ font:700 9px     'Segoe UI',Arial,sans-serif; fill:#7fb3e8; text-anchor:middle; dominant-baseline:middle; }}
  .tpv {{ font:400 8.5px   'Segoe UI',Arial,sans-serif; fill:#a8d0f5; text-anchor:middle; dominant-baseline:middle; }}
</style>
</defs>

<!-- ══════════════════════════════════════════════════════
     BACKGROUND
══════════════════════════════════════════════════════ -->
<rect width="1200" height="720" fill="#07071a"/>

<!-- ══════════════════════════════════════════════════════
     SECTION BORDERS
     HTC: x=235, y=52, w=440, h=330
     AD:  x=715, y=52, w=240, h=260
══════════════════════════════════════════════════════ -->
<rect x="235" y="52" width="440" height="330"
      fill="none" stroke="#2471a3" stroke-width="1.5"
      stroke-dasharray="8 4" rx="12" opacity="0.65"/>
<text x="455" y="69" class="tsec">HTC Steam Cycle</text>

<rect x="715" y="52" width="248" height="270"
      fill="none" stroke="#1e8449" stroke-width="1.5"
      stroke-dasharray="8 4" rx="12" opacity="0.65"/>
<text x="839" y="69" class="tsec">Anaerobic Digestion</text>

<!-- ══════════════════════════════════════════════════════
     COMPONENT BOXES
     Layout (cx = horizontal centre, cy = vertical centre):
       Homogenizer:  x=28,  y=128, w=148, h=70  → cx=102, cy=163
       Boiler:       x=262, y=88,  w=150, h=72  → cx=337, cy=124
       Steam Turbine (polygon): wide-left at x=428, narrow-right x=570, y=86–170
       Pump (circle): cx=292, cy=290, r=36
       HTC Reactor:  x=338, y=256, w=155, h=80  → cx=415, cy=296
       Condenser:    x=500, y=256, w=162, h=80  → cx=581, cy=296
       AD:           x=722, y=80,  w=228, h=80  → cx=836, cy=120
       Biogas Coll:  x=722, y=216, w=228, h=74  → cx=836, cy=253
       Combustion:   x=722, y=358, w=228, h=78  → cx=836, cy=397
       Compressor (polygon): left cx≈375, y=500–590
       Gas Turbine   (polygon): right cx≈900, y=500–590
══════════════════════════════════════════════════════ -->

<!-- BIOMASS HOMOGENIZER -->
<rect x="28" y="128" width="148" height="70" rx="10"
      fill="url(#gPurple)" stroke="#7b2fbe" stroke-width="2.2" filter="url(#fSh)"/>
<text x="102" y="154" class="tt">Biomass</text>
<text x="102" y="170" class="tt">Homogenizer</text>
<text x="102" y="186" class="ts">Feed Pre-processing</text>

<!-- BOILER -->
<rect x="262" y="88" width="150" height="72" rx="10"
      fill="url(#gBlue)" stroke="#2471a3" stroke-width="2.2" filter="url(#fSh)"/>
<text x="337" y="114" class="tt">Boiler</text>
<text x="337" y="130" class="ts">Heat Exchange</text>
<text x="337" y="147" class="ts">P={p_boiler:.0f} kPa | T={t_boiler:.0f}°C</text>

<!-- STEAM TURBINE — proper engineering trapezoid (wide-left intake, narrow-right exhaust) -->
<polygon points="428,86  570,106  570,154  428,170"
         fill="url(#gBlue)" stroke="#2471a3" stroke-width="2.2" filter="url(#fSh)"/>
<text x="496" y="118" class="tt">Steam</text>
<text x="496" y="134" class="tt">Turbine</text>
<text x="496" y="152" class="ts">η_t = {s_turb_eff:.0f}%</text>

<!-- PUMP — circle -->
<circle cx="292" cy="290" r="36"
        fill="url(#gBlue)" stroke="#2471a3" stroke-width="2.2" filter="url(#fSh)"/>
<!-- pump symbol lines -->
<line x1="276" y1="290" x2="308" y2="290" stroke="#3498db" stroke-width="1.5" opacity="0.5"/>
<line x1="292" y1="274" x2="292" y2="306" stroke="#3498db" stroke-width="1.5" opacity="0.5"/>
<circle cx="292" cy="290" r="36" fill="none" stroke="#3498db" stroke-width="1" opacity="0.35"/>
<text x="292" y="285" class="tt" style="font-size:12px;">Pump</text>
<text x="292" y="301" class="ts">η_p = {pump_eff:.0f}%</text>

<!-- HTC REACTOR -->
<rect x="338" y="256" width="155" height="80" rx="10"
      fill="url(#gBlue)" stroke="#2471a3" stroke-width="2.2" filter="url(#fSh)"/>
<text x="415" y="282" class="tt">HTC Reactor</text>
<text x="415" y="299" class="ts">220–280°C, 20–60 bar</text>
<text x="415" y="315" class="ts">η_conv = {htc_conv}%</text>

<!-- CONDENSER -->
<rect x="500" y="256" width="162" height="80" rx="10"
      fill="url(#gBlue)" stroke="#2471a3" stroke-width="2.2" filter="url(#fSh)"/>
<text x="581" y="282" class="tt">Condenser</text>
<text x="581" y="299" class="ts">Low Pressure</text>
<text x="581" y="315" class="ts">P₁ = {p_cond:.0f} kPa</text>

<!-- AD UNIT -->
<rect x="722" y="80" width="228" height="78" rx="10"
      fill="url(#gGreen)" stroke="#1e8449" stroke-width="2.2" filter="url(#fSh)"/>
<text x="836" y="108" class="tt">Anaerobic</text>
<text x="836" y="124" class="tt">Digestion</text>
<text x="836" y="142" class="ts">AD Yield = {ad_eff}%</text>

<!-- ENHANCED BIOGAS COLLECTOR -->
<rect x="722" y="216" width="228" height="74" rx="10"
      fill="url(#gPurple)" stroke="#7b2fbe" stroke-width="2.2" filter="url(#fSh)"/>
<text x="836" y="242" class="tt">Enhanced</text>
<text x="836" y="258" class="tt">Biogas Collector</text>
<text x="836" y="274" class="ts">LHV = {biogas_lhv:.1f} MJ/kg</text>

<!-- COMBUSTION CHAMBER -->
<rect x="722" y="358" width="228" height="78" rx="10"
      fill="url(#gRed)" stroke="#c0392b" stroke-width="2.5" filter="url(#fGR)"/>
<!-- inner glow -->
<rect x="722" y="358" width="228" height="78" rx="10"
      fill="none" stroke="#e74c3c" stroke-width="1.2" opacity="0.6" filter="url(#fGR)"/>
<text x="836" y="384" class="tt">Biogas</text>
<text x="836" y="400" class="tt">Combustion Chamber</text>
<text x="836" y="418" class="ts">High-Temperature Combustion</text>

<!-- COMPRESSOR — engineering shape: narrow intake (left), wide discharge (right) -->
<polygon points="285,505  430,480  430,580  285,555"
         fill="url(#gPurple)" stroke="#7b2fbe" stroke-width="2.2" filter="url(#fSh)"/>
<text x="358" y="527" class="tt">Compressor</text>
<text x="358" y="544" class="ts">η_c = {comp_eff:.0f}%</text>

<!-- GAS TURBINE — engineering shape: wide intake (left), narrow exhaust (right) -->
<polygon points="920,480  1065,505  1065,555  920,580"
         fill="url(#gPurple)" stroke="#7b2fbe" stroke-width="2.2" filter="url(#fSh)"/>
<text x="990" y="527" class="tt">Gas Turbine</text>
<text x="990" y="544" class="ts">η_t = {g_turb_eff:.0f}%</text>

<!-- ══════════════════════════════════════════════════════
     MECHANICAL SHAFT
══════════════════════════════════════════════════════ -->
<line x1="430" y1="530" x2="920" y2="530"
      stroke="url(#gShaft)" stroke-width="6"
      stroke-dasharray="22 9"
      style="animation:dash 3.0s linear infinite"/>
<text x="675" y="517" class="tsh">⚡  Mechanical Shaft — Net Power Output</text>

<!-- ══════════════════════════════════════════════════════
     FLOW LINES — all orthogonal, labels clear of boxes
══════════════════════════════════════════════════════ -->

<!-- 1. BIOMASS FEEDSTOCK → Homogenizer (enter from left) -->
<line x1="0" y1="163" x2="28" y2="163"
      stroke="#27ae60" stroke-width="2.8" fill="none"
      stroke-dasharray="16 8" class="anim"
      marker-end="url(#mGn)"/>
<text x="2" y="150" class="tfl" fill="#27ae60">Biomass</text>
<text x="2" y="163" class="tfl" fill="#27ae60">Feedstock</text>

<!-- 2. Homogenizer → AD  (moisture-rich)
     Route: right edge of Hom (176,148) → horizontal to x=836 → down to AD top (836,80) -->
<polyline points="176,148  836,148  836,80"
          stroke="#1abc9c" stroke-width="2.8" fill="none"
          stroke-dasharray="16 8" class="anim"
          marker-end="url(#mTl)"/>
<!-- label above the horizontal run, centred between Hom and AD, clear of HTC box top (y=52) -->
<text x="506" y="138" class="tfl" fill="#1abc9c" text-anchor="middle">Moisture-rich Biomass Feedstock → AD</text>

<!-- 3. Homogenizer → HTC Reactor  (moisture-lean)
     Route: bottom of Hom (102,198) → down to y=232 → right to Reactor left (338,232) → down to (338,296) -->
<polyline points="102,198  102,232  338,232  338,296"
          stroke="#e67e22" stroke-width="2.8" fill="none"
          stroke-dasharray="16 8" class="anim-slow"
          marker-end="url(#mOr)"/>
<text x="220" y="225" class="tfl" fill="#e67e22" text-anchor="middle">Moisture-lean → HTC Reactor</text>

<!-- 4. RANKINE STEAM LOOP (blue)
     State 1→2 : Condenser left (500,296) → Pump right (328,296)          -->
<line x1="500" y1="296" x2="328" y2="296"
      stroke="#3498db" stroke-width="2.8" fill="none"
      stroke-dasharray="16 8" class="anim"
      marker-end="url(#mBl)"/>
<text x="414" y="285" class="tfl" fill="#3498db" text-anchor="middle">①→② Condensate</text>

<!--     State 2→3 : Pump top (292,254) → up to y=72 → right to Boiler left (262,124) -->
<polyline points="292,254  292,72  262,72  262,124"
          stroke="#3498db" stroke-width="2.8" fill="none"
          stroke-dasharray="16 8" class="anim"
          marker-end="url(#mBl)"/>
<text x="244" y="105" class="tfl" fill="#3498db" style="font-size:9px;">②→③</text>

<!--     State 3→4 : Boiler right (412,124) → Turbine left (428,124) -->
<line x1="412" y1="124" x2="428" y2="124"
      stroke="#3498db" stroke-width="2.8" fill="none"
      stroke-dasharray="16 8" class="anim"
      marker-end="url(#mBl)"/>

<!--     State 4→1 : Turbine right (570,130) → x=652 → down to y=296 → Condenser right (662,296) -->
<polyline points="570,130  652,130  652,296  662,296"
          stroke="#3498db" stroke-width="2.8" fill="none"
          stroke-dasharray="16 8" class="anim"
          marker-end="url(#mBl)"/>
<text x="662" y="118" class="tfl" fill="#3498db">④→① Steam</text>

<!-- 5. HTC Reactor → Volatile Matters (down, exits bottom)
     From bottom of Reactor (415,336) → down to y=460 -->
<line x1="415" y1="336" x2="415" y2="480"
      stroke="#8e44ad" stroke-width="2.8" fill="none"
      stroke-dasharray="16 8" class="anim-slow"
      marker-end="url(#mPu)"/>
<!-- label to the right of the line -->
<text x="428" y="375" class="tfl" fill="#8e44ad">Volatile Matters</text>
<text x="428" y="390" class="tfl" fill="#8e44ad">&amp; Feedstock Waste</text>

<!-- 6. AD → Enhanced Biogas Collector (down)
     From AD bottom (836,158) → Collector top (836,216) -->
<line x1="836" y1="158" x2="836" y2="216"
      stroke="#e74c3c" stroke-width="2.8" fill="none"
      stroke-dasharray="16 8" class="anim"
      marker-end="url(#mRd)"/>
<text x="850" y="191" class="tfl" fill="#e74c3c">Biogas</text>

<!-- 7. Biogas Collector → Building Distribution (right)
     From Collector right (950,253) → x=1130 -->
<line x1="950" y1="253" x2="1130" y2="253"
      stroke="#e67e22" stroke-width="2.8" fill="none"
      stroke-dasharray="16 8" class="anim-slow"
      marker-end="url(#mO2)"/>
<text x="1040" y="240" class="tfl" fill="#e67e22" text-anchor="middle">Biogas Distribution</text>
<text x="1040" y="254" class="tfl" fill="#e67e22" text-anchor="middle">to Building Envelopes</text>

<!-- 8. Biogas Collector → Combustion Chamber (down)
     From Collector bottom (836,290) → Comb top (836,358) -->
<line x1="836" y1="290" x2="836" y2="358"
      stroke="#e74c3c" stroke-width="2.8" fill="none"
      stroke-dasharray="16 8" class="anim"
      marker-end="url(#mRd)"/>
<text x="850" y="328" class="tfl" fill="#e74c3c">Biogas Fuel</text>

<!-- 9. AIR INLET → Compressor (from below)
     From y=670 up to Compressor bottom-left area -->
<line x1="358" y1="668" x2="358" y2="580"
      stroke="#95a5a6" stroke-width="2.8" fill="none"
      stroke-dasharray="16 8" class="anim-slow"
      marker-end="url(#mGr)"/>
<text x="358" y="690" class="tfl" fill="#95a5a6" text-anchor="middle">Air Inlet</text>
<text x="358" y="704" class="tfl" fill="#95a5a6" text-anchor="middle">{t_air_in:.0f}°C  |  101.3 kPa</text>

<!-- 10. Compressor → Combustion Chamber (compressed air)
     From Compressor right (430,530) → right along y=490 → up to Comb left at y=397 -->
<polyline points="430,505  680,505  680,397  722,397"
          stroke="#95a5a6" stroke-width="2.8" fill="none"
          stroke-dasharray="16 8" class="anim"
          marker-end="url(#mGr)"/>
<text x="555" y="493" class="tfl" fill="#95a5a6" text-anchor="middle">Compressed Air</text>

<!-- 11. Combustion Chamber → Gas Turbine (hot gas)
     From Comb right (950,397) → right to x=1100 → down to y=515 → left to Turbine (1065,515) -->
<polyline points="950,397  1100,397  1100,515  1065,515"
          stroke="#e74c3c" stroke-width="3.2" fill="none"
          stroke-dasharray="12 6" class="anim-fast"
          filter="url(#fG)"
          marker-end="url(#mRd)"/>
<text x="1108" y="380" class="tfl" fill="#e74c3c">Hot</text>
<text x="1108" y="395" class="tfl" fill="#e74c3c">Gas</text>
<text x="1108" y="410" class="tfl" fill="#e74c3c">{t_turb_in:.0f}°C</text>

<!-- 12. Gas Turbine → Exhaust
     From Turbine right (1065,530) → right to edge -->
<line x1="1065" y1="530" x2="1195" y2="530"
      stroke="#7f8c8d" stroke-width="2.8" fill="none"
      stroke-dasharray="16 8" class="anim"
      marker-end="url(#mDGr)"/>
<text x="1130" y="517" class="tfl" fill="#7f8c8d" text-anchor="middle">Exhaust Gases</text>

<!-- ══════════════════════════════════════════════════════
     INPUT PARAMETER BADGES (embedded, clear of flow lines)
══════════════════════════════════════════════════════ -->
<!-- Biomass badge — below homogenizer -->
<rect x="28" y="210" width="148" height="44" rx="6"
      fill="#0d0a1f" stroke="#3d1a6e" stroke-width="1.2"/>
<text x="102" y="226" class="tpb">ṁ_total = {m_biomass:.1f} kg/s</text>
<text x="102" y="240" class="tpv">Moisture: {moist_pct}%  |  AD Yield: {ad_eff}%</text>

<!-- Gas cycle badge — between compressor and turbine, below shaft -->
<rect x="490" y="555" width="300" height="44" rx="6"
      fill="#0d0a1f" stroke="#3d1a6e" stroke-width="1.2"/>
<text x="640" y="571" class="tpb">Gas Cycle: rp={pr_ratio:.1f}  |  TIT={t_turb_in:.0f}°C  |  η_c={comp_eff:.0f}%  |  η_t={g_turb_eff:.0f}%</text>
<text x="640" y="586" class="tpv">ṁ_gas = {m_dot_gas:.1f} kg/s  |  Air Inlet = {t_air_in:.0f}°C</text>

<!-- ══════════════════════════════════════════════════════
     STATE POINT DOTS WITH TOOLTIPS (SVG <title> hover)
══════════════════════════════════════════════════════ -->
<!-- STEAM STATE 1 — left of Condenser -->
<g filter="url(#fG)">
  <circle cx="500" cy="296" r="10" fill="#3498db" stroke="#fff" stroke-width="1.8"/>
  <text x="500" y="296" class="tsn">1</text>
  <title>Steam State 1 (Condenser Out / Pump In)&#10;{s_tip[0]}</title>
</g>
<text x="488" y="313" class="ts" style="fill:#3498db;font-size:8.5px;">Sat. Liq.</text>

<!-- STEAM STATE 2 — top of Pump -->
<g filter="url(#fG)">
  <circle cx="292" cy="254" r="10" fill="#1abc9c" stroke="#fff" stroke-width="1.8"/>
  <text x="292" y="254" class="tsn">2</text>
  <title>Steam State 2 (Pump Out / Boiler In)&#10;{s_tip[1]}</title>
</g>
<text x="307" y="250" class="ts" style="fill:#1abc9c;font-size:8.5px;">Pump Out</text>

<!-- STEAM STATE 3 — Boiler-Turbine junction -->
<g filter="url(#fG)">
  <circle cx="428" cy="124" r="10" fill="#e74c3c" stroke="#fff" stroke-width="1.8"/>
  <text x="428" y="124" class="tsn">3</text>
  <title>Steam State 3 (Turbine Inlet — Superheated Steam)&#10;{s_tip[2]}</title>
</g>
<text x="416" y="112" class="ts" style="fill:#e74c3c;font-size:8.5px;">Sup. Steam</text>

<!-- STEAM STATE 4 — Turbine right exit -->
<g filter="url(#fG)">
  <circle cx="570" cy="130" r="10" fill="#9b59b6" stroke="#fff" stroke-width="1.8"/>
  <text x="570" y="130" class="tsn">4</text>
  <title>Steam State 4 (Turbine Outlet / Condenser In)&#10;{s_tip[3]}</title>
</g>
<text x="582" y="118" class="ts" style="fill:#9b59b6;font-size:8.5px;">Turb. Out</text>

<!-- GAS STATE ① — Compressor inlet -->
<g filter="url(#fG)">
  <circle cx="358" cy="580" r="10" fill="#95a5a6" stroke="#fff" stroke-width="1.8"/>
  <text x="358" y="580" class="tsn">①</text>
  <title>Gas State 1 (Compressor Inlet — Ambient Air)&#10;{g_tip[0]}</title>
</g>

<!-- GAS STATE ② — Compressor outlet / Comb inlet -->
<g filter="url(#fG)">
  <circle cx="680" cy="505" r="10" fill="#f39c12" stroke="#fff" stroke-width="1.8"/>
  <text x="680" y="505" class="tsn">②</text>
  <title>Gas State 2 (Compressor Outlet)&#10;{g_tip[1]}</title>
</g>
<text x="695" y="499" class="ts" style="fill:#f39c12;font-size:8.5px;">Comp. Out</text>

<!-- GAS STATE ③ — Comb outlet / Turbine inlet -->
<g filter="url(#fGR)">
  <circle cx="950" cy="397" r="10" fill="#e74c3c" stroke="#fff" stroke-width="1.8"/>
  <text x="950" y="397" class="tsn">③</text>
  <title>Gas State 3 (Turbine Inlet — TIT)&#10;{g_tip[2]}</title>
</g>
<text x="935" y="385" class="ts" style="fill:#e74c3c;font-size:8.5px;">TIT</text>

<!-- GAS STATE ④ — Turbine outlet -->
<g filter="url(#fG)">
  <circle cx="1065" cy="515" r="10" fill="#8e44ad" stroke="#fff" stroke-width="1.8"/>
  <text x="1065" y="515" class="tsn">④</text>
  <title>Gas State 4 (Turbine Outlet)&#10;{g_tip[3]}</title>
</g>
<text x="1052" y="503" class="ts" style="fill:#8e44ad;font-size:8.5px;">Turb. Out</text>

<!-- ══════════════════════════════════════════════════════
     LEGEND  (bottom-left, well below components)
══════════════════════════════════════════════════════ -->
<rect x="12" y="470" width="210" height="234" rx="8"
      fill="#0e0720" stroke="#3d1a6e" stroke-width="1.5"/>
<text x="117" y="491" class="tleg">Flow Legend</text>

<line x1="22" y1="510" x2="68" y2="510" stroke="#27ae60"  stroke-width="2.6" stroke-dasharray="10 5"/>
<text x="76" y="510" class="tli">Biomass Feedstock</text>

<line x1="22" y1="530" x2="68" y2="530" stroke="#1abc9c"  stroke-width="2.6" stroke-dasharray="10 5"/>
<text x="76" y="530" class="tli">Moisture-rich Flow</text>

<line x1="22" y1="550" x2="68" y2="550" stroke="#e67e22"  stroke-width="2.6" stroke-dasharray="10 5"/>
<text x="76" y="550" class="tli">Moisture-lean Flow</text>

<line x1="22" y1="570" x2="68" y2="570" stroke="#3498db"  stroke-width="2.6" stroke-dasharray="10 5"/>
<text x="76" y="570" class="tli">Steam (HTC Cycle)</text>

<line x1="22" y1="590" x2="68" y2="590" stroke="#e74c3c"  stroke-width="2.6" stroke-dasharray="10 5"/>
<text x="76" y="590" class="tli">Biogas</text>

<line x1="22" y1="610" x2="68" y2="610" stroke="#95a5a6"  stroke-width="2.6" stroke-dasharray="10 5"/>
<text x="76" y="610" class="tli">Air (Brayton Cycle)</text>

<line x1="22" y1="630" x2="68" y2="630" stroke="#7f8c8d"  stroke-width="2.6" stroke-dasharray="10 5"/>
<text x="76" y="630" class="tli">Exhaust Gases</text>

<line x1="22" y1="650" x2="68" y2="650" stroke="#8e44ad"  stroke-width="2.6" stroke-dasharray="10 5"/>
<text x="76" y="650" class="tli">Volatile Matters</text>

<line x1="22" y1="670" x2="68" y2="670" stroke="#ffd700"  stroke-width="5" stroke-dasharray="16 6"/>
<text x="76" y="670" class="tli">Mechanical Shaft</text>

<!-- ══════════════════════════════════════════════════════
     FOOTER
══════════════════════════════════════════════════════ -->
<text x="600" y="713" class="tft">
  AD-HTC Fuel-Enhanced Power Gas Cycle  —  Energhx Research Group, University of Lagos  ©  2025
</text>

</svg>
</body>
</html>"""

    components.html(schematic_html, height=740, scrolling=False)


# ==========================================
# ANALYSIS RESULTS
# ==========================================
if analyze_btn:
    try:
        # Run calculations
        h_s, s_s, sf, hf, sg, hg, pw, bq, tw, cq, nw_s, eff_s, steam_states = \
            calculate_steam_cycle(p_cond, p_boiler, t_boiler, pump_eff, s_turb_eff)

        gas_traces, h_gas, T_gas, g_labels, cw, qi, tw_g, nw_g, bwr, eff_g, gas_states = \
            calculate_gas_cycle_THdot(pr_ratio, t_air_in, t_turb_in, comp_eff, g_turb_eff, m_dot_gas)

        m_rich, m_lean, m_bio, m_char, m_vol = biomass_outputs(m_biomass, moist_pct, ad_eff, htc_conv)
        bio_power = m_bio * biogas_lhv * 1000  # kW

        # Store states for schematic tooltips
        st.session_state['steam_states'] = steam_states
        st.session_state['gas_states']   = gas_states
        st.session_state['bio_lbl']      = {'m_rich': m_rich, 'm_bio': m_bio}

        # ── TAB 1: DASHBOARD ───────────────────────────────────
        with tab1:
            # --- KPIs ---
            st.markdown("### 🔢 Key Performance Indicators")
            k1,k2,k3,k4,k5,k6 = st.columns(6)
            k1.metric("Gas Cycle Net Work",    f"{nw_g:.1f} kJ/kg",     delta=f"BWR: {bwr:.1f}%")
            k2.metric("Gas Cycle η",           f"{max(0,eff_g):.1f} %")
            k3.metric("Steam Cycle Net Work",  f"{nw_s:.1f} kJ/kg")
            k4.metric("Steam Cycle η",         f"{max(0,eff_s):.1f} %")
            k5.metric("Biogas Flow",           f"{m_bio:.2f} kg/s")
            k6.metric("Biogas Power",          f"{bio_power/1000:.2f} MW")

            st.markdown("---")
            b1,b2,b3,b4 = st.columns(4)
            b1.metric("Moisture-rich (→ AD)",  f"{m_rich:.2f} kg/s")
            b2.metric("Moisture-lean (→ HTC)", f"{m_lean:.2f} kg/s")
            b3.metric("Hydrochar Produced",    f"{m_char:.2f} kg/s")
            b4.metric("Volatile Matters",      f"{m_vol:.2f} kg/s")

            st.markdown("---")
            st.markdown("### 📈 Thermodynamic Cycle Diagrams")

            ch1, ch2 = st.columns(2)

            # ── h-s Diagram ─────────────────────────────────────
            with ch1:
                st.markdown("#### HTC Steam Cycle — *h–s* Diagram")
                fig_hs = go.Figure()
                # Saturation dome
                fig_hs.add_trace(go.Scatter(
                    x=list(sf)+list(sg[::-1]), y=list(hf)+list(hg[::-1]),
                    mode='lines', name='Saturation Dome',
                    line=dict(color='rgba(52,152,219,0.4)', width=1.5, dash='dot'),
                    fill='toself', fillcolor='rgba(52,152,219,0.06)'
                ))
                # Cycle processes with labels
                process_colors = ['#f39c12','#27ae60','#e74c3c','#3498db']
                process_names  = ['1→2 Pumping','2→3 Boiling/Superheat','3→4 Expansion','4→1 Condensation']
                for i in range(4):
                    j = (i+1) % 4
                    fig_hs.add_trace(go.Scatter(
                        x=[s_s[i], s_s[j]], y=[h_s[i], h_s[j]],
                        mode='lines', name=process_names[i],
                        line=dict(color=process_colors[i], width=2.5)
                    ))
                # State points
                state_colors = ['#1abc9c','#f39c12','#e74c3c','#9b59b6']
                for i in range(4):
                    fig_hs.add_trace(go.Scatter(
                        x=[s_s[i]], y=[h_s[i]], mode='markers+text',
                        name=steam_states['labels'][i],
                        marker=dict(size=12, color=state_colors[i], line=dict(color='white',width=1.5)),
                        text=[f"  {i+1}: {steam_states['T_C'][i]:.1f}°C"],
                        textposition='middle right',
                        textfont=dict(size=9, color='#c39bd3'),
                        showlegend=False
                    ))
                fig_hs.update_layout(
                    xaxis_title="Entropy, s (kJ/kg·K)",
                    yaxis_title="Enthalpy, h (kJ/kg)",
                    template="plotly_dark",
                    paper_bgcolor="#0a0a14", plot_bgcolor="#0d0d1f",
                    font=dict(color='#c39bd3', size=11),
                    legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0)'),
                    margin=dict(t=20, b=40, l=60, r=20), height=400,
                    xaxis=dict(gridcolor='#1a1a35', zeroline=False),
                    yaxis=dict(gridcolor='#1a1a35', zeroline=False),
                )
                st.plotly_chart(fig_hs, use_container_width=True)

            # ── T-Hdot Diagram ──────────────────────────────────
            with ch2:
                st.markdown("#### Gas Power Cycle — *T–Ḣ* Diagram")
                fig_th = go.Figure()
                process_meta = {
                    '1-2 Compression':   ('Compression (1→2)',   '#f39c12'),
                    '2-3 Combustion':    ('Combustion (2→3)',     '#e74c3c'),
                    '3-4 Expansion':     ('Expansion (3→4)',      '#9b59b6'),
                    '4-1 Heat Rejection':('Heat Rejection (4→1)', '#3498db'),
                }
                for key, (H_pts, T_pts, _) in gas_traces.items():
                    name, color = process_meta[key]
                    fig_th.add_trace(go.Scatter(
                        x=H_pts, y=T_pts, mode='lines',
                        name=name, line=dict(color=color, width=2.5)
                    ))
                # State point markers
                gas_colors = ['#95a5a6','#f39c12','#e74c3c','#8e44ad']
                for i in range(4):
                    fig_th.add_trace(go.Scatter(
                        x=[h_gas[i]], y=[T_gas[i]], mode='markers+text',
                        marker=dict(size=12, color=gas_colors[i], line=dict(color='white',width=1.5)),
                        text=[f"  {['①','②','③','④'][i]}: {T_gas[i]:.0f}°C"],
                        textposition='middle right',
                        textfont=dict(size=9, color='#c39bd3'),
                        showlegend=False
                    ))
                # Annotations for work/heat
                fig_th.add_annotation(
                    x=(h_gas[0]+h_gas[1])/2, y=(T_gas[0]+T_gas[1])/2,
                    text=f"W_comp = {cw:.1f} kJ/kg",
                    showarrow=False, font=dict(size=9,color='#f39c12'),
                    bgcolor='rgba(20,10,40,0.7)'
                )
                fig_th.add_annotation(
                    x=(h_gas[2]+h_gas[3])/2, y=(T_gas[2]+T_gas[3])/2 + 50,
                    text=f"W_turb = {tw_g:.1f} kJ/kg",
                    showarrow=False, font=dict(size=9,color='#9b59b6'),
                    bgcolor='rgba(20,10,40,0.7)'
                )
                fig_th.update_layout(
                    xaxis_title="Total Enthalpy Rate, Ḣ (kJ/kg referenced to State 1)",
                    yaxis_title="Temperature, T (°C)",
                    template="plotly_dark",
                    paper_bgcolor="#0a0a14", plot_bgcolor="#0d0d1f",
                    font=dict(color='#c39bd3', size=11),
                    legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0)'),
                    margin=dict(t=20, b=40, l=60, r=20), height=400,
                    xaxis=dict(gridcolor='#1a1a35', zeroline=False),
                    yaxis=dict(gridcolor='#1a1a35', zeroline=False),
                )
                st.plotly_chart(fig_th, use_container_width=True)

            # ── Energy Bars ─────────────────────────────────────
            st.markdown("---")
            st.markdown("### ⚡ Energy Balance Summary")
            eb1, eb2 = st.columns(2)

            with eb1:
                st.markdown("#### Steam Cycle Energy Flows")
                vals_s = [bq, tw, pw, nw_s, cq]
                lbls_s = ['Boiler Heat In','Turbine Work','Pump Work','Net Output','Condenser Reject']
                clrs_s = ['#3498db','#1abc9c','#f39c12','#27ae60','#e74c3c']
                fig_bs = go.Figure(go.Bar(
                    x=lbls_s, y=vals_s, marker_color=clrs_s,
                    text=[f"{v:.2f}" for v in vals_s], textposition='outside',
                    textfont=dict(color='#c39bd3', size=10)
                ))
                fig_bs.update_layout(
                    yaxis_title="Specific Energy (kJ/kg)",
                    template="plotly_dark", paper_bgcolor="#0a0a14", plot_bgcolor="#0d0d1f",
                    font=dict(color='#c39bd3'), margin=dict(t=10,b=40), height=300,
                    xaxis=dict(gridcolor='#1a1a35'), yaxis=dict(gridcolor='#1a1a35')
                )
                st.plotly_chart(fig_bs, use_container_width=True)

            with eb2:
                st.markdown("#### Gas Cycle Energy Flows")
                vals_g = [qi, tw_g, cw, nw_g, qi-nw_g]
                lbls_g = ['Heat Input','Turbine Work','Comp Work','Net Output','Heat Reject']
                clrs_g = ['#e74c3c','#9b59b6','#f39c12','#27ae60','#3498db']
                fig_bg = go.Figure(go.Bar(
                    x=lbls_g, y=vals_g, marker_color=clrs_g,
                    text=[f"{v:.2f}" for v in vals_g], textposition='outside',
                    textfont=dict(color='#c39bd3', size=10)
                ))
                fig_bg.update_layout(
                    yaxis_title="Specific Energy (kJ/kg)",
                    template="plotly_dark", paper_bgcolor="#0a0a14", plot_bgcolor="#0d0d1f",
                    font=dict(color='#c39bd3'), margin=dict(t=10,b=40), height=300,
                    xaxis=dict(gridcolor='#1a1a35'), yaxis=dict(gridcolor='#1a1a35')
                )
                st.plotly_chart(fig_bg, use_container_width=True)

        # ── TAB 3: STATE PROPERTIES ──────────────────────────
        with tab3:
            st.markdown("### 📋 Thermodynamic State Properties")

            st.markdown("#### 💧 HTC Steam Cycle — State Points")
            df_steam = pd.DataFrame({
                'State':      steam_states['labels'],
                'T (°C)':     [f"{v:.2f}" for v in steam_states['T_C']],
                'P (kPa)':    [f"{v:.1f}"  for v in steam_states['P_kPa']],
                'h (kJ/kg)':  [f"{v:.2f}" for v in steam_states['h_kJ']],
                's (kJ/kg·K)':[f"{v:.4f}" for v in steam_states['s_kJ']],
            })
            st.dataframe(df_steam, use_container_width=True, hide_index=True)

            st.markdown("#### 💨 Gas Power Cycle — State Points")
            df_gas = pd.DataFrame({
                'State':      gas_states['labels'],
                'T (°C)':     [f"{v:.2f}" for v in gas_states['T_C']],
                'P (kPa)':    [f"{v:.2f}"  for v in gas_states['P_kPa']],
                'h (kJ/kg)':  [f"{v:.2f}" for v in gas_states['h_kJ']],
                's (kJ/kg·K)':[f"{v:.4f}" for v in gas_states['s_kJ']],
            })
            st.dataframe(df_gas, use_container_width=True, hide_index=True)

            st.markdown("#### 🌿 Biomass Mass Flow Summary")
            df_bio = pd.DataFrame({
                'Stream':     ['Total Biomass','Moisture-rich → AD','Moisture-lean → HTC',
                               'Biogas from AD','Hydrochar Produced','Volatile Matters/Waste'],
                'Flow (kg/s)':[f"{m_biomass:.3f}", f"{m_rich:.3f}", f"{m_lean:.3f}",
                               f"{m_bio:.3f}", f"{m_char:.3f}", f"{m_vol:.3f}"],
                'Notes':      ['Feed input','To AD unit','To HTC reactor',
                               f'LHV = {biogas_lhv} MJ/kg → {bio_power/1000:.2f} MW potential',
                               'Solid fuel product','Sent to waste treatment']
            })
            st.dataframe(df_bio, use_container_width=True, hide_index=True)

            st.markdown("#### ⚡ Cycle Performance Summary")
            df_perf = pd.DataFrame({
                'Parameter': [
                    'Steam Cycle — Pump Work (kJ/kg)',
                    'Steam Cycle — Turbine Work (kJ/kg)',
                    'Steam Cycle — Net Work (kJ/kg)',
                    'Steam Cycle — Boiler Heat Input (kJ/kg)',
                    'Steam Cycle — Condenser Rejection (kJ/kg)',
                    'Steam Cycle — Thermal Efficiency (%)',
                    'Gas Cycle — Compressor Work (kJ/kg)',
                    'Gas Cycle — Turbine Work (kJ/kg)',
                    'Gas Cycle — Net Work (kJ/kg)',
                    'Gas Cycle — Heat Input (kJ/kg)',
                    'Gas Cycle — Back-Work Ratio (%)',
                    'Gas Cycle — Thermal Efficiency (%)',
                    'Biogas Power Potential (MW)',
                ],
                'Value': [
                    f"{pw:.3f}", f"{tw:.3f}", f"{nw_s:.3f}",
                    f"{bq:.3f}", f"{cq:.3f}", f"{max(0,eff_s):.2f}",
                    f"{cw:.3f}", f"{tw_g:.3f}", f"{nw_g:.3f}",
                    f"{qi:.3f}", f"{bwr:.2f}", f"{max(0,eff_g):.2f}",
                    f"{bio_power/1000:.3f}",
                ]
            })
            st.dataframe(df_perf, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"**Calculation Error:** {e}")
        st.warning("Please check your inputs — common issues: condenser pressure must be below boiler pressure; boiler temperature must exceed saturation temperature at boiler pressure.")

else:
    with tab1:
        st.info("👈 Set parameters in the **Control Panel** then click **🚀 Analyze System** to generate results.")
        st.markdown("""
**This tool computes and visualizes:**
- **HTC Steam (Rankine) Cycle** — Full *h–s* diagram with saturation dome, all 4 state points
- **Gas Power (Brayton) Cycle** — Rigorous *T–Ḣ* diagram using CoolProp Air properties across all 4 processes
- **Biomass mass flow routing** — moisture-rich → AD, moisture-lean → HTC reactor
- **Energy balances** — pump, boiler, turbine, condenser, compressor work and heat
- **Embedded state properties** — hover over state dots on the schematic after analysis
        """)
    with tab3:
        st.info("👈 Run the analysis first to see detailed state property tables.")