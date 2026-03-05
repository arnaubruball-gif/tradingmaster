import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
import requests
from datetime import datetime

# --- 1. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(page_title="JDetector Pro - Scalping Edition", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; }
    .stMetric { background-color: #111111; padding: 15px; border-radius: 10px; border: 1px solid #222; }
    .metric-box { background-color: #0e1117; padding: 15px; border-radius: 8px; border: 1px solid #30363d; text-align: center; }
    .signal-buy { background-color: #00ffcc; color: black; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; }
    .signal-sell { background-color: #ff4b4b; color: white; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MOTOR DE CÁLCULO (OPTIMIZADO SCALPING) ---
def calcular_hurst(ts):
    if len(ts) < 20: return 0.5
    lags = range(2, 15)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(ttl=300) # Cache más corto para scalping
def analyze_asset(ticker):
    try:
        # Descarga rápida de 60 días para cálculos de corto plazo
        df = yf.download(ticker, period='60d', interval='1d', progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Indicadores Micro
        df['Ret'] = df['Close'].pct_change()
        df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
        df['RMF'] = df['Close'] * df['Vol_Proxy'] # Money Flow Residual
        df['RVOL'] = df['Vol_Proxy'] / df['Vol_Proxy'].rolling(15).mean()
        
        # Z-DIFF (Presión de Flujo)
        periodo_z = 20
        diff = df['Ret'].rolling(periodo_z).sum() - df['RMF'].pct_change().rolling(periodo_z).sum()
        z_series = (diff - diff.rolling(periodo_z).mean()) / (diff.rolling(periodo_z).std() + 1e-10)
        
        # DIVERGENCIA DE SCALPING (Precio vs Flujo Real)
        # Si el precio sube pero el flujo baja = Distribución (Venta)
        price_mom = df['Close'].diff(3)
        flow_mom = df['RMF'].diff(3)
        div_val = 0 # Neutral
        if price_mom.iloc[-1] > 0 and flow_mom.iloc[-1] < 0: div_val = -1 # Trampa alcista
        elif price_mom.iloc[-1] < 0 and flow_mom.iloc[-1] > 0: div_val = 1 # Acumulación oculta

        hurst = calcular_hurst(df['Close'].tail(30).values.flatten())
        
        return {
            'df': df, 'price': float(df['Close'].iloc[-1]), 'z': z_series.iloc[-1], 
            'z_series': z_series, 'hurst': hurst, 'vol': df['Ret'].tail(20).std(), 
            'rvol': df['RVOL'].iloc[-1], 'div': div_val
        }
    except: return None

# --- 3. LISTA DE ACTIVOS (ÍNDICES Y FILTROS) ---
ASSETS = ['^GSPC', '^IXIC', '^DJI', 'BTC-USD', 'EURUSD=X', 'GC=F', 'CL=F']

# --- 4. INTERFAZ ---
st.title("⚡ JDetector Pro: Scalping & Institutional Edge")

# Sidebar para Gestión de Riesgo
with st.sidebar:
    st.header("💰 Risk Manager")
    balance = st.number_input("Balance Cuenta ($)", value=10000)
    risk_per_trade = st.slider("Riesgo por Operación (%)", 0.5, 5.0, 1.0)
    st.markdown("---")
    st.write("El sistema utiliza el **Criterio de Kelly** para ajustar el tamaño según la probabilidad de éxito.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 SNIPER SCAN", "🎲 PROBABILIDAD", "📊 FLUJO (RMF)", "🏛️ MACRO FILTERS", "⚖️ BEER MODEL"])

# --- LÓGICA VIX (FILTRO DE PÁNICO) ---
try:
    vix_df = yf.download('^VIX', period='2d', interval='1d', progress=False)['Close']
    vix_now = vix_df.iloc[-1]
    vix_prev = vix_df.iloc[-2]
    vix_status = "🔴 ALERTA: Miedo Subiendo" if vix_now > vix_prev else "🟢 CALMA: VIX Bajando"
except:
    vix_now, vix_status = 20, "N/A"

with tab1:
    st.subheader(f"📡 Escaneo en Tiempo Real | VIX: {vix_now:.2f} ({vix_status})")
    if st.button('🔍 INICIAR ESCANEO ADN'):
        results = []
        for t in ASSETS:
            d = analyze_asset(t)
            if d:
                # Lógica de Veredicto combinando ADN + Divergencia
                status = "⚪ NEUTRAL"
                if d['z'] < -1.5 and d['div'] >= 0: status = "🟢 COMPRA"
                elif d['z'] > 1.5 and d['div'] <= 0: status = "🚨 VENTA"
                
                # Filtro VIX para Índices
                if "^" in t and vix_now > vix_prev and "COMPRA" in status:
                    status = "🟡 COMPRA (Riesgo VIX)"

                results.append([t, f"{d['price']:.2f}", f"{d['z']:.2f}", f"{d['hurst']:.2f}", d['div'], status])
        
        res_df = pd.DataFrame(results, columns=['Activo', 'Precio', 'Z-Diff', 'Hurst', 'Div. Flujo', 'Veredicto'])
        st.dataframe(res_df.style.applymap(lambda x: 'color: #00ffcc' if 'COMPRA' in str(x) else ('color: #ff4b4b' if 'VENTA' in str(x) else ''), subset=['Veredicto']), use_container_width=True)

with tab2:
    st.subheader("🎲 Análisis de Probabilidad (Montecarlo 15d)")
    target_m = st.selectbox("Seleccionar Activo:", ASSETS, key="mc_s")
    dm = analyze_asset(target_m)
    if dm:
        sims, dias = 1000, 15 
        rets = np.random.normal(dm['df']['Ret'].mean(), dm['vol'], (sims, dias))
        caminos = dm['price'] * (1 + rets).cumprod(axis=1)
        
        exitos = (caminos[:, -1] > dm['price']).sum() if dm['z'] <= 0 else (caminos[:, -1] < dm['price']).sum()
        prob = (exitos / sims) * 100
        
        # CÁLCULO KELLY PARA SCALPING
        k_perc = max(0, ((prob/100) * 1.5 - 0.5)) # Asumiendo R:R 1:1.5
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Probabilidad de Éxito", f"{prob:.1f}%")
            st.metric("Sugerencia Kelly", f"{k_perc*100:.1f}% del capital")
        with c2:
            st.write(f"**Lotes Sugeridos:** {(balance * k_perc) / (dm['price'] * 0.01):.2f}")
            st.write(f"**Stop Loss (ATR):** {dm['price'] * (1 - dm['vol']):.2f}")

        fig_m = go.Figure()
        for i in range(15): fig_m.add_trace(go.Scatter(y=caminos[i], line=dict(width=1), opacity=0.1, showlegend=False))
        fig_m.add_trace(go.Scatter(y=np.percentile(caminos, 50, axis=0), line=dict(color="#00ffcc", width=3), name="Mediana"))
        st.plotly_chart(fig_m.update_layout(template="plotly_dark", height=350), use_container_width=True)

with tab3:
    st.subheader("🌊 Shadow Money Flow (Institutional Activity)")
    target_b = st.selectbox("Analizar RMF:", ASSETS, key="b_s")
    db = analyze_asset(target_b)
    if db:
        df_b = db['df'].copy()
        # Detectar Anomalías de Volumen (Smart Money)
        df_b['Anom'] = df_b['RMF'].abs() / df_b['RMF'].abs().rolling(20).mean()
        colors = ['#00ffcc' if x > 2.0 else '#333' for x in df_b['Anom']]
        
        st.plotly_chart(go.Figure(data=[go.Bar(x=df_b.index, y=df_b['RMF'], marker_color=colors)]).update_layout(template="plotly_dark", title="Money Flow Residual (Barras verdes = Actividad Institucional)"), use_container_width=True)

with tab4:
    st.subheader("🏛️ Institutional COT & Vol-Monitor")
    # Monitor de Eficiencia de Kaufman
    target_v = st.selectbox("Efficiency Ratio:", ASSETS, key="v_s")
    dv = analyze_asset(target_v)
    if dv:
        change = abs(dv['df']['Close'] - dv['df']['Close'].shift(10))
        volat = abs(dv['df']['Close'] - dv['df']['Close'].shift(1)).rolling(10).sum()
        er = (change / (volat + 1e-10)).iloc[-1]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("RVOL (Volumen Relativo)", f"{dv['rvol']:.2f}x")
        m2.metric("Efficiency Ratio", f"{er:.2f}")
        m3.metric("Hurst Exponent", f"{dv['hurst']:.2f}")
        
        if er > 0.6: st.success("🚀 Tendencia Altamente Eficiente")
        else: st.warning("🌀 Mercado en Rango / Ruido")

with tab5:
    st.subheader("⚖️ BEER Model (Forex/Index Equilibrium)")
    pair_beer = st.selectbox("Par/Activo:", ['EURUSD=X', 'GBPUSD=X', '^GSPC'], key="sb_beer")
    try:
        bond_ref = yf.download('^TNX', period='60d', interval='1d', progress=False)['Close']
        price_ref = yf.download(pair_beer, period='60d', interval='1d', progress=False)['Close']
        
        df_beer = pd.concat([bond_ref, price_ref], axis=1).dropna()
        df_beer.columns = ['Bond', 'Price']
        df_beer['Bond_N'] = (df_beer['Bond'] - df_beer['Bond'].rolling(20).mean()) / df_beer['Bond'].rolling(20).std()
        df_beer['Price_N'] = (df_beer['Price'] - df_beer['Price'].rolling(20).mean()) / df_beer['Price'].rolling(20).std()
        
        actual_desv = df_beer['Price_N'].iloc[-1] - df_beer['Bond_N'].iloc[-1]
        
        st.metric("Desviación BEER", f"{actual_desv:.2f}", 
                  delta="SOBREVALORADO" if actual_desv > 1.2 else "INFRAVALORADO" if actual_desv < -1.2 else "FAIR VALUE")
        
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=df_beer.index, y=df_beer['Price_N'], name="Precio (Norm)"))
        fig_p.add_trace(go.Scatter(x=df_beer.index, y=df_beer['Bond_N'], name="Yield 10Y (Norm)", line=dict(dash='dot')))
        st.plotly_chart(fig_p.update_layout(template="plotly_dark"), use_container_width=True)
    except: st.error("Error cargando datos macro.")

st.markdown("---")
st.caption("JDetector Pro | Use bajo su propio riesgo. El scalping requiere ejecución inmediata.")
