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
    </style>
    """, unsafe_allow_html=True)

# --- 2. MOTOR DE CÁLCULO (OPTIMIZADO) ---
def calcular_hurst(ts):
    if len(ts) < 20: return 0.5
    lags = range(2, 15)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(ttl=300)
def analyze_asset(ticker):
    try:
        # Descarga forzando una sola columna para evitar MultiIndex
        data = yf.download(ticker, period='60d', interval='1d', progress=False)
        if data.empty: return None
        
        # Limpieza de columnas (yfinance v0.2.40+ Fix)
        if isinstance(data.columns, pd.MultiIndex):
            df = data.copy()
            df.columns = df.columns.get_level_values(0)
        else:
            df = data

        # Indicadores de Microestructura
        df['Ret'] = df['Close'].pct_change()
        df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
        df['RMF'] = df['Close'] * df['Vol_Proxy'] 
        df['RVOL'] = df['Vol_Proxy'] / df['Vol_Proxy'].rolling(15).mean()
        
        # Z-DIFF (Presión de Flujo)
        periodo_z = 20
        diff = df['Ret'].rolling(periodo_z).sum() - df['RMF'].pct_change().rolling(periodo_z).sum()
        z_series = (diff - diff.rolling(periodo_z).mean()) / (diff.rolling(periodo_z).std() + 1e-10)
        
        # Divergencia de Scalping
        price_mom = df['Close'].diff(3).iloc[-1]
        flow_mom = df['RMF'].diff(3).iloc[-1]
        div_val = 0
        if price_mom > 0 and flow_mom < 0: div_val = -1 # Venta (Precio sube sin volumen real)
        elif price_mom < 0 and flow_mom > 0: div_val = 1 # Compra (Acumulación oculta)

        hurst = calcular_hurst(df['Close'].tail(30).values.flatten())
        
        return {
            'df': df, 'price': float(df['Close'].iloc[-1]), 'z': float(z_series.iloc[-1]), 
            'z_series': z_series, 'hurst': hurst, 'vol': float(df['Ret'].tail(20).std()), 
            'rvol': float(df['RVOL'].iloc[-1]), 'div': div_val
        }
    except Exception as e:
        return None

# --- 3. LISTA DE ACTIVOS ---
ASSETS = ['^GSPC', '^IXIC', '^DJI', 'BTC-USD', 'EURUSD=X', 'GC=F', 'CL=F']

# --- 4. LÓGICA VIX (FIX PARA EVITAR VALUEERROR) ---
try:
    vix_raw = yf.download('^VIX', period='2d', interval='1d', progress=False)['Close']
    # Extraemos valores como float puro (escalares)
    if isinstance(vix_raw, pd.DataFrame):
        vix_series = vix_raw.iloc[:, 0]
    else:
        vix_series = vix_raw
        
    vix_now = float(vix_series.iloc[-1])
    vix_prev = float(vix_series.iloc[-2])
    vix_status = "🔴 RIESGO (VIX ↑)" if vix_now > vix_prev else "🟢 CALMA (VIX ↓)"
except:
    vix_now, vix_prev, vix_status = 20.0, 20.0, "N/A"

# --- 5. INTERFAZ ---
st.title("⚡ JDetector Pro: Sniper Scalper")

with st.sidebar:
    st.header("💰 Risk Manager")
    balance = st.number_input("Balance Cuenta ($)", value=10000)
    risk_pct = st.slider("Riesgo Máximo por Operación (%)", 0.5, 5.0, 1.0)
    st.info(f"VIX Actual: {vix_now:.2f}")

tab1, tab2, tab3, tab4 = st.tabs(["🚀 SCANNER", "🎲 PROBABILIDAD", "🌊 FLUJO RMF", "⚖️ BEER MACRO"])

with tab1:
    st.subheader(f"📡 Escaneo Institucional | {vix_status}")
    if st.button('🔍 INICIAR ESCANEO ADN'):
        results = []
        for t in ASSETS:
            d = analyze_asset(t)
            if d:
                # Lógica de Veredicto
                status = "⚪ NEUTRAL"
                if d['z'] < -1.5 and d['div'] >= 0: status = "🟢 COMPRA"
                elif d['z'] > 1.5 and d['div'] <= 0: status = "🚨 VENTA"
                
                # Filtro de seguridad VIX para índices
                if "^" in t and vix_now > vix_prev and "COMPRA" in status:
                    status = "🟡 COMPRA (Riesgo VIX)"

                results.append([t, f"{d['price']:.2f}", f"{d['z']:.2f}", f"{d['hurst']:.2f}", d['div'], status])
        
        df_res = pd.DataFrame(results, columns=['Activo', 'Precio', 'Z-Diff', 'Hurst', 'Div. Flujo', 'Veredicto'])
        st.table(df_res)

with tab2:
    st.subheader("🎲 Montecarlo & Kelly Criterion")
    target_m = st.selectbox("Seleccionar Activo:", ASSETS)
    dm = analyze_asset(target_m)
    if dm:
        # Simulación
        sims, dias = 1000, 15 
        rets = np.random.normal(dm['df']['Ret'].mean(), dm['vol'], (sims, dias))
        caminos = dm['price'] * (1 + rets).cumprod(axis=1)
        
        # Probabilidad según Z-Diff
        exitos = (caminos[:, -1] > dm['price']).sum() if dm['z'] <= 0 else (caminos[:, -1] < dm['price']).sum()
        prob = (exitos / sims) * 100
        
        # Criterio de Kelly (f = p - q/b)
        kelly = max(0, ((prob/100) * 1.5 - 0.5)) 
        
        c1, c2 = st.columns(2)
        c1.metric("Probabilidad Éxito", f"{prob:.1f}%")
        c2.metric("Sugerencia Kelly", f"{kelly*100:.1f}%")
        
        fig = go.Figure()
        for i in range(10): fig.add_trace(go.Scatter(y=caminos[i], line=dict(width=1), opacity=0.2, showlegend=False))
        fig.add_trace(go.Scatter(y=np.median(caminos, axis=0), line=dict(color="#00ffcc", width=3), name="Tendencia Central"))
        st.plotly_chart(fig.update_layout(template="plotly_dark", height=300), use_container_width=True)

with tab3:
    st.subheader("🌊 Residual Money Flow (RMF)")
    target_f = st.selectbox("Monitor de Flujo:", ASSETS, key="flujo")
    dflow = analyze_asset(target_f)
    if dflow:
        # Detectamos anomalías de volumen institucional
        anomalia = dflow['df']['RMF'].abs() / dflow['df']['RMF'].abs().rolling(20).mean()
        colores = ['#00ffcc' if x > 2.0 else '#333' for x in anomalia]
        
        fig_flow = go.Figure(go.Bar(x=dflow['df'].index, y=dflow['df']['RMF'], marker_color=colores))
        st.plotly_chart(fig_flow.update_layout(template="plotly_dark", title="Barras turquesa = Actividad Institucional Inusual"), use_container_width=True)

with tab4:
    st.subheader("⚖️ BEER Model Equilibrium")
    pair = st.selectbox("Activo Macro:", ['EURUSD=X', '^GSPC'], key="macro")
    try:
        bond = yf.download('^TNX', period='60d', progress=False)['Close']
        price = yf.download(pair, period='60d', progress=False)['Close']
        
        # Sincronización
        df_b = pd.concat([bond, price], axis=1).dropna()
        df_b.columns = ['Bond', 'Price']
        
        # Normalización Z-Score
        z_p = (df_b['Price'] - df_b['Price'].rolling(20).mean()) / df_b['Price'].rolling(20).std()
        z_b = (df_b['Bond'] - df_b['Bond'].rolling(20).mean()) / df_b['Bond'].rolling(20).std()
        
        desv = float(z_p.iloc[-1] - z_b.iloc[-1])
        st.metric("Desviación BEER", f"{desv:.2f}", delta="Sobrevalorado" if desv > 1 else "Infravalorado" if desv < -1 else "Fair Value")
    except:
        st.warning("Datos macro no disponibles temporalmente.")

st.markdown("---")
st.caption("JDetector Sniper v2.0 - Corregido para Pandas Scalar Logic")
