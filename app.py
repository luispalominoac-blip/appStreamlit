import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import time

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="StockSense",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

USERS = {"admin": "admin123", "gerente": "gerente2026", "analista": "analista2026"}

C = {
    "bg":       "#0b1d33",
    "bg2":      "#11253f",
    "card":     "#16304f",
    "border":   "#22436b",
    "text":     "#eef3fa",
    "dim":      "#9fb3cc",
    "accent":   "#3b82f6",
    "accent2":  "#60a5fa",
    "danger":   "#f87171",
    "dbg":      "#3a1620",
    "warning":  "#fbbf24",
    "wbg":      "#3a2a10",
    "success":  "#4ade80",
    "sbg":      "#0f3322",
    "white":    "#ffffff",
    "dark":     "#0b1d33",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {C['bg']} !important; }}
.block-container {{ padding-top: 1rem; padding-bottom: 3rem; max-width: 1280px; }}
.stApp p, .stApp span, .stApp label, .stApp li, .stApp .stMarkdown {{ color: {C['text']} !important; }}
h1, h2, h3, h4 {{ color: {C['text']} !important; font-weight: 800 !important; }}
.stCaption, [data-testid="stCaptionContainer"] p {{ color: {C['dim']} !important; }}

.topbar {{
    display:flex; align-items:center; justify-content:space-between;
    padding:12px 20px; background:{C['bg2']};
    border-radius:12px; margin-bottom:16px;
    border:1px solid {C['border']};
}}
.logo-box {{
    width:36px; height:36px; border-radius:9px;
    background:linear-gradient(135deg,{C['accent']},#1e40af);
    display:inline-flex; align-items:center; justify-content:center;
    font-weight:800; color:white; font-size:15px; margin-right:10px;
}}
.brand {{ font-size:16px; font-weight:800; color:{C['text']}; }}
.brand-sub {{ font-size:10px; color:{C['dim']}; letter-spacing:.5px; }}
.topbar-user {{ font-size:13px; color:{C['dim']}; text-align:right; }}
.topbar-user b {{ color:{C['text']}; }}

.mcard {{
    background:{C['white']}; border-radius:14px; padding:18px 16px;
    text-align:center; margin-bottom:10px;
    box-shadow:0 4px 16px rgba(0,0,0,.25);
    border-top:4px solid #94a3b8;
}}
.mcard.d {{ border-top-color:#dc2626; }}
.mcard.w {{ border-top-color:#d97706; }}
.mcard.s {{ border-top-color:#16a34a; }}
.mcard.i {{ border-top-color:{C['accent']}; }}
.mcard .ml {{ font-size:11px; color:#64748b !important; font-weight:700; text-transform:uppercase; letter-spacing:.7px; }}
.mcard .mv {{ font-size:27px; font-weight:800; color:{C['dark']} !important; margin:6px 0 3px; }}
.mcard .ms {{ font-size:11px; color:#8a93a3 !important; }}

.sh {{
    background:linear-gradient(120deg,{C['accent']},#1e3a8a);
    color:#fff !important; padding:12px 18px; border-radius:10px;
    font-size:14px; font-weight:700; margin:22px 0 12px;
}}
.sh * {{ color:#fff !important; }}

.subh {{
    font-size:12px; font-weight:700; color:{C['dim']} !important;
    text-transform:uppercase; letter-spacing:.6px;
    margin:18px 0 8px; padding-bottom:6px;
    border-bottom:1px solid {C['border']};
}}

.ad {{ padding:12px 16px; border-radius:9px; margin:8px 0; font-weight:600; font-size:14px; }}
.ad.d {{ background:{C['dbg']}; border-left:4px solid {C['danger']}; color:#fecaca !important; }}
.ad.w {{ background:{C['wbg']}; border-left:4px solid {C['warning']}; color:#fde68a !important; }}
.ad.s {{ background:{C['sbg']}; border-left:4px solid {C['success']}; color:#bbf7d0 !important; }}
.ad * {{ color:inherit !important; }}

.prow {{
    background:{C['card']}; border-radius:10px; padding:12px 16px;
    margin-bottom:6px; border:1px solid {C['border']};
    border-left:4px solid #475569;
}}
.prow.d {{ border-left-color:{C['danger']}; }}
.prow.w {{ border-left-color:{C['warning']}; }}
.prow b {{ color:{C['text']} !important; }}
.prow span {{ color:{C['dim']} !important; }}

.stTextInput input, .stNumberInput input {{
    background:{C['bg2']} !important; color:{C['text']} !important;
    border:1px solid {C['border']} !important; border-radius:8px !important;
}}
.stSelectbox > div > div {{
    background:{C['bg2']} !important; border:1px solid {C['border']} !important;
    border-radius:8px !important; color:{C['text']} !important;
}}
.stSelectbox div[data-baseweb="select"] span {{ color:{C['text']} !important; }}
div[data-baseweb="popover"] li {{ background:{C['bg2']} !important; color:{C['text']} !important; }}
div[data-baseweb="popover"] li:hover {{ background:{C['card']} !important; }}

.stButton > button {{
    border-radius:8px !important; font-weight:700 !important;
    padding:9px 16px !important; border:none !important;
    color:#fff !important;
}}
.stButton > button[kind="primary"] {{
    background:linear-gradient(120deg,{C['accent']},#1e3a8a) !important;
    box-shadow:0 4px 12px rgba(59,130,246,.3) !important;
}}
.stButton > button[kind="secondary"] {{
    background:{C['card']} !important;
    border:1px solid {C['border']} !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap:4px; background:{C['bg2']}; padding:5px;
    border-radius:11px; border:1px solid {C['border']};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius:7px; font-weight:700; color:{C['dim']} !important;
    padding:9px 20px; font-size:13.5px;
}}
.stTabs [data-baseweb="tab"] p {{ color:inherit !important; }}
.stTabs [aria-selected="true"] {{ background:{C['accent']} !important; color:#fff !important; }}
.stTabs [aria-selected="true"] p {{ color:#fff !important; }}

[data-testid="stDataFrame"] {{ border-radius:10px; border:1px solid {C['border']}; }}

section[data-testid="stSidebar"] {{ background:#081628 !important; border-right:1px solid {C['border']}; }}
section[data-testid="stSidebar"] * {{ color:{C['text']} !important; }}
section[data-testid="stSidebar"] hr {{ border-color:{C['border']}; }}
section[data-testid="stSidebar"] .stButton > button {{
    background:{C['card']} !important; color:{C['text']} !important;
    border:1px solid {C['border']} !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background:{C['danger']} !important; color:#fff !important;
}}

.fc {{
    text-align:center; font-size:11px; color:{C['dim']} !important;
    margin-top:44px; padding-top:14px; border-top:1px solid {C['border']};
}}
footer {{ visibility:hidden; }}
#MainMenu {{ visibility:hidden; }}
header[data-testid="stHeader"] {{ background:transparent; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def mcard(label, value, sub="", tone="i"):
    st.markdown(f"""<div class="mcard {tone}">
        <div class="ml">{label}</div>
        <div class="mv">{value}</div>
        <div class="ms">{sub}</div>
    </div>""", unsafe_allow_html=True)

def sh(t): st.markdown(f'<div class="sh">{t}</div>', unsafe_allow_html=True)
def subh(t): st.markdown(f'<div class="subh">{t}</div>', unsafe_allow_html=True)
def ad(t, tone="s"): st.markdown(f'<div class="ad {tone}">{t}</div>', unsafe_allow_html=True)
def fc(): st.markdown(f'<div class="fc">StockSense &nbsp;·&nbsp; Desarrollado por Sara Camila Mayorca Parra &amp; Luis Alejandro Palomino Acuña</div>', unsafe_allow_html=True)

def setup_plt():
    plt.rcParams.update({
        'figure.facecolor': C['card'], 'axes.facecolor': C['card'],
        'savefig.facecolor': C['card'], 'axes.edgecolor': C['border'],
        'axes.labelcolor': C['text'], 'xtick.color': C['dim'],
        'ytick.color': C['dim'], 'text.color': C['text'],
        'axes.titlecolor': C['text'], 'grid.color': '#1e3a5f',
    })

setup_plt()


# ── Procesamiento exactamente igual que tu Colab ──────────────────────────────

def cargar_online_retail(raw_bytes):
    """Carga el ZIP del Online Retail exactamente como en el Colab."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
            xlsx_files = [f for f in z.namelist() if f.endswith('.xlsx') or f.endswith('.xls')]
            if not xlsx_files:
                return None, "No se encontró un archivo Excel dentro del ZIP."
            with z.open(xlsx_files[0]) as f:
                df = pd.read_excel(f)
    except zipfile.BadZipFile:
        try:
            df = pd.read_excel(io.BytesIO(raw_bytes))
        except Exception as e:
            return None, f"No se pudo leer el archivo: {e}"
    except Exception as e:
        return None, f"Error al abrir el archivo: {e}"
    return df, None


def limpiar_datos(df):
    """Limpieza igual que Fase 3 del Colab."""
    col_map = {c.lower().strip(): c for c in df.columns}
    col_invoice  = col_map.get('invoiceno',   col_map.get('invoice no', None))
    col_qty      = col_map.get('quantity',     col_map.get('cantidad', None))
    col_price    = col_map.get('unitprice',    col_map.get('unit price', col_map.get('precio', None)))
    col_date     = col_map.get('invoicedate',  col_map.get('invoice date', col_map.get('fecha', None)))
    col_desc     = col_map.get('description',  col_map.get('descripcion', col_map.get('producto', None)))
    col_country  = col_map.get('country',      col_map.get('pais', col_map.get('país', None)))
    col_stock    = col_map.get('stockcode',    col_map.get('stock code', col_map.get('codigo', None)))
    col_customer = col_map.get('customerid',   col_map.get('customer id', None))

    missing = []
    if not col_qty:    missing.append('Quantity / Cantidad')
    if not col_price:  missing.append('UnitPrice / Precio')
    if not col_date:   missing.append('InvoiceDate / Fecha')
    if not col_desc:   missing.append('Description / Producto')
    if missing:
        return None, f"No se encontraron columnas requeridas: {', '.join(missing)}"

    df = df.copy()
    rename = {col_date: 'InvoiceDate', col_qty: 'Quantity', col_price: 'UnitPrice', col_desc: 'Description'}
    if col_country:  rename[col_country]  = 'Country'
    if col_invoice:  rename[col_invoice]  = 'InvoiceNo'
    if col_stock:    rename[col_stock]    = 'StockCode'
    if col_customer: rename[col_customer] = 'CustomerID'
    df = df.rename(columns=rename)

    if 'Country' not in df.columns:  df['Country']   = 'General'
    if 'InvoiceNo' not in df.columns: df['InvoiceNo'] = 'N/A'

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['Quantity']    = pd.to_numeric(df['Quantity'], errors='coerce')
    df['UnitPrice']   = pd.to_numeric(df['UnitPrice'], errors='coerce')

    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df = df.dropna(subset=['InvoiceDate', 'Quantity', 'UnitPrice', 'Description'])
    df = df.drop_duplicates()

    if len(df) < 10:
        return None, "Quedan menos de 10 registros válidos después de la limpieza."
    return df, None


def agregar_y_entrenar(df):
    """Ingeniería de características y entrenamiento igual que Fases 3-4 del Colab."""
    df['Month']      = df['InvoiceDate'].dt.month
    df['Quarter']    = df['InvoiceDate'].dt.quarter
    df['DayOfWeek']  = df['InvoiceDate'].dt.dayofweek
    df['Hour']       = df['InvoiceDate'].dt.hour
    df['Week']       = df['InvoiceDate'].dt.isocalendar().week.astype(int)
    df['Year']       = df['InvoiceDate'].dt.year

    df_agg = df.groupby(['Year', 'Week', 'Description', 'Country']).agg(
        Quantity      = ('Quantity',  'sum'),
        UnitPrice     = ('UnitPrice', 'mean'),
        Month         = ('Month',     'first'),
        Quarter       = ('Quarter',   'first'),
        DayOfWeek     = ('DayOfWeek', 'mean'),
        Hour          = ('Hour',      'mean'),
        Transacciones = ('Quantity',  'count'),
    ).reset_index()

    le_country = LabelEncoder()
    le_product = LabelEncoder()
    df_agg['Country_Code']  = le_country.fit_transform(df_agg['Country'])
    df_agg['Product_Code']  = le_product.fit_transform(df_agg['Description'])

    features = ['UnitPrice', 'Country_Code', 'Month', 'Quarter',
                'Week', 'DayOfWeek', 'Hour', 'Year', 'Transacciones']

    X = df_agg[features]
    y = df_agg['Quantity']

    if len(df_agg) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    modelo.fit(Xtr, y_train)
    y_pred = modelo.predict(Xte)

    r2  = max(0, r2_score(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    confiabilidad = round(min(95, max(55, r2 * 100)), 0)

    return {
        'df':            df,
        'df_agg':        df_agg,
        'features':      features,
        'scaler':        scaler,
        'le_country':    le_country,
        'le_product':    le_product,
        'modelo':        modelo,
        'r2':            r2,
        'mae':           mae,
        'confiabilidad': confiabilidad,
        'ts':            datetime.now(),
    }, None


# ── Predicción e inventario ───────────────────────────────────────────────────

def predecir(p, producto, semanas=1):
    df_agg = p['df_agg']
    hist = df_agg[df_agg['Description'] == producto]
    if len(hist) == 0:
        return 0

    precio  = hist['UnitPrice'].mean()
    pedidos = hist['Transacciones'].mean()
    sem     = int(hist['Week'].max()) + semanas
    anio    = int(hist['Year'].max())
    if sem > 52: sem -= 52; anio += 1
    mes  = min(12, max(1, round(sem / 4.33)))
    trim = ((mes - 1) // 3) + 1

    pais = hist['Country'].mode().iloc[0]
    try: pc = p['le_country'].transform([pais])[0]
    except: pc = 0
    try: prod_c = p['le_product'].transform([producto])[0]
    except: prod_c = 0

    entrada = pd.DataFrame([[precio, pc, mes, trim, sem, 3, anio, pedidos, prod_c]],
                           columns=p['features'])
    pred = max(0, round(p['modelo'].predict(p['scaler'].transform(entrada))[0]))
    return pred


def estado_inv(p, producto, stock):
    hist = p['df_agg'][p['df_agg']['Description'] == producto]
    dem  = predecir(p, producto)
    std  = hist['Quantity'].std()
    if pd.isna(std) or std == 0: std = hist['Quantity'].mean() * 0.3
    seg  = round(1.65 * std)
    repo = dem + seg
    dd   = dem / 7 if dem > 0 else 0.01
    dias = round(stock / dd)

    if stock < seg:
        est, unis = 'critico', max(0, repo + dem - stock)
    elif stock < repo:
        est, unis = 'atencion', max(0, repo - stock)
    elif stock > repo * 1.8:
        est, unis = 'exceso', 0
    else:
        est, unis = 'optimo', 0

    fecha_dt = datetime.now() + timedelta(days=max(1, dias - 2))
    return {
        'dem': dem, 'seg': int(seg), 'repo': int(repo),
        'dias': dias, 'estado': est, 'unis': int(unis),
        'fecha_dt': fecha_dt, 'fecha': fecha_dt.strftime('%d/%m/%Y'),
    }


# ── Login ─────────────────────────────────────────────────────────────────────

def show_login():
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        st.markdown('<br><br>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{C['bg2']};border:1px solid {C['border']};
                    border-radius:16px;padding:34px 30px;">
            <div style="text-align:center;font-size:44px;margin-bottom:14px;">📦</div>
            <div style="text-align:center;font-size:22px;font-weight:800;
                        color:{C['text']};margin-bottom:4px;">StockSense</div>
            <div style="text-align:center;font-size:13px;color:{C['dim']};
                        margin-bottom:24px;">Predicción inteligente de demanda e inventario</div>
        """, unsafe_allow_html=True)

        with st.form("lf"):
            user = st.text_input("Usuario", placeholder="Tu usuario")
            pwd  = st.text_input("Contraseña", type="password", placeholder="Tu contraseña")
            ok   = st.form_submit_button("Ingresar", use_container_width=True, type="primary")

        if ok:
            if user in USERS and USERS[user] == pwd:
                st.session_state.update({'logged_in': True, 'usuario': user})
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")

        st.markdown(f"""
            <div style="text-align:center;margin-top:16px;padding:10px;
                        background:rgba(255,255,255,.04);border-radius:8px;
                        font-size:11.5px;color:{C['dim']};">
                Acceso demo: admin / admin123
            </div>
        </div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def show_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:8px 0 16px;">
            <div style="font-size:30px;">📦</div>
            <div style="font-size:14px;font-weight:800;margin-top:6px;">StockSense</div>
        </div>
        <hr style="border-color:{C['border']};">
        """, unsafe_allow_html=True)
        st.caption(f"Usuario: {st.session_state.get('usuario','').upper()}")
        if 'payload' in st.session_state:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Borrar datos actuales", use_container_width=True):
                del st.session_state['payload']
                st.session_state['stocks'] = {}
                st.rerun()
        st.markdown(f"<br><hr style='border-color:{C['border']};'>", unsafe_allow_html=True)
        if st.button("Cerrar sesión", use_container_width=True):
            st.session_state.clear()
            st.rerun()


# ── Topbar ────────────────────────────────────────────────────────────────────

def show_topbar():
    u = st.session_state.get('usuario', '')
    st.markdown(f"""
    <div class="topbar">
        <div style="display:flex;align-items:center;">
            <div class="logo-box">S</div>
            <div>
                <div class="brand">StockSense</div>
                <div class="brand-sub">INVENTARIO INTELIGENTE</div>
            </div>
        </div>
        <div class="topbar-user">Sesión activa<br><b>{u.upper()}</b></div>
    </div>""", unsafe_allow_html=True)


# ── Página: Inicio ────────────────────────────────────────────────────────────

def page_inicio():
    if 'payload' not in st.session_state:
        st.markdown("## Bienvenido a StockSense")
        st.markdown("""
        <div style="background:{C['bg2']};border:1px solid {C['border']};
                    border-radius:12px;padding:22px 24px;margin:20px 0;">
            <b style="color:{C['text']};">Comienza cargando tu historial de ventas</b><br>
            <span style="color:{C['dim']};font-size:14px;">
            Usa la pestaña <b style="color:{C['text']};">Cargar datos</b> para subir
            el archivo Online Retail ZIP o un Excel / CSV con fecha, producto, cantidad y precio.
            El sistema analizará los datos y entrenará un modelo de Machine Learning automáticamente.
            </span>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("Ir a Cargar datos", use_container_width=True, type="primary"):
                st.session_state['active_tab'] = 1
                st.rerun()
        fc()
        return

    p      = st.session_state['payload']
    df_agg = p['df_agg']
    prods  = df_agg['Description'].unique()

    if 'stocks' not in st.session_state:
        st.session_state['stocks'] = {}

    # Calcular estado de productos (máx 60 para performance)
    estados = []
    for prod in prods[:60]:
        s = st.session_state['stocks'].get(prod)
        if s is None:
            h = df_agg[df_agg['Description'] == prod]['Quantity']
            s = int(h.mean() * 2) if len(h) else 50
            st.session_state['stocks'][prod] = s
        info = estado_inv(p, prod, s)
        info['prod'] = prod
        info['stock'] = s
        estados.append(info)

    criticos = [e for e in estados if e['estado'] == 'critico']
    atencion = [e for e in estados if e['estado'] == 'atencion']
    exceso   = [e for e in estados if e['estado'] == 'exceso']
    dem_tot  = sum(e['dem'] for e in estados)

    st.caption(f"Actualizado: {p['ts'].strftime('%d/%m/%Y %H:%M')}  ·  "
              f"Confiabilidad del modelo: {p['confiabilidad']:.0f}%  ·  "
              f"R²: {p['r2']:.3f}  ·  MAE: {p['mae']:.1f} unidades")

    c1, c2, c3, c4 = st.columns(4)
    with c1: mcard("Por agotarse",       str(len(criticos)), "riesgo crítico",    "d")
    with c2: mcard("Necesitan atención", str(len(atencion)), "bajo nivel ideal",  "w")
    with c3: mcard("Con exceso",         str(len(exceso)),   "sobrestock",        "i")
    with c4: mcard("Ventas próx. semana", f"{dem_tot:,.0f}", "unidades estimadas","s")

    sh("Productos que se van a agotar")
    if criticos:
        for e in sorted(criticos, key=lambda x: x['dias'])[:8]:
            ca, cb = st.columns([4, 1])
            with ca:
                st.markdown(f"""<div class="prow d">
                    <b>{e['prod'][:60]}</b><br>
                    <span style="font-size:13px;">Se agota en {e['dias']} días &nbsp;·&nbsp;
                    Stock actual: {e['stock']} &nbsp;·&nbsp;
                    Pedir <b>{e['unis']} unidades</b> antes del {e['fecha']}</span>
                </div>""", unsafe_allow_html=True)
            with cb:
                if st.button("Ver", key=f"v_{e['prod'][:30]}", use_container_width=True):
                    st.session_state['_det'] = e['prod']
                    st.session_state['active_tab'] = 2  # Inventario
                    st.rerun()
    else:
        ad("No hay productos en riesgo crítico de desabastecimiento.", "s")

    sh("Productos con sobrestock")
    if exceso:
        for e in exceso[:5]:
            st.markdown(f"""<div class="prow w">
                <b>{e['prod'][:60]}</b><br>
                <span style="font-size:13px;">Stock: {e['stock']} &nbsp;·&nbsp;
                Demanda estimada: {e['dem']} unidades/semana</span>
            </div>""", unsafe_allow_html=True)
    else:
        ad("No se detecta sobrestock relevante en tu inventario.", "s")

    sh("Demanda estimada próxima semana")
    top = sorted(estados, key=lambda x: -x['dem'])[:6]
    df_top = pd.DataFrame([{'Producto': e['prod'][:38], 'Unidades': e['dem']} for e in top])
    if df_top['Unidades'].max() > 0:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.barh(df_top['Producto'][::-1], df_top['Unidades'][::-1],
                color=C['accent2'], edgecolor=C['bg'], height=0.6)
        ax.set_title('Top productos por demanda estimada', fontweight='bold', fontsize=11, pad=8)
        ax.spines[['top', 'right']].set_visible(False)
        mx = df_top['Unidades'].max()
        for i, v in enumerate(df_top['Unidades'][::-1]):
            ax.text(v + mx * 0.02, i, f'{v:.0f}', va='center',
                   fontsize=9, color=C['text'], fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    fc()


# ── Página: Cargar datos ─────────────────────────────────────────────────────

def page_carga():
    st.markdown("## Cargar historial de ventas")
    st.markdown(f"""
    <div style="background:{C['bg2']};border:1px solid {C['border']};
                border-radius:12px;padding:18px 20px;margin-bottom:18px;">
        <b style="color:{C['text']};">Formatos aceptados</b><br>
        <span style="color:{C['dim']};font-size:13.5px;">
        El sistema acepta el archivo <b style="color:{C['text']};">Online Retail ZIP</b>
        descargado de UCI o Kaggle, o cualquier Excel / CSV con columnas de fecha,
        producto, cantidad y precio. Las columnas se detectan automáticamente.
        </span>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Selecciona tu archivo",
        type=['zip', 'xlsx', 'xls', 'csv'],
        help="El archivo Online Retail ZIP funciona directamente sin configuración."
    )

    if uploaded is not None:
        if st.session_state.get('_last_file') != uploaded.name:
            st.session_state['_raw']       = uploaded.getvalue()
            st.session_state['_last_file'] = uploaded.name

        raw = st.session_state['_raw']
        fname = uploaded.name.lower()

        if fname.endswith('.zip'):
            df_raw, err = cargar_online_retail(raw)
        elif fname.endswith(('.xlsx', '.xls')):
            try:
                df_raw, err = pd.read_excel(io.BytesIO(raw)), None
            except Exception as e:
                df_raw, err = None, str(e)
        else:
            try:
                df_raw, err = pd.read_csv(io.BytesIO(raw)), None
            except Exception as e:
                df_raw, err = None, str(e)

        if err:
            st.error(f"No se pudo leer el archivo: {err}")
            fc()
            return

        st.success(f"Archivo leído: **{len(df_raw):,} filas** · {len(df_raw.columns)} columnas detectadas")

        with st.expander("Ver columnas del archivo"):
            st.write(list(df_raw.columns))

        if st.button("Analizar y entrenar modelo", type="primary", use_container_width=True):
            with st.status("Analizando datos, por favor espera...", expanded=True) as status:
                st.write("Limpiando registros inválidos...")
                df_limpio, err2 = limpiar_datos(df_raw)
                if err2:
                    st.error(err2)
                    status.update(label="Error en la limpieza", state="error")
                    return
                st.write(f"Datos válidos: {len(df_limpio):,} filas")
                st.write("Agregando por semana y entrenando modelo Random Forest...")
                payload, err3 = agregar_y_entrenar(df_limpio)
                if err3:
                    st.error(err3)
                    status.update(label="Error en el entrenamiento", state="error")
                    return
                status.update(label="Modelo entrenado correctamente", state="complete")

            st.session_state['payload'] = payload
            st.session_state['stocks']  = {}
            st.success("Modelo entrenado correctamente.")
            st.info("Redirigiendo al panel de inicio...")
            time.sleep(1)
            st.session_state['active_tab'] = 0
            st.rerun()

    elif 'payload' in st.session_state:
        st.success("Ya tienes datos cargados. Puedes subir un archivo nuevo para reemplazarlos.")
    fc()


# ── Página: Inventario ────────────────────────────────────────────────────────

def page_inventario():
    if 'payload' not in st.session_state:
        st.warning("Primero carga un archivo de ventas en la pestaña 'Cargar datos'.")
        if st.button("Ir a Cargar datos"):
            st.session_state['active_tab'] = 1
            st.rerun()
        return

    p      = st.session_state['payload']
    df_agg = p['df_agg']
    prods  = sorted(df_agg['Description'].unique())

    if 'stocks' not in st.session_state:
        st.session_state['stocks'] = {}

    presel = st.session_state.pop('_det', None)

    subh("Buscar producto")
    c1, c2 = st.columns([3, 1])
    with c1:
        idx  = prods.index(presel) if presel in prods else 0
        prod = st.selectbox("Producto", prods, index=idx, label_visibility="collapsed")
    with c2:
        default = st.session_state['stocks'].get(
            prod, int(df_agg[df_agg['Description'] == prod]['Quantity'].mean() * 2))
        stock = st.number_input("Stock actual", min_value=0, max_value=1000000,
                                value=default, step=1)
        st.session_state['stocks'][prod] = stock

    info = estado_inv(p, prod, stock)

    labels = {
        'critico':  ("Riesgo crítico de desabastecimiento", "d"),
        'atencion': ("Requiere reposición pronto",          "w"),
        'exceso':   ("Sobrestock detectado",                "w"),
        'optimo':   ("Nivel óptimo",                        "s"),
    }
    txt, tone = labels[info['estado']]
    ad(txt, tone)

    c1, c2, c3, c4 = st.columns(4)
    with c1: mcard("Demanda estimada",     f"{info['dem']:,}",  "unidades próx. semana")
    with c2: mcard("Stock mínimo",         f"{info['seg']:,}",  "stock de seguridad")
    with c3: mcard("Punto de reposición",  f"{info['repo']:,}", "reponer cuando llegue a este nivel")
    with c4: mcard("Cobertura actual",     f"{info['dias']} días", "con el stock de hoy")

    if info['unis'] > 0:
        ad(f"Te recomendamos pedir <b>{info['unis']} unidades</b> antes del <b>{info['fecha']}</b>.", "w")

    sh(f"Historial de ventas — {prod[:50]}")
    hist = df_agg[df_agg['Description'] == prod].sort_values(['Year', 'Week'])
    if len(hist) > 1:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(range(len(hist)), hist['Quantity'],
               color=C['accent2'], linewidth=2.2, marker='o', markersize=4)
        ax.fill_between(range(len(hist)), hist['Quantity'], alpha=0.12, color=C['accent2'])
        ax.axhline(info['dem'], color=C['success'], linestyle='--', linewidth=2,
                  label=f"Predicción próx. semana: {info['dem']} unidades")
        labels_x = [f"S{int(s)}-{int(a)}" for s, a in zip(hist['Week'], hist['Year'])]
        step = max(1, len(labels_x) // 12)
        ax.set_xticks(range(0, len(labels_x), step))
        ax.set_xticklabels(labels_x[::step], rotation=40, fontsize=8)
        ax.set_ylabel('Unidades')
        ax.spines[['top', 'right']].set_visible(False)
        leg = ax.legend(facecolor=C['card'], edgecolor=C['border'], fontsize=9)
        for t in leg.get_texts(): t.set_color(C['text'])
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Historial insuficiente para mostrar el gráfico de tendencia.")

    sh("Todos los productos")
    tabla = []
    for pp in prods[:200]:
        s = st.session_state['stocks'].get(pp,
            int(df_agg[df_agg['Description'] == pp]['Quantity'].mean() * 2))
        i = estado_inv(p, pp, s)
        tabla.append({
            'Producto':          pp[:50],
            'Stock actual':      s,
            'Demanda estimada':  i['dem'],
            'Estado':            {'critico': 'Crítico', 'atencion': 'Atención',
                                  'exceso': 'Exceso', 'optimo': 'Óptimo'}[i['estado']],
        })
    st.dataframe(pd.DataFrame(tabla), use_container_width=True, height=360)
    fc()


# ── Página: Predicciones / Calendario ────────────────────────────────────────

def page_predicciones():
    if 'payload' not in st.session_state:
        st.warning("Primero carga un archivo de ventas en la pestaña 'Cargar datos'.")
        if st.button("Ir a Cargar datos"):
            st.session_state['active_tab'] = 1
            st.rerun()
        return

    p      = st.session_state['payload']
    df_agg = p['df_agg']

    if 'stocks' not in st.session_state:
        st.session_state['stocks'] = {}

    st.markdown(f"<span style='color:{C['dim']};font-size:14px;'>"
               "Planifica tus próximos pedidos por orden de urgencia. "
               "Los productos están ordenados por la fecha en que necesitan reposición.</span>",
               unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    prods = df_agg['Description'].unique()
    eventos = []
    for prod in prods[:80]:
        s = st.session_state['stocks'].get(prod)
        if s is None:
            h = df_agg[df_agg['Description'] == prod]['Quantity']
            s = int(h.mean() * 2) if len(h) else 50
            st.session_state['stocks'][prod] = s
        info = estado_inv(p, prod, s)
        if info['unis'] > 0:
            info['prod'] = prod
            eventos.append(info)

    eventos = sorted(eventos, key=lambda x: x['fecha_dt'])
    prox7   = [e for e in eventos if (e['fecha_dt'] - datetime.now()).days <= 7]
    tot_u   = sum(e['unis'] for e in eventos)

    c1, c2, c3 = st.columns(3)
    with c1: mcard("Pedidos pendientes",    str(len(eventos)), "productos a reponer",        "w")
    with c2: mcard("Urgentes esta semana",  str(len(prox7)),   "requieren pedido pronto",    "d")
    with c3: mcard("Unidades totales",      f"{tot_u:,}",      "a pedir en todos los productos", "i")

    sh("Calendario de reposición")
    MESES = {1:'ENE',2:'FEB',3:'MAR',4:'ABR',5:'MAY',6:'JUN',
             7:'JUL',8:'AGO',9:'SEP',10:'OCT',11:'NOV',12:'DIC'}

    if not eventos:
        ad("No hay pedidos pendientes. Tu inventario está en buen estado.", "s")
    else:
        for e in eventos[:25]:
            dias = (e['fecha_dt'] - datetime.now()).days
            if dias <= 3:    col, bg = C['danger'],  C['dbg']
            elif dias <= 10: col, bg = C['warning'], C['wbg']
            else:            col, bg = C['accent2'], C['bg2']
            urgencia = "Urgente" if dias <= 3 else ("Próximamente" if dias <= 10 else "Planificado")

            ca, cb, cc = st.columns([0.85, 4, 1])
            with ca:
                st.markdown(f"""
                <div style="text-align:center;padding:10px 6px;background:{bg};
                            border-radius:10px;color:{col};font-weight:800;margin-top:4px;">
                    <div style="font-size:20px;line-height:1;">{e['fecha_dt'].day}</div>
                    <div style="font-size:10px;letter-spacing:.5px;">{MESES[e['fecha_dt'].month]}</div>
                </div>""", unsafe_allow_html=True)
            with cb:
                st.markdown(f"""
                <div style="padding-top:6px;">
                    <b style="color:{C['text']};">{e['prod'][:55]}</b><br>
                    <span style="font-size:13px;color:{C['dim']};">
                        {urgencia} &nbsp;·&nbsp;
                        Pedir <b style="color:{C['text']};">{e['unis']} unidades</b>
                        &nbsp;·&nbsp; Cobertura actual: {e['dias']} días
                    </span>
                </div>""", unsafe_allow_html=True)
            with cc:
                if st.button("Ver", key=f"r_{e['prod'][:30]}", use_container_width=True):
                    st.session_state['_det'] = e['prod']
                    st.session_state['active_tab'] = 2  # Inventario
                    st.rerun()
            st.markdown("<div style='height:3px;'></div>", unsafe_allow_html=True)
    fc()


# ── Página: Reportes ──────────────────────────────────────────────────────────

def page_reportes():
    if 'payload' not in st.session_state:
        st.warning("Primero carga un archivo de ventas en la pestaña 'Cargar datos'.")
        if st.button("Ir a Cargar datos"):
            st.session_state['active_tab'] = 1
            st.rerun()
        return

    p      = st.session_state['payload']
    df_agg = p['df_agg']

    if 'stocks' not in st.session_state:
        st.session_state['stocks'] = {}

    st.markdown(f"<span style='color:{C['dim']};font-size:14px;'>"
               "Exporta el estado completo de tu inventario con predicciones y alertas, "
               "listo para compartir con tu equipo o proveedores.</span>",
               unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    prods = df_agg['Description'].unique()
    filas = []
    for prod in prods[:200]:
        s = st.session_state['stocks'].get(prod,
            int(df_agg[df_agg['Description'] == prod]['Quantity'].mean() * 2))
        i = estado_inv(p, prod, s)
        filas.append({
            'Producto':                    prod,
            'Stock actual':                s,
            'Demanda (próx. semana)':      i['dem'],
            'Stock mínimo':                i['seg'],
            'Punto de reposición':         i['repo'],
            'Días de cobertura':           i['dias'],
            'Estado':                      {'critico': 'Crítico', 'atencion': 'Atención',
                                            'exceso': 'Exceso', 'optimo': 'Óptimo'}[i['estado']],
            'Unidades a pedir':            i['unis'],
            'Pedir antes del':             i['fecha'],
        })

    df_rep = pd.DataFrame(filas)
    sh("Vista previa del reporte")
    st.dataframe(df_rep, use_container_width=True, height=340)

    c1, c2 = st.columns(2)
    with c1:
        csv = df_rep.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "Descargar reporte (CSV / Excel)",
            csv,
            file_name=f"reporte_inventario_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
            use_container_width=True,
            type="primary"
        )
    with c2:
        criticos = df_rep[df_rep['Estado'] == 'Crítico']
        exceso   = df_rep[df_rep['Estado'] == 'Exceso']
        resumen  = f"""RESUMEN EJECUTIVO DE INVENTARIO
Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')}
{'='*50}
Ventas estimadas próxima semana: {df_rep['Demanda (próx. semana)'].sum():,.0f} unidades

PRODUCTOS EN RIESGO CRÍTICO: {len(criticos)}
{chr(10).join('- ' + str(r['Producto'])[:60] + f" → pedir {r['Unidades a pedir']} unidades antes del {r['Pedir antes del']}" for _, r in criticos.head(15).iterrows())}

PRODUCTOS CON SOBRESTOCK: {len(exceso)}
{chr(10).join('- ' + str(r['Producto'])[:60] for _, r in exceso.head(15).iterrows())}

{'='*50}
Generado por StockSense"""
        st.download_button(
            "Descargar resumen ejecutivo (TXT)",
            resumen.encode('utf-8'),
            file_name=f"resumen_{datetime.now().strftime('%Y%m%d')}.txt",
            mime='text/plain',
            use_container_width=True
        )
    fc()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not st.session_state.get('logged_in'):
        show_login()
        return

    show_sidebar()
    show_topbar()

    # Definir pestañas siempre visibles
    tabs = st.tabs(["Inicio", "Cargar datos", "Mi Inventario", "Predicciones", "Reportes"])

    # Obtener índice de pestaña activa desde session_state (por defecto 0)
    active_tab = st.session_state.get('active_tab', 0)

    with tabs[0]:
        page_inicio()
    with tabs[1]:
        page_carga()
    with tabs[2]:
        page_inventario()
    with tabs[3]:
        page_predicciones()
    with tabs[4]:
        page_reportes()

    # Establecer el tab activo visualmente (parche para forzar el cambio)
    # Como Streamlit no permite cambiar el tab programáticamente de forma directa,
    # usamos un pequeño script JS para seleccionar la pestaña deseada.
    if st.session_state.get('active_tab') is not None:
        js = f"""
        <script>
            var tabs = window.parent.document.querySelectorAll('.stTabs button[data-baseweb="tab"]');
            if (tabs.length > {st.session_state.active_tab}) {{
                tabs[{st.session_state.active_tab}].click();
            }}
        </script>
        """
        st.components.v1.html(js, height=0, width=0)


if __name__ == "__main__":
    main()
