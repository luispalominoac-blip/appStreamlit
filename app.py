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
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================
#  CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="StockSense | Predicción de Inventario",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

USERS = {
    "admin":    "admin123",
    "gerente":  "gerente2026",
    "analista": "analista2026",
}

# ============================================================
#  PALETA DE COLORES (referencia única, evita inconsistencias)
# ============================================================
C_BG          = "#0b1d33"   # fondo general (azul marino profundo)
C_BG_SOFT     = "#11253f"   # fondo secundario (un poco más claro)
C_CARD        = "#16304f"   # fondo de tarjetas
C_CARD_BORDER = "#22436b"   # borde de tarjetas
C_TEXT        = "#eef3fa"   # texto principal (sobre fondo oscuro)
C_TEXT_DIM    = "#9fb3cc"   # texto secundario / labels
C_ACCENT      = "#3b82f6"   # azul de acento (botones, gráficos principales)
C_ACCENT_SOFT = "#60a5fa"
C_DANGER      = "#f87171"
C_DANGER_BG   = "#3a1620"
C_WARNING     = "#fbbf24"
C_WARNING_BG  = "#3a2a10"
C_SUCCESS     = "#4ade80"
C_SUCCESS_BG  = "#0f3322"
C_WHITE_CARD  = "#ffffff"   # para tarjetas claras (sobre fondo oscuro general)
C_DARK_TEXT   = "#0b1d33"   # texto oscuro para usar SOLO sobre tarjetas blancas/claras

# ============================================================
#  ESTILOS CSS
# ============================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    /* Fondo general de toda la app */
    .stApp {{
        background: linear-gradient(180deg, {C_BG} 0%, #0d2138 100%) !important;
    }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1280px; }}

    /* Texto por defecto: claro, porque el fondo es oscuro */
    .stApp, .stApp p, .stApp span, .stApp label,
    .stApp .stMarkdown, .stApp .stText, .stApp li {{
        color: {C_TEXT} !important;
    }}
    h1, h2, h3, h4 {{ color: {C_TEXT} !important; font-weight: 800 !important; letter-spacing: -0.3px; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {C_TEXT_DIM} !important; }}

    /* ───────── Topbar / título de marca ───────── */
    .topbar {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 14px 22px; background: {C_BG_SOFT};
        border-radius: 14px; margin-bottom: 18px;
        border: 1px solid {C_CARD_BORDER};
    }}
    .topbar-brand {{ display: flex; align-items: center; gap: 12px; }}
    .topbar-brand .logo-box {{
        width: 38px; height: 38px; border-radius: 10px;
        background: linear-gradient(135deg, {C_ACCENT} 0%, #1e40af 100%);
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; color: white; font-size: 16px;
    }}
    .topbar-brand .brand-name {{ font-size: 17px; font-weight: 800; color: {C_TEXT}; }}
    .topbar-brand .brand-sub {{ font-size: 11px; color: {C_TEXT_DIM}; letter-spacing: 0.4px; }}
    .topbar-user {{
        font-size: 13px; color: {C_TEXT_DIM}; text-align: right;
    }}
    .topbar-user b {{ color: {C_TEXT}; }}

    /* ───────── Tarjetas de métricas (claras, alto contraste) ───────── */
    .metric-card {{
        background: {C_WHITE_CARD};
        border-radius: 14px;
        padding: 20px 20px;
        text-align: center;
        margin-bottom: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.22);
        border-top: 4px solid #cbd5e1;
    }}
    .metric-card.danger  {{ border-top-color: #dc2626; }}
    .metric-card.warning {{ border-top-color: #d97706; }}
    .metric-card.success {{ border-top-color: #16a34a; }}
    .metric-card.info    {{ border-top-color: {C_ACCENT}; }}
    .metric-card .label {{
        font-size: 11.5px; color: #64748b !important; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.7px;
    }}
    .metric-card .value {{
        font-size: 29px; font-weight: 800; color: {C_DARK_TEXT} !important;
        margin: 7px 0 3px 0; line-height: 1.1;
    }}
    .metric-card .sub {{ font-size: 11.5px; color: #8a93a3 !important; font-weight: 500; }}

    /* ───────── Encabezado de sección ───────── */
    .section-header {{
        background: linear-gradient(120deg, {C_ACCENT} 0%, #1e3a8a 100%);
        color: #ffffff !important; padding: 13px 20px; border-radius: 10px;
        font-size: 15px; font-weight: 700; margin: 24px 0 14px 0;
        letter-spacing: 0.2px;
    }}
    .section-header * {{ color: #ffffff !important; }}

    .subtle-header {{
        font-size: 14px; font-weight: 700; color: {C_TEXT_DIM} !important;
        text-transform: uppercase; letter-spacing: 0.6px;
        margin: 20px 0 10px 0; padding-bottom: 8px;
        border-bottom: 1px solid {C_CARD_BORDER};
    }}

    /* ───────── Alertas ───────── */
    .alert-danger, .alert-warning, .alert-success {{
        padding: 13px 18px; border-radius: 10px; margin: 9px 0;
        font-weight: 600; font-size: 14px;
    }}
    .alert-danger  {{ background: {C_DANGER_BG};  border-left: 4px solid {C_DANGER};  color: #fecaca !important; }}
    .alert-warning {{ background: {C_WARNING_BG}; border-left: 4px solid {C_WARNING}; color: #fde68a !important; }}
    .alert-success {{ background: {C_SUCCESS_BG}; border-left: 4px solid {C_SUCCESS}; color: #bbf7d0 !important; }}
    .alert-danger *, .alert-warning *, .alert-success * {{ color: inherit !important; }}

    /* ───────── Filas de producto ───────── */
    .product-row {{
        background: {C_CARD};
        border-radius: 10px;
        padding: 13px 18px;
        margin-bottom: 7px;
        border: 1px solid {C_CARD_BORDER};
        border-left: 4px solid #475569;
    }}
    .product-row.danger  {{ border-left-color: {C_DANGER}; }}
    .product-row.warning {{ border-left-color: {C_WARNING}; }}
    .product-row.success {{ border-left-color: {C_SUCCESS}; }}
    .product-row b {{ color: {C_TEXT} !important; }}
    .product-row span {{ color: {C_TEXT_DIM} !important; }}

    /* ───────── Tarjeta de calendario de reposición ───────── */
    .repo-card {{
        background: {C_CARD};
        border: 1px solid {C_CARD_BORDER};
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 10px;
        display: flex; align-items: center; justify-content: space-between;
    }}
    .repo-date-box {{
        min-width: 76px; text-align: center; padding: 8px 10px;
        border-radius: 10px; font-weight: 800; margin-right: 16px;
    }}
    .repo-date-box .day {{ font-size: 20px; line-height: 1; }}
    .repo-date-box .mon {{ font-size: 10px; letter-spacing: 0.5px; opacity: 0.85; }}

    /* ───────── Login ───────── */
    .login-card-icon {{ text-align: center; margin-bottom: 16px; }}
    .login-box {{
        background: {C_BG_SOFT}; border: 1px solid {C_CARD_BORDER};
        border-radius: 18px; padding: 36px 34px;
    }}
    .login-title {{
        text-align: center; font-size: 23px; font-weight: 800;
        color: {C_TEXT} !important; margin-bottom: 4px;
    }}
    .login-sub {{
        text-align: center; font-size: 13px; color: {C_TEXT_DIM} !important;
        margin-bottom: 26px;
    }}
    .login-footer-box {{
        text-align: center; margin-top: 18px; padding: 12px;
        background: rgba(255,255,255,0.04); border-radius: 10px;
        font-size: 11.5px; color: {C_TEXT_DIM} !important; line-height: 1.8;
    }}

    /* ───────── Inputs (forzar texto y fondo legibles) ───────── */
    .stTextInput input, .stNumberInput input {{
        background-color: {C_BG_SOFT} !important;
        color: {C_TEXT} !important;
        border: 1px solid {C_CARD_BORDER} !important;
        border-radius: 9px !important;
    }}
    .stSelectbox > div > div {{
        background-color: {C_BG_SOFT} !important;
        border: 1px solid {C_CARD_BORDER} !important;
        border-radius: 9px !important;
        color: {C_TEXT} !important;
    }}
    .stSelectbox div[data-baseweb="select"] span {{ color: {C_TEXT} !important; }}
    /* Lista desplegable del selectbox */
    div[data-baseweb="popover"] li {{
        background-color: {C_BG_SOFT} !important;
        color: {C_TEXT} !important;
    }}
    div[data-baseweb="popover"] li:hover {{
        background-color: {C_CARD} !important;
    }}

    /* ───────── Botones ───────── */
    .stButton > button {{
        border-radius: 9px !important; font-weight: 700 !important;
        padding: 9px 18px !important; border: none !important;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(120deg, {C_ACCENT} 0%, #1e3a8a 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(59,130,246,0.35) !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: {C_CARD} !important;
        color: {C_TEXT} !important;
        border: 1px solid {C_CARD_BORDER} !important;
    }}

    /* ───────── Tabs (navegación principal arriba) ───────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px; background: {C_BG_SOFT}; padding: 6px;
        border-radius: 12px; border: 1px solid {C_CARD_BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px; font-weight: 700; color: {C_TEXT_DIM} !important;
        padding: 10px 22px; font-size: 14px;
    }}
    .stTabs [data-baseweb="tab"] p {{ color: inherit !important; }}
    .stTabs [aria-selected="true"] {{
        background: {C_ACCENT} !important; color: #ffffff !important;
    }}
    .stTabs [aria-selected="true"] p {{ color: #ffffff !important; }}

    /* ───────── Dataframes ───────── */
    [data-testid="stDataFrame"] {{
        border-radius: 12px; overflow: hidden; border: 1px solid {C_CARD_BORDER};
    }}

    /* ───────── Sidebar ───────── */
    section[data-testid="stSidebar"] {{
        background: #081628 !important;
        border-right: 1px solid {C_CARD_BORDER};
    }}
    section[data-testid="stSidebar"] * {{ color: {C_TEXT} !important; }}
    section[data-testid="stSidebar"] hr {{ border-color: {C_CARD_BORDER}; }}
    section[data-testid="stSidebar"] .stButton > button {{
        background: {C_CARD} !important;
        color: {C_TEXT} !important;
        border: 1px solid {C_CARD_BORDER} !important;
        width: 100%;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: {C_DANGER} !important;
        border-color: {C_DANGER} !important;
        color: #ffffff !important;
    }}

    [data-testid="stAlert"] {{ border-radius: 10px; }}

    .footer-credits {{
        text-align: center; font-size: 11px; color: {C_TEXT_DIM} !important;
        margin-top: 46px; padding-top: 16px;
        border-top: 1px solid {C_CARD_BORDER};
    }}

    footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
#  HELPERS DE UI
# ============================================================

def metric_card(label, value, sub="", tone="info"):
    st.markdown(f"""
    <div class="metric-card {tone}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def subtle_header(title):
    st.markdown(f'<div class="subtle-header">{title}</div>', unsafe_allow_html=True)


def footer_credits():
    st.markdown("""
    <div class="footer-credits">
        StockSense &nbsp;·&nbsp; Plataforma de predicción de demanda e inventario<br>
        Desarrollado por Sara Camila Mayorca Parra &amp; Luis Alejandro Palomino Acuña
    </div>
    """, unsafe_allow_html=True)


def configurar_matplotlib_oscuro():
    """Configura matplotlib para que combine con el tema oscuro de la app."""
    plt.rcParams.update({
        'figure.facecolor':  C_CARD,
        'axes.facecolor':    C_CARD,
        'savefig.facecolor': C_CARD,
        'axes.edgecolor':    C_CARD_BORDER,
        'axes.labelcolor':   C_TEXT,
        'xtick.color':       C_TEXT_DIM,
        'ytick.color':       C_TEXT_DIM,
        'text.color':        C_TEXT,
        'axes.titlecolor':   C_TEXT,
        'grid.color':        '#22436b',
        'font.family':       'sans-serif',
    })


configurar_matplotlib_oscuro()


# ============================================================
#  MOTOR DE DATOS Y MODELO
# ============================================================

REQUIRED_COLS_MAP = {
    'fecha':    ['fecha', 'date', 'invoicedate', 'fecha_venta'],
    'producto': ['producto', 'product', 'description', 'item', 'articulo'],
    'cantidad': ['cantidad', 'quantity', 'qty', 'unidades'],
    'precio':   ['precio', 'price', 'unitprice', 'precio_unitario'],
}
OPTIONAL_COLS_MAP = {
    'pais':   ['pais', 'país', 'country', 'tienda', 'store', 'sucursal'],
}


def detectar_columnas(df):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    mapeo, faltantes = {}, []
    for campo, alias in REQUIRED_COLS_MAP.items():
        encontrado = next((cols_lower[a] for a in alias if a in cols_lower), None)
        if encontrado:
            mapeo[campo] = encontrado
        else:
            faltantes.append(campo)
    for campo, alias in OPTIONAL_COLS_MAP.items():
        encontrado = next((cols_lower[a] for a in alias if a in cols_lower), None)
        if encontrado:
            mapeo[campo] = encontrado
    return mapeo, faltantes


@st.cache_data(show_spinner=False)
def procesar_archivo(file_bytes, file_name):
    try:
        if file_name.endswith('.zip'):
            with zipfile.ZipFile(file_bytes) as z:
                fname = [f for f in z.namelist() if f.endswith(('.xlsx', '.xls'))][0]
                with z.open(fname) as f:
                    df = pd.read_excel(f)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_bytes)
        elif file_name.endswith('.csv'):
            df = pd.read_csv(file_bytes)
        else:
            return None, "Formato de archivo no compatible. Usa CSV, Excel o ZIP."
    except Exception as e:
        return None, f"No se pudo leer el archivo. Verifica que no esté dañado. ({e})"
    return df, None


@st.cache_data(show_spinner=False)
def entrenar_sistema(df_raw, col_fecha, col_producto, col_cantidad, col_precio, col_pais):
    df = df_raw.copy()
    df = df.rename(columns={
        col_fecha: 'Fecha', col_producto: 'Producto',
        col_cantidad: 'Cantidad', col_precio: 'Precio',
    })
    df['Pais'] = df[col_pais] if col_pais else 'General'
    if col_pais:
        df = df.rename(columns={col_pais: 'Pais'})

    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.dropna(subset=['Fecha', 'Producto', 'Cantidad', 'Precio'])
    df = df[df['Cantidad'] > 0]
    df = df[df['Precio'] > 0]
    df = df.drop_duplicates()

    if len(df) < 50:
        return None, "El archivo tiene muy pocos registros válidos para generar predicciones confiables."

    df['Mes']      = df['Fecha'].dt.month
    df['Trimestre']= df['Fecha'].dt.quarter
    df['DiaSemana']= df['Fecha'].dt.dayofweek
    df['Semana']   = df['Fecha'].dt.isocalendar().week.astype(int)
    df['Anio']     = df['Fecha'].dt.year

    df_agg = df.groupby(['Anio', 'Semana', 'Producto', 'Pais']).agg(
        Cantidad  = ('Cantidad', 'sum'),
        Precio    = ('Precio', 'mean'),
        Mes       = ('Mes', 'first'),
        Trimestre = ('Trimestre', 'first'),
        DiaSemana = ('DiaSemana', 'mean'),
        Pedidos   = ('Cantidad', 'count'),
    ).reset_index()

    le_pais = LabelEncoder()
    df_agg['Pais_Code'] = le_pais.fit_transform(df_agg['Pais'])
    le_prod = LabelEncoder()
    df_agg['Producto_Code'] = le_prod.fit_transform(df_agg['Producto'])

    features = ['Precio', 'Pais_Code', 'Mes', 'Trimestre',
                'Semana', 'DiaSemana', 'Anio', 'Pedidos', 'Producto_Code']
    X = df_agg[features]
    y = df_agg['Cantidad']

    if len(df_agg) < 30:
        X_train, y_train, X_test, y_test = X, y, X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    modelo = RandomForestRegressor(
        n_estimators=150, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1)
    modelo.fit(X_train_s, y_train)
    y_pred = modelo.predict(X_test_s)

    precision_r2 = max(0, r2_score(y_test, y_pred))
    confiabilidad_pct = round(min(95, max(55, precision_r2 * 100)), 0)

    return {
        'df_raw': df, 'df_agg': df_agg, 'features': features,
        'scaler': scaler, 'le_pais': le_pais, 'le_prod': le_prod,
        'modelo': modelo, 'mae': mean_absolute_error(y_test, y_pred),
        'confiabilidad': confiabilidad_pct, 'fecha_proceso': datetime.now(),
    }, None


def predecir_producto(payload, producto, pais=None, semanas_adelante=1):
    df_agg, le_pais, le_prod = payload['df_agg'], payload['le_pais'], payload['le_prod']
    scaler, modelo, features = payload['scaler'], payload['modelo'], payload['features']

    hist = df_agg[df_agg['Producto'] == producto]
    if len(hist) == 0:
        return None
    if pais is None:
        pais = hist['Pais'].mode().iloc[0]

    precio_prom  = hist['Precio'].mean()
    pedidos_prom = hist['Pedidos'].mean()
    ultima_semana = hist['Semana'].max()
    anio_actual   = hist['Anio'].max()

    semana_obj, anio_obj = ultima_semana + semanas_adelante, anio_actual
    if semana_obj > 52:
        semana_obj -= 52
        anio_obj += 1
    mes_obj  = min(12, max(1, round(semana_obj / 4.33)))
    trim_obj = ((mes_obj - 1) // 3) + 1

    try: pais_code = le_pais.transform([pais])[0]
    except Exception: pais_code = 0
    try: prod_code = le_prod.transform([producto])[0]
    except Exception: prod_code = 0

    entrada = pd.DataFrame([[precio_prom, pais_code, mes_obj, trim_obj,
                             semana_obj, 3, anio_obj, pedidos_prom, prod_code]], columns=features)
    pred = max(0, round(modelo.predict(scaler.transform(entrada))[0]))
    return pred


def calcular_estado_inventario(payload, producto, stock_actual):
    df_agg = payload['df_agg']
    hist = df_agg[df_agg['Producto'] == producto]
    demanda_pred = predecir_producto(payload, producto, semanas_adelante=1)
    if demanda_pred is None:
        return None

    std_hist = hist['Cantidad'].std()
    if pd.isna(std_hist) or std_hist == 0:
        std_hist = hist['Cantidad'].mean() * 0.3

    stock_seguridad  = round(1.65 * std_hist)
    punto_reposicion = demanda_pred + stock_seguridad
    demanda_diaria   = demanda_pred / 7 if demanda_pred > 0 else 0.01
    dias_cobertura   = round(stock_actual / demanda_diaria)

    if stock_actual < stock_seguridad:
        estado, unidades_sugeridas = 'critico', max(0, punto_reposicion + demanda_pred - stock_actual)
    elif stock_actual < punto_reposicion:
        estado, unidades_sugeridas = 'atencion', max(0, punto_reposicion - stock_actual)
    elif stock_actual > punto_reposicion * 1.8:
        estado, unidades_sugeridas = 'exceso', 0
    else:
        estado, unidades_sugeridas = 'optimo', 0

    fecha_sugerida = datetime.now() + timedelta(days=max(1, dias_cobertura - 2))
    return {
        'demanda_predicha': demanda_pred, 'stock_seguridad': stock_seguridad,
        'punto_reposicion': round(punto_reposicion), 'dias_cobertura': dias_cobertura,
        'estado': estado, 'unidades_sugeridas': round(unidades_sugeridas),
        'fecha_sugerida_dt': fecha_sugerida,
        'fecha_sugerida': fecha_sugerida.strftime('%d/%m/%Y'),
    }


# ============================================================
#  LOGIN
# ============================================================

def show_login():
    col1, col2, col3 = st.columns([1, 1.3, 1])
    with col2:
        st.markdown('<br><br>', unsafe_allow_html=True)
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("""
        <div class="login-card-icon"><span style="font-size:48px;">📦</span></div>
        <div class="login-title">StockSense</div>
        <div class="login-sub">Predicción inteligente de demanda e inventario</div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            usuario  = st.text_input("Usuario", placeholder="Ingresa tu usuario")
            password = st.text_input("Contraseña", type="password", placeholder="Ingresa tu contraseña")
            submit   = st.form_submit_button("Ingresar", use_container_width=True, type="primary")

        if submit:
            if usuario in USERS and USERS[usuario] == password:
                st.session_state['logged_in'] = True
                st.session_state['usuario'] = usuario
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")

        st.markdown("""
        <div class="login-footer-box">
            <b>Acceso de demostración</b><br>
            admin / admin123 &nbsp;·&nbsp; gerente / gerente2026 &nbsp;·&nbsp; analista / analista2026
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
#  TOPBAR Y SIDEBAR
# ============================================================

def show_topbar():
    usuario = st.session_state.get('usuario', '')
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-brand">
            <div class="logo-box">S</div>
            <div>
                <div class="brand-name">StockSense</div>
                <div class="brand-sub">INVENTARIO INTELIGENTE</div>
            </div>
        </div>
        <div class="topbar-user">Sesión activa<br><b>{usuario.upper()}</b></div>
    </div>
    """, unsafe_allow_html=True)


def show_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:10px 0 18px;">
            <div style="font-size:34px;">📦</div>
            <div style="font-size:15px; font-weight:800; margin-top:6px;">StockSense</div>
        </div>
        <hr>
        """, unsafe_allow_html=True)

        st.markdown("**Cuenta**")
        st.caption(st.session_state.get('usuario', '').upper())

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Configuración y datos", use_container_width=True):
            st.session_state['_force_config'] = True
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("Cerrar sesión", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ============================================================
#  VISTA: INICIO
# ============================================================

def view_inicio():
    if 'payload' not in st.session_state:
        st.markdown("## Bienvenido a StockSense")
        st.markdown("Tu sistema de predicción de demanda y gestión inteligente de inventario.")
        col1, col2 = st.columns([1.3, 1])
        with col1:
            subtle_header("Cómo empezar")
            st.markdown("""
            **1.** Ve a **Configuración y datos** en el menú lateral y sube tu historial de ventas.

            **2.** El sistema analizará tu información automáticamente.

            **3.** En segundos tendrás tu panel con alertas de inventario, predicciones
            de ventas y reportes listos para compartir.
            """)
            if st.button("Subir mi historial de ventas", type="primary", use_container_width=True):
                st.session_state['_force_config'] = True
                st.rerun()
        with col2:
            subtle_header("Lo que vas a obtener")
            st.markdown("""
            - Qué productos se te van a acabar
            - Qué productos te están sobrando
            - Cuánto vas a vender la próxima semana
            - Sugerencias de cuánto y cuándo reponer
            - Reportes listos para tu equipo o proveedores
            """)
        footer_credits()
        return

    payload = st.session_state['payload']
    df_agg  = payload['df_agg']
    productos = df_agg['Producto'].unique()

    if 'stock_data' not in st.session_state:
        st.session_state['stock_data'] = {}

    productos_muestra = productos[:60] if len(productos) > 60 else productos
    estados = []
    for p in productos_muestra:
        stock_actual = st.session_state['stock_data'].get(p)
        if stock_actual is None:
            hist_prod = df_agg[df_agg['Producto'] == p]['Cantidad']
            stock_actual = int(hist_prod.mean() * 2) if len(hist_prod) else 50
            st.session_state['stock_data'][p] = stock_actual
        info = calcular_estado_inventario(payload, p, stock_actual)
        if info:
            info['producto'], info['stock_actual'] = p, stock_actual
            estados.append(info)

    criticos = [e for e in estados if e['estado'] == 'critico']
    atencion = [e for e in estados if e['estado'] == 'atencion']
    exceso   = [e for e in estados if e['estado'] == 'exceso']
    demanda_total = sum(e['demanda_predicha'] for e in estados)

    st.caption(f"Última actualización: {payload['fecha_proceso'].strftime('%d/%m/%Y %H:%M')}  ·  "
              f"Confiabilidad del modelo: {payload['confiabilidad']:.0f}%")

    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_card("Por agotarse", f"{len(criticos)}", "riesgo crítico", "danger")
    with col2: metric_card("Necesitan atención", f"{len(atencion)}", "bajo nivel ideal", "warning")
    with col3: metric_card("Con exceso", f"{len(exceso)}", "sobrestock", "info")
    with col4: metric_card("Ventas próx. semana", f"{demanda_total:,.0f}", "unidades estimadas", "success")

    section_header("¿Qué productos se te van a acabar?")
    if criticos:
        for e in sorted(criticos, key=lambda x: x['dias_cobertura'])[:8]:
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"""
                <div class="product-row danger">
                    <b>{e['producto'][:60]}</b><br>
                    <span style="font-size:13px;">
                        Se agota en {e['dias_cobertura']} días · Stock actual: {e['stock_actual']} ·
                        Sugerido pedir <b>{e['unidades_sugeridas']} unidades</b> antes del {e['fecha_sugerida']}
                    </span>
                </div>""", unsafe_allow_html=True)
            with col_b:
                if st.button("Ver detalle", key=f"crit_{e['producto']}", use_container_width=True):
                    st.session_state['producto_detalle'] = e['producto']
                    st.session_state['_force_inventario'] = True
                    st.rerun()
    else:
        st.markdown('<div class="alert-success">No tienes productos en riesgo crítico de desabastecimiento.</div>',
                   unsafe_allow_html=True)

    section_header("¿Qué productos te están sobrando?")
    if exceso:
        for e in exceso[:6]:
            st.markdown(f"""
            <div class="product-row warning">
                <b>{e['producto'][:60]}</b><br>
                <span style="font-size:13px;">
                    Stock actual: {e['stock_actual']} unidades · Demanda estimada: {e['demanda_predicha']} unidades/semana
                </span>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-success">No se detecta sobrestock relevante en tu inventario.</div>',
                   unsafe_allow_html=True)

    section_header("¿Cuánto vas a vender la próxima semana?")
    top_demanda = sorted(estados, key=lambda x: -x['demanda_predicha'])[:6]
    df_top = pd.DataFrame([{'Producto': e['producto'][:38], 'Unidades': e['demanda_predicha']}
                           for e in top_demanda])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    bars = ax.barh(df_top['Producto'][::-1], df_top['Unidades'][::-1],
                   color=C_ACCENT_SOFT, edgecolor=C_BG, linewidth=0.5, height=0.6)
    ax.set_title('Productos con mayor demanda estimada', fontweight='bold', fontsize=12, pad=10)
    ax.spines[['top', 'right']].set_visible(False)
    for i, v in enumerate(df_top['Unidades'][::-1]):
        ax.text(v + max(df_top['Unidades'])*0.02, i, f'{v:.0f}', va='center',
               fontsize=9, color=C_TEXT, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    footer_credits()


# ============================================================
#  VISTA: MI INVENTARIO
# ============================================================

def view_inventario():
    payload = st.session_state['payload']
    df_agg  = payload['df_agg']
    productos = sorted(df_agg['Producto'].unique())

    if 'stock_data' not in st.session_state:
        st.session_state['stock_data'] = {}

    preseleccionado = st.session_state.pop('producto_detalle', None)

    subtle_header("Buscar un producto")
    col1, col2 = st.columns([3, 1])
    with col1:
        idx = productos.index(preseleccionado) if preseleccionado in productos else 0
        producto = st.selectbox("Producto", productos, index=idx, label_visibility="collapsed")
    with col2:
        valor_default = st.session_state['stock_data'].get(
            producto, int(df_agg[df_agg['Producto']==producto]['Cantidad'].mean()*2))
        stock_actual = st.number_input("Stock actual", min_value=0, max_value=1000000,
                                       value=valor_default, step=1)
        st.session_state['stock_data'][producto] = stock_actual

    info = calcular_estado_inventario(payload, producto, stock_actual)
    if info is None:
        st.warning("No hay suficiente historial para este producto.")
        return

    estado_labels = {
        'critico':  ('Riesgo crítico de desabastecimiento', 'alert-danger'),
        'atencion': ('Requiere reposición pronto', 'alert-warning'),
        'exceso':   ('Sobrestock detectado', 'alert-warning'),
        'optimo':   ('Nivel óptimo', 'alert-success'),
    }
    texto, clase = estado_labels[info['estado']]
    st.markdown(f'<div class="{clase}">{texto}</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_card("Demanda estimada", f"{info['demanda_predicha']:,}", "unidades próxima semana")
    with col2: metric_card("Stock de seguridad", f"{info['stock_seguridad']:,}", "mínimo recomendado")
    with col3: metric_card("Punto de reposición", f"{info['punto_reposicion']:,}", "umbral para pedir")
    with col4: metric_card("Cobertura actual", f"{info['dias_cobertura']} días", "con el stock de hoy")

    if info['unidades_sugeridas'] > 0:
        st.markdown(f"""
        <div class="alert-warning">
            Te recomendamos pedir <b>{info['unidades_sugeridas']} unidades</b>
            antes del <b>{info['fecha_sugerida']}</b> para evitar quiebre de stock.
        </div>""", unsafe_allow_html=True)

    section_header(f"Histórico de ventas — {producto[:50]}")
    hist = df_agg[df_agg['Producto'] == producto].sort_values(['Anio', 'Semana'])
    if len(hist) > 1:
        fig, ax = plt.subplots(figsize=(12, 4))
        x_labels = [f"S{int(s)}-{int(a)}" for s, a in zip(hist['Semana'], hist['Anio'])]
        ax.plot(range(len(hist)), hist['Cantidad'], color=C_ACCENT_SOFT,
               linewidth=2.2, marker='o', markersize=4, markerfacecolor=C_ACCENT_SOFT)
        ax.fill_between(range(len(hist)), hist['Cantidad'], alpha=0.15, color=C_ACCENT_SOFT)
        ax.axhline(info['demanda_predicha'], color=C_SUCCESS, linestyle='--', linewidth=2,
                  label=f"Predicción próxima semana: {info['demanda_predicha']}")
        step = max(1, len(x_labels)//12)
        ax.set_xticks(range(0, len(x_labels), step))
        ax.set_xticklabels(x_labels[::step], rotation=40, fontsize=8)
        ax.set_ylabel('Unidades vendidas')
        ax.spines[['top', 'right']].set_visible(False)
        leg = ax.legend(facecolor=C_CARD, edgecolor=C_CARD_BORDER, fontsize=9)
        for text in leg.get_texts(): text.set_color(C_TEXT)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Historial insuficiente para mostrar tendencia gráfica.")

    section_header("Todos tus productos")
    tabla = []
    for p in productos[:200]:
        s = st.session_state['stock_data'].get(p)
        if s is None:
            s = int(df_agg[df_agg['Producto']==p]['Cantidad'].mean()*2)
        i = calcular_estado_inventario(payload, p, s)
        if i:
            tabla.append({
                'Producto': p[:50], 'Stock actual': s, 'Demanda estimada': i['demanda_predicha'],
                'Estado': {'critico':'Crítico','atencion':'Atención',
                          'exceso':'Exceso','optimo':'Óptimo'}[i['estado']],
            })
    st.dataframe(pd.DataFrame(tabla), use_container_width=True, height=380)
    footer_credits()


# ============================================================
#  VISTA: PREDICCIONES — Calendario de reposición
# ============================================================

def view_predicciones():
    payload = st.session_state['payload']
    df_agg  = payload['df_agg']

    if 'stock_data' not in st.session_state:
        st.session_state['stock_data'] = {}

    st.markdown("Planifica tus próximos pedidos antes de que falte stock. "
               "Los productos se ordenan por urgencia: primero los que necesitas "
               "reponer más pronto.")

    productos = df_agg['Producto'].unique()
    eventos = []
    for p in productos[:80]:
        s = st.session_state['stock_data'].get(p)
        if s is None:
            s = int(df_agg[df_agg['Producto']==p]['Cantidad'].mean()*2)
            st.session_state['stock_data'][p] = s
        info = calcular_estado_inventario(payload, p, s)
        if info and info['unidades_sugeridas'] > 0:
            info['producto'] = p
            eventos.append(info)

    eventos = sorted(eventos, key=lambda x: x['fecha_sugerida_dt'])

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Pedidos pendientes", f"{len(eventos)}", "productos a reponer", "warning")
    with col2:
        proximos_7d = [e for e in eventos if (e['fecha_sugerida_dt'] - datetime.now()).days <= 7]
        metric_card("Urgentes (7 días)", f"{len(proximos_7d)}", "requieren pedido esta semana", "danger")
    with col3:
        total_unidades = sum(e['unidades_sugeridas'] for e in eventos)
        metric_card("Unidades a pedir", f"{total_unidades:,}", "en total, todos los productos", "info")

    section_header("Calendario de reposición")

    if not eventos:
        st.markdown('<div class="alert-success">No hay pedidos pendientes por ahora. Tu inventario está en buen estado.</div>',
                   unsafe_allow_html=True)
        footer_credits()
        return

    meses_es = {1:'ENE',2:'FEB',3:'MAR',4:'ABR',5:'MAY',6:'JUN',
               7:'JUL',8:'AGO',9:'SEP',10:'OCT',11:'NOV',12:'DIC'}

    for e in eventos[:25]:
        dias_restantes = (e['fecha_sugerida_dt'] - datetime.now()).days
        if dias_restantes <= 3:
            color_box, bg_box = C_DANGER, C_DANGER_BG
        elif dias_restantes <= 10:
            color_box, bg_box = C_WARNING, C_WARNING_BG
        else:
            color_box, bg_box = C_ACCENT_SOFT, C_BG_SOFT

        col_fecha, col_info, col_btn = st.columns([0.9, 4, 1])
        with col_fecha:
            st.markdown(f"""
            <div class="repo-date-box" style="background:{bg_box}; color:{color_box};">
                <div class="day">{e['fecha_sugerida_dt'].day}</div>
                <div class="mon">{meses_es[e['fecha_sugerida_dt'].month]}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_info:
            urgencia = "Urgente" if dias_restantes <= 3 else ("Próximamente" if dias_restantes <= 10 else "Planificado")
            st.markdown(f"""
            <div style="padding-top:4px;">
                <b style="color:{C_TEXT};">{e['producto'][:55]}</b><br>
                <span style="font-size:13px; color:{C_TEXT_DIM};">
                    {urgencia} · Pedir <b style="color:{C_TEXT};">{e['unidades_sugeridas']} unidades</b> ·
                    Cobertura actual: {e['dias_cobertura']} días
                </span>
            </div>
            """, unsafe_allow_html=True)
        with col_btn:
            if st.button("Ver producto", key=f"repo_{e['producto']}", use_container_width=True):
                st.session_state['producto_detalle'] = e['producto']
                st.session_state['_force_inventario'] = True
                st.rerun()
        st.markdown("<div style='height:2px;'></div>", unsafe_allow_html=True)

    footer_credits()


# ============================================================
#  VISTA: REPORTES
# ============================================================

def view_reportes():
    payload = st.session_state['payload']
    df_agg  = payload['df_agg']

    if 'stock_data' not in st.session_state:
        st.session_state['stock_data'] = {}

    st.markdown("Descarga un resumen con el estado de tu inventario, las alertas activas "
               "y las predicciones de demanda, listo para compartir con tu equipo de "
               "compras o tus proveedores.")

    productos = df_agg['Producto'].unique()
    filas = []
    for p in productos[:200]:
        s = st.session_state['stock_data'].get(p)
        if s is None:
            s = int(df_agg[df_agg['Producto']==p]['Cantidad'].mean()*2)
        i = calcular_estado_inventario(payload, p, s)
        if i:
            filas.append({
                'Producto': p, 'Stock actual': s,
                'Demanda estimada (próx. semana)': i['demanda_predicha'],
                'Stock de seguridad': i['stock_seguridad'],
                'Punto de reposición': i['punto_reposicion'],
                'Días de cobertura': i['dias_cobertura'],
                'Estado': {'critico':'Crítico','atencion':'Atención',
                          'exceso':'Exceso','optimo':'Óptimo'}[i['estado']],
                'Unidades sugeridas a pedir': i['unidades_sugeridas'],
            })

    df_reporte = pd.DataFrame(filas)
    section_header("Vista previa del reporte")
    st.dataframe(df_reporte, use_container_width=True, height=340)

    col1, col2 = st.columns(2)
    with col1:
        csv = df_reporte.to_csv(index=False).encode('utf-8-sig')
        st.download_button("Descargar reporte en Excel (CSV)", data=csv,
                          file_name=f"reporte_inventario_{datetime.now().strftime('%Y%m%d')}.csv",
                          mime='text/csv', use_container_width=True, type="primary")
    with col2:
        resumen_txt = generar_resumen_texto(df_reporte)
        st.download_button("Descargar resumen ejecutivo (TXT)", data=resumen_txt.encode('utf-8'),
                          file_name=f"resumen_ejecutivo_{datetime.now().strftime('%Y%m%d')}.txt",
                          mime='text/plain', use_container_width=True)
    footer_credits()


def generar_resumen_texto(df_reporte):
    criticos = df_reporte[df_reporte['Estado'] == 'Crítico']
    exceso   = df_reporte[df_reporte['Estado'] == 'Exceso']
    total_demanda = df_reporte['Demanda estimada (próx. semana)'].sum()
    lineas_criticos = chr(10).join(
        '- ' + str(r['Producto'])[:60] + f" (pedir {r['Unidades sugeridas a pedir']} unidades)"
        for _, r in criticos.head(15).iterrows())
    lineas_exceso = chr(10).join('- ' + str(r['Producto'])[:60] for _, r in exceso.head(15).iterrows())
    return f"""RESUMEN EJECUTIVO DE INVENTARIO
Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')}
{'='*50}

VENTAS ESTIMADAS PROXIMA SEMANA: {total_demanda:,.0f} unidades

PRODUCTOS EN RIESGO CRITICO: {len(criticos)}
{lineas_criticos}

PRODUCTOS CON SOBRESTOCK: {len(exceso)}
{lineas_exceso}

{'='*50}
Reporte generado por StockSense
"""


# ============================================================
#  VISTA: CONFIGURACIÓN
# ============================================================

def view_configuracion():
    st.markdown("## Configuración y datos")
    tab1, tab2 = st.tabs(["Cargar historial de ventas", "Usuarios y acceso"])

    with tab1:
        subtle_header("Sube tu historial de ventas")
        st.markdown("""
        Acepta archivos en formato **Excel (.xlsx)**, **CSV (.csv)** o **ZIP**.

        Tu archivo debe incluir, como mínimo: **fecha** de la venta, **producto**
        vendido, **cantidad** de unidades y **precio** unitario.
        Opcionalmente puedes incluir **país** o **tienda** para predicciones
        más precisas por ubicación. Los nombres de columna pueden variar:
        el sistema los detecta automáticamente.
        """)

        uploaded = st.file_uploader("Arrastra tu archivo aquí",
                                    type=['csv', 'xlsx', 'xls', 'zip'])

        if uploaded:
            df_raw, error = procesar_archivo(io.BytesIO(uploaded.read()), uploaded.name)
            if error:
                st.error(error)
            else:
                mapeo, faltantes = detectar_columnas(df_raw)
                if faltantes:
                    st.warning(f"No pudimos identificar automáticamente: **{', '.join(faltantes)}**. "
                             "Selecciona manualmente qué columna corresponde a cada dato:")
                    cols_disponibles = df_raw.columns.tolist()
                    seleccion_manual = {}
                    for campo in faltantes:
                        seleccion_manual[campo] = st.selectbox(f"¿Cuál columna es '{campo}'?",
                                                               cols_disponibles, key=f"map_{campo}")
                    mapeo.update(seleccion_manual)

                if st.button("Confirmar y analizar mi información", type="primary",
                           use_container_width=True):
                    with st.spinner("Analizando tu historial de ventas..."):
                        payload, err = entrenar_sistema(
                            df_raw, mapeo['fecha'], mapeo['producto'],
                            mapeo['cantidad'], mapeo['precio'], mapeo.get('pais'))
                    if err:
                        st.error(err)
                    else:
                        st.session_state['payload'] = payload
                        st.success("Tu información fue analizada correctamente.")
                        col1, col2, col3 = st.columns(3)
                        with col1: metric_card("Registros procesados", f"{len(payload['df_raw']):,}", "ventas válidas")
                        with col2: metric_card("Productos detectados", f"{payload['df_agg']['Producto'].nunique():,}", "artículos distintos")
                        with col3: metric_card("Confiabilidad", f"{payload['confiabilidad']:.0f}%", "del modelo predictivo")
                        st.info("Ve a la pestaña **Inicio** para ver tu panel completo.")
        elif 'payload' in st.session_state:
            st.success("Ya tienes un historial cargado y analizado. Puedes subir uno nuevo para reemplazarlo.")

    with tab2:
        subtle_header("Usuarios con acceso a esta plataforma")
        df_users = pd.DataFrame([
            {'Usuario': 'admin', 'Rol': 'Administrador'},
            {'Usuario': 'gerente', 'Rol': 'Gerencia / Toma de decisiones'},
            {'Usuario': 'analista', 'Rol': 'Análisis y reportes'},
        ])
        st.dataframe(df_users, use_container_width=True, hide_index=True)
    footer_credits()


# ============================================================
#  MAIN
# ============================================================

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if not st.session_state['logged_in']:
        show_login()
        return

    show_sidebar()
    show_topbar()

    if st.session_state.pop('_force_config', False):
        view_configuracion()
        return

    tiene_datos = 'payload' in st.session_state
    fuerza_inv = st.session_state.pop('_force_inventario', False)

    if tiene_datos:
        labels = ["Inicio", "Mi Inventario", "Predicciones", "Reportes"]
        default_idx = 1 if fuerza_inv else 0
        tabs = st.tabs(labels)
        with tabs[0]:
            view_inicio()
        with tabs[1]:
            view_inventario()
        with tabs[2]:
            view_predicciones()
        with tabs[3]:
            view_reportes()
    else:
        view_inicio()


if __name__ == "__main__":
    main()
