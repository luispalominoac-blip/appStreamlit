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

C_BG          = "#0b1d33"
C_BG_SOFT     = "#11253f"
C_CARD        = "#16304f"
C_CARD_BORDER = "#22436b"
C_TEXT        = "#eef3fa"
C_TEXT_DIM    = "#9fb3cc"
C_ACCENT      = "#3b82f6"
C_ACCENT_SOFT = "#60a5fa"
C_DANGER      = "#f87171"
C_DANGER_BG   = "#3a1620"
C_WARNING     = "#fbbf24"
C_WARNING_BG  = "#3a2a10"
C_SUCCESS     = "#4ade80"
C_SUCCESS_BG  = "#0f3322"
C_WHITE_CARD  = "#ffffff"
C_DARK_TEXT   = "#0b1d33"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{
        background: linear-gradient(180deg, {C_BG} 0%, #0d2138 100%) !important;
    }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1280px; }}
    .stApp, .stApp p, .stApp span, .stApp label,
    .stApp .stMarkdown, .stApp .stText, .stApp li {{
        color: {C_TEXT} !important;
    }}
    h1, h2, h3, h4 {{ color: {C_TEXT} !important; font-weight: 800 !important; }}
    .stCaption {{ color: {C_TEXT_DIM} !important; }}
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
    .topbar-user {{ font-size: 13px; color: {C_TEXT_DIM}; text-align: right; }}
    .topbar-user b {{ color: {C_TEXT}; }}
    .metric-card {{
        background: {C_WHITE_CARD}; border-radius: 14px; padding: 20px;
        text-align: center; margin-bottom: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.22); border-top: 4px solid #cbd5e1;
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
        margin: 7px 0 3px 0;
    }}
    .metric-card .sub {{ font-size: 11.5px; color: #8a93a3 !important; }}
    .section-header {{
        background: linear-gradient(120deg, {C_ACCENT} 0%, #1e3a8a 100%);
        color: #ffffff !important; padding: 13px 20px; border-radius: 10px;
        font-size: 15px; font-weight: 700; margin: 24px 0 14px 0;
    }}
    .section-header * {{ color: #ffffff !important; }}
    .subtle-header {{
        font-size: 13px; font-weight: 700; color: {C_TEXT_DIM} !important;
        text-transform: uppercase; letter-spacing: 0.6px;
        margin: 20px 0 10px 0; padding-bottom: 8px;
        border-bottom: 1px solid {C_CARD_BORDER};
    }}
    .alert-danger, .alert-warning, .alert-success {{
        padding: 13px 18px; border-radius: 10px; margin: 9px 0;
        font-weight: 600; font-size: 14px;
    }}
    .alert-danger  {{ background: {C_DANGER_BG};  border-left: 4px solid {C_DANGER};  color: #fecaca !important; }}
    .alert-warning {{ background: {C_WARNING_BG}; border-left: 4px solid {C_WARNING}; color: #fde68a !important; }}
    .alert-success {{ background: {C_SUCCESS_BG}; border-left: 4px solid {C_SUCCESS}; color: #bbf7d0 !important; }}
    .alert-danger *, .alert-warning *, .alert-success * {{ color: inherit !important; }}
    .product-row {{
        background: {C_CARD}; border-radius: 10px; padding: 13px 18px;
        margin-bottom: 7px; border: 1px solid {C_CARD_BORDER}; border-left: 4px solid #475569;
    }}
    .product-row.danger  {{ border-left-color: {C_DANGER}; }}
    .product-row.warning {{ border-left-color: {C_WARNING}; }}
    .product-row b {{ color: {C_TEXT} !important; }}
    .product-row span {{ color: {C_TEXT_DIM} !important; }}
    .login-title {{
        text-align: center; font-size: 23px; font-weight: 800;
        color: {C_TEXT} !important; margin-bottom: 4px;
    }}
    .login-sub {{
        text-align: center; font-size: 13px; color: {C_TEXT_DIM} !important; margin-bottom: 26px;
    }}
    .login-footer-box {{
        text-align: center; margin-top: 18px; padding: 12px;
        background: rgba(255,255,255,0.04); border-radius: 10px;
        font-size: 11.5px; color: {C_TEXT_DIM} !important; line-height: 1.8;
    }}
    .stTextInput input, .stNumberInput input {{
        background-color: {C_BG_SOFT} !important;
        color: {C_TEXT} !important;
        border: 1px solid {C_CARD_BORDER} !important;
        border-radius: 9px !important;
    }}
    .stSelectbox > div > div {{
        background-color: {C_BG_SOFT} !important;
        border: 1px solid {C_CARD_BORDER} !important;
        border-radius: 9px !important; color: {C_TEXT} !important;
    }}
    .stSelectbox div[data-baseweb="select"] span {{ color: {C_TEXT} !important; }}
    div[data-baseweb="popover"] li {{
        background-color: {C_BG_SOFT} !important; color: {C_TEXT} !important;
    }}
    div[data-baseweb="popover"] li:hover {{ background-color: {C_CARD} !important; }}
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
        background: {C_CARD} !important; color: {C_TEXT} !important;
        border: 1px solid {C_CARD_BORDER} !important;
    }}
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
    [data-testid="stDataFrame"] {{
        border-radius: 12px; overflow: hidden; border: 1px solid {C_CARD_BORDER};
    }}
    section[data-testid="stSidebar"] {{
        background: #081628 !important; border-right: 1px solid {C_CARD_BORDER};
    }}
    section[data-testid="stSidebar"] * {{ color: {C_TEXT} !important; }}
    section[data-testid="stSidebar"] hr {{ border-color: {C_CARD_BORDER}; }}
    section[data-testid="stSidebar"] .stButton > button {{
        background: {C_CARD} !important; color: {C_TEXT} !important;
        border: 1px solid {C_CARD_BORDER} !important; width: 100%;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: {C_DANGER} !important; color: #ffffff !important;
    }}
    .footer-credits {{
        text-align: center; font-size: 11px; color: {C_TEXT_DIM} !important;
        margin-top: 46px; padding-top: 16px; border-top: 1px solid {C_CARD_BORDER};
    }}
    footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers UI ────────────────────────────────────────────────

def metric_card(label, value, sub="", tone="info"):
    st.markdown(f"""
    <div class="metric-card {tone}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def section_header(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def subtle_header(title):
    st.markdown(f'<div class="subtle-header">{title}</div>', unsafe_allow_html=True)

def footer_credits():
    st.markdown("""
    <div class="footer-credits">
        StockSense &nbsp;·&nbsp; Plataforma de predicción de demanda e inventario<br>
        Desarrollado por Sara Camila Mayorca Parra &amp; Luis Alejandro Palomino Acuña
    </div>""", unsafe_allow_html=True)

def setup_plot():
    plt.rcParams.update({
        'figure.facecolor': C_CARD, 'axes.facecolor': C_CARD,
        'savefig.facecolor': C_CARD, 'axes.edgecolor': C_CARD_BORDER,
        'axes.labelcolor': C_TEXT, 'xtick.color': C_TEXT_DIM,
        'ytick.color': C_TEXT_DIM, 'text.color': C_TEXT,
        'axes.titlecolor': C_TEXT, 'grid.color': '#22436b',
    })

setup_plot()


# ── Detección de columnas ─────────────────────────────────────

COLS_MAP = {
    'fecha':    ['fecha', 'date', 'invoicedate', 'fecha_venta', 'invoice date'],
    'producto': ['producto', 'product', 'description', 'item', 'articulo', 'nombre'],
    'cantidad': ['cantidad', 'quantity', 'qty', 'unidades', 'units'],
    'precio':   ['precio', 'price', 'unitprice', 'precio_unitario', 'unit price', 'valor'],
}
COLS_OPT = {
    'pais': ['pais', 'país', 'country', 'tienda', 'store', 'sucursal', 'region'],
}

def detectar_columnas(df):
    cols_lower = {c.lower().strip().replace(' ', ''): c for c in df.columns}
    mapeo, faltantes = {}, []
    for campo, alias in COLS_MAP.items():
        alias_clean = [a.replace(' ', '') for a in alias]
        encontrado = next((cols_lower[a] for a in alias_clean if a in cols_lower), None)
        if encontrado:
            mapeo[campo] = encontrado
        else:
            faltantes.append(campo)
    for campo, alias in COLS_OPT.items():
        alias_clean = [a.replace(' ', '') for a in alias]
        encontrado = next((cols_lower[a] for a in alias_clean if a in cols_lower), None)
        if encontrado:
            mapeo[campo] = encontrado
    return mapeo, faltantes


# ── Motor: carga y entrenamiento ─────────────────────────────

def cargar_archivo(file_bytes, file_name):
    nombre = file_name.lower()
    try:
        if nombre.endswith('.zip'):
            with zipfile.ZipFile(file_bytes) as z:
                xlsx = [f for f in z.namelist() if f.endswith(('.xlsx','.xls'))][0]
                with z.open(xlsx) as f:
                    return pd.read_excel(f), None
        elif nombre.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_bytes), None
        elif nombre.endswith('.csv'):
            return pd.read_csv(file_bytes), None
        else:
            return None, "Formato no soportado. Usa .xlsx, .csv o .zip"
    except Exception as e:
        return None, f"Error al leer el archivo: {e}"


def entrenar(df_raw, col_fecha, col_producto, col_cantidad, col_precio, col_pais):
    df = df_raw[[col_fecha, col_producto, col_cantidad, col_precio]].copy()
    df.columns = ['Fecha', 'Producto', 'Cantidad', 'Precio']

    if col_pais and col_pais in df_raw.columns:
        df['Pais'] = df_raw[col_pais].values
    else:
        df['Pais'] = 'General'

    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df['Cantidad'] = pd.to_numeric(df['Cantidad'], errors='coerce')
    df['Precio']   = pd.to_numeric(df['Precio'], errors='coerce')
    df = df.dropna(subset=['Fecha', 'Producto', 'Cantidad', 'Precio'])
    df = df[df['Cantidad'] > 0]
    df = df[df['Precio'] > 0]
    df = df.drop_duplicates()

    if len(df) < 10:
        return None, "No hay suficientes registros válidos (mínimo 10) para generar predicciones."

    df['Mes']       = df['Fecha'].dt.month
    df['Trimestre'] = df['Fecha'].dt.quarter
    df['DiaSemana'] = df['Fecha'].dt.dayofweek
    df['Semana']    = df['Fecha'].dt.isocalendar().week.astype(int)
    df['Anio']      = df['Fecha'].dt.year

    df_agg = df.groupby(['Anio','Semana','Producto','Pais']).agg(
        Cantidad  = ('Cantidad', 'sum'),
        Precio    = ('Precio', 'mean'),
        Mes       = ('Mes', 'first'),
        Trimestre = ('Trimestre', 'first'),
        DiaSemana = ('DiaSemana', 'mean'),
        Pedidos   = ('Cantidad', 'count'),
    ).reset_index()

    le_pais = LabelEncoder()
    le_prod = LabelEncoder()
    df_agg['Pais_Code']    = le_pais.fit_transform(df_agg['Pais'])
    df_agg['Producto_Code'] = le_prod.fit_transform(df_agg['Producto'])

    features = ['Precio','Pais_Code','Mes','Trimestre','Semana','DiaSemana','Anio','Pedidos','Producto_Code']
    X, y = df_agg[features], df_agg['Cantidad']

    if len(df_agg) < 20:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    modelo = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    modelo.fit(Xtr, y_train)
    pred = modelo.predict(Xte)

    r2 = max(0, r2_score(y_test, pred))
    return {
        'df': df, 'df_agg': df_agg, 'features': features,
        'scaler': scaler, 'le_pais': le_pais, 'le_prod': le_prod,
        'modelo': modelo, 'confiabilidad': round(min(95, max(55, r2*100)), 0),
        'ts': datetime.now(),
    }, None


# ── Predicción e inventario ───────────────────────────────────

def predecir(p, producto, semanas=1):
    df_agg, le_pais, le_prod = p['df_agg'], p['le_pais'], p['le_prod']
    hist = df_agg[df_agg['Producto'] == producto]
    if len(hist) == 0:
        return 0
    pais = hist['Pais'].mode().iloc[0]
    precio = hist['Precio'].mean()
    pedidos = hist['Pedidos'].mean()
    sem = hist['Semana'].max() + semanas
    anio = hist['Anio'].max()
    if sem > 52: sem -= 52; anio += 1
    mes = min(12, max(1, round(sem / 4.33)))
    trim = ((mes-1)//3)+1
    try: pc = le_pais.transform([pais])[0]
    except: pc = 0
    try: prod_c = le_prod.transform([producto])[0]
    except: prod_c = 0
    entrada = pd.DataFrame([[precio, pc, mes, trim, sem, 3, anio, pedidos, prod_c]], columns=p['features'])
    return max(0, round(p['modelo'].predict(p['scaler'].transform(entrada))[0]))


def estado_inventario(p, producto, stock):
    hist = p['df_agg'][p['df_agg']['Producto'] == producto]
    dem = predecir(p, producto)
    std = hist['Cantidad'].std()
    if pd.isna(std) or std == 0: std = hist['Cantidad'].mean() * 0.3
    seg = round(1.65 * std)
    repo = dem + seg
    dd = dem/7 if dem > 0 else 0.01
    dias = round(stock / dd)
    if stock < seg: est, unis = 'critico', max(0, repo + dem - stock)
    elif stock < repo: est, unis = 'atencion', max(0, repo - stock)
    elif stock > repo * 1.8: est, unis = 'exceso', 0
    else: est, unis = 'optimo', 0
    fecha = datetime.now() + timedelta(days=max(1, dias-2))
    return {'dem': dem, 'seg': int(seg), 'repo': int(repo), 'dias': dias,
            'estado': est, 'unis': int(unis),
            'fecha_dt': fecha, 'fecha': fecha.strftime('%d/%m/%Y')}


# ── Login ─────────────────────────────────────────────────────

def show_login():
    c1, c2, c3 = st.columns([1, 1.3, 1])
    with c2:
        st.markdown('<br><br>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{C_BG_SOFT}; border:1px solid {C_CARD_BORDER};
                    border-radius:18px; padding:36px 34px;">
            <div style="text-align:center; font-size:48px; margin-bottom:14px;">📦</div>
            <div class="login-title">StockSense</div>
            <div class="login-sub">Predicción inteligente de demanda e inventario</div>
        """, unsafe_allow_html=True)
        with st.form("lf"):
            usuario  = st.text_input("Usuario", placeholder="Tu usuario")
            password = st.text_input("Contraseña", type="password", placeholder="Tu contraseña")
            ok = st.form_submit_button("Ingresar", use_container_width=True, type="primary")
        if ok:
            if usuario in USERS and USERS[usuario] == password:
                st.session_state.update({'logged_in': True, 'usuario': usuario})
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")
        st.markdown(f"""
            <div class="login-footer-box">
                Acceso demo: admin / admin123
            </div>
        </div>""", unsafe_allow_html=True)


# ── Topbar y sidebar ──────────────────────────────────────────

def show_topbar():
    u = st.session_state.get('usuario', '')
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-brand">
            <div class="logo-box">S</div>
            <div>
                <div class="brand-name">StockSense</div>
                <div class="brand-sub">INVENTARIO INTELIGENTE</div>
            </div>
        </div>
        <div class="topbar-user">Sesión activa<br><b>{u.upper()}</b></div>
    </div>""", unsafe_allow_html=True)

def show_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:10px 0 18px;">
            <div style="font-size:34px;">📦</div>
            <div style="font-size:15px; font-weight:800; margin-top:6px;">StockSense</div>
        </div>
        <hr style="border-color:{C_CARD_BORDER};">
        """, unsafe_allow_html=True)
        st.caption(f"Usuario: {st.session_state.get('usuario','').upper()}")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Configuración y datos", use_container_width=True):
            st.session_state['_cfg'] = True
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<hr style='border-color:{C_CARD_BORDER};'>", unsafe_allow_html=True)
        if st.button("Cerrar sesión", use_container_width=True):
            st.session_state.clear()
            st.rerun()


# ── Página: Configuración ─────────────────────────────────────

def page_config():
    st.markdown("## Configuración y datos")
    t1, t2 = st.tabs(["Cargar historial de ventas", "Usuarios"])

    with t1:
        subtle_header("Sube tu historial de ventas")
        st.markdown("""
        Sube un archivo **Excel (.xlsx)**, **CSV (.csv)** o **ZIP**.
        El archivo debe tener columnas de: **fecha**, **producto**, **cantidad** y **precio**.
        Los nombres exactos pueden variar — el sistema los detecta automáticamente.
        """)

        uploaded = st.file_uploader("Selecciona tu archivo", type=['csv','xlsx','xls','zip'])

        if uploaded is not None:
            # Guardamos los bytes en session_state para no re-leer en cada rerun
            if st.session_state.get('_uf_name') != uploaded.name:
                st.session_state['_uf_bytes'] = uploaded.getvalue()
                st.session_state['_uf_name']  = uploaded.name

            file_bytes = io.BytesIO(st.session_state['_uf_bytes'])
            df_raw, error = cargar_archivo(file_bytes, uploaded.name)

            if error:
                st.error(error)
                return

            st.success(f"Archivo leído: {len(df_raw):,} filas · {len(df_raw.columns)} columnas")

            mapeo, faltantes = detectar_columnas(df_raw)

            if faltantes:
                st.warning(f"No se detectaron automáticamente: **{', '.join(faltantes)}**. "
                          "Selecciona a qué columna corresponde cada campo:")
                cols = df_raw.columns.tolist()
                manual = {}
                for campo in faltantes:
                    manual[campo] = st.selectbox(f"Columna para '{campo}'", cols, key=f"m_{campo}")
                mapeo.update(manual)
            else:
                st.info(f"Columnas detectadas — fecha: **{mapeo['fecha']}** · "
                       f"producto: **{mapeo['producto']}** · "
                       f"cantidad: **{mapeo['cantidad']}** · "
                       f"precio: **{mapeo['precio']}**")

            if st.button("Analizar mi información", type="primary", use_container_width=True):
                with st.spinner("Procesando tu historial de ventas..."):
                    payload, err = entrenar(
                        df_raw, mapeo['fecha'], mapeo['producto'],
                        mapeo['cantidad'], mapeo['precio'], mapeo.get('pais')
                    )
                if err:
                    st.error(err)
                else:
                    st.session_state['payload'] = payload
                    st.success("Listo. Tu información fue procesada correctamente.")
                    c1, c2, c3 = st.columns(3)
                    with c1: metric_card("Registros", f"{len(payload['df']):,}", "ventas válidas")
                    with c2: metric_card("Productos", f"{payload['df_agg']['Producto'].nunique():,}", "detectados")
                    with c3: metric_card("Confiabilidad", f"{payload['confiabilidad']:.0f}%", "del modelo")
                    st.info("Ve a la pestaña Inicio para ver tu panel.")

        elif 'payload' in st.session_state:
            st.success("Ya tienes datos cargados. Puedes subir un archivo nuevo para reemplazarlos.")

    with t2:
        subtle_header("Usuarios con acceso")
        st.dataframe(pd.DataFrame([
            {'Usuario': 'admin',    'Rol': 'Administrador'},
            {'Usuario': 'gerente',  'Rol': 'Gerencia'},
            {'Usuario': 'analista', 'Rol': 'Análisis y reportes'},
        ]), use_container_width=True, hide_index=True)

    footer_credits()


# ── Página: Inicio ────────────────────────────────────────────

def page_inicio():
    if 'payload' not in st.session_state:
        st.markdown("## Bienvenido a StockSense")
        st.markdown("Sube tu historial de ventas desde **Configuración y datos** para comenzar.")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            subtle_header("Cómo empezar")
            st.markdown("""
            1. Ve a **Configuración y datos** en el menú lateral y sube tu archivo.
            2. El sistema analiza tu información automáticamente.
            3. En segundos tienes alertas, predicciones y reportes listos.
            """)
            if st.button("Subir mi historial de ventas", type="primary", use_container_width=True):
                st.session_state['_cfg'] = True
                st.rerun()
        with c2:
            subtle_header("Qué obtienes")
            st.markdown("""
            - Qué productos se te van a acabar
            - Qué productos te están sobrando
            - Cuánto vas a vender la próxima semana
            - Cuándo y cuánto reponer
            - Reportes para tu equipo o proveedores
            """)
        footer_credits()
        return

    p = st.session_state['payload']
    df_agg = p['df_agg']
    productos = df_agg['Producto'].unique()

    if 'stocks' not in st.session_state:
        st.session_state['stocks'] = {}

    estados = []
    for prod in productos[:60]:
        s = st.session_state['stocks'].get(prod)
        if s is None:
            h = df_agg[df_agg['Producto']==prod]['Cantidad']
            s = int(h.mean()*2) if len(h) else 50
            st.session_state['stocks'][prod] = s
        info = estado_inventario(p, prod, s)
        info['producto'] = prod
        info['stock'] = s
        estados.append(info)

    criticos = [e for e in estados if e['estado']=='critico']
    atencion = [e for e in estados if e['estado']=='atencion']
    exceso   = [e for e in estados if e['estado']=='exceso']
    dem_total = sum(e['dem'] for e in estados)

    st.caption(f"Actualizado: {p['ts'].strftime('%d/%m/%Y %H:%M')} · Confiabilidad: {p['confiabilidad']:.0f}%")

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Por agotarse", str(len(criticos)), "riesgo crítico", "danger")
    with c2: metric_card("Necesitan atención", str(len(atencion)), "bajo nivel ideal", "warning")
    with c3: metric_card("Con exceso", str(len(exceso)), "sobrestock", "info")
    with c4: metric_card("Ventas próx. semana", f"{dem_total:,.0f}", "unidades estimadas", "success")

    section_header("Productos que se van a agotar")
    if criticos:
        for e in sorted(criticos, key=lambda x: x['dias'])[:8]:
            ca, cb = st.columns([4,1])
            with ca:
                st.markdown(f"""<div class="product-row danger">
                    <b>{e['producto'][:60]}</b><br>
                    <span style="font-size:13px;">Se agota en {e['dias']} días · Stock: {e['stock']} · Pedir {e['unis']} unidades antes del {e['fecha']}</span>
                </div>""", unsafe_allow_html=True)
            with cb:
                if st.button("Ver", key=f"v_{e['producto']}", use_container_width=True):
                    st.session_state['_det'] = e['producto']
                    st.session_state['_inv'] = True
                    st.rerun()
    else:
        st.markdown('<div class="alert-success">No hay productos en riesgo crítico.</div>', unsafe_allow_html=True)

    section_header("Productos con sobrestock")
    if exceso:
        for e in exceso[:5]:
            st.markdown(f"""<div class="product-row warning">
                <b>{e['producto'][:60]}</b><br>
                <span style="font-size:13px;">Stock: {e['stock']} · Demanda estimada: {e['dem']} unidades/semana</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-success">No se detecta sobrestock relevante.</div>', unsafe_allow_html=True)

    section_header("Demanda estimada próxima semana")
    top = sorted(estados, key=lambda x: -x['dem'])[:6]
    df_top = pd.DataFrame([{'Producto': e['producto'][:38], 'Unidades': e['dem']} for e in top])
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.barh(df_top['Producto'][::-1], df_top['Unidades'][::-1], color=C_ACCENT_SOFT, edgecolor=C_BG, height=0.6)
    ax.set_title('Top productos por demanda estimada', fontweight='bold', fontsize=11, pad=8)
    ax.spines[['top','right']].set_visible(False)
    mx = df_top['Unidades'].max() if df_top['Unidades'].max() > 0 else 1
    for i, v in enumerate(df_top['Unidades'][::-1]):
        ax.text(v + mx*0.02, i, f'{v:.0f}', va='center', fontsize=9, color=C_TEXT, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    footer_credits()


# ── Página: Inventario ────────────────────────────────────────

def page_inventario():
    p = st.session_state['payload']
    df_agg = p['df_agg']
    productos = sorted(df_agg['Producto'].unique())

    if 'stocks' not in st.session_state:
        st.session_state['stocks'] = {}

    presel = st.session_state.pop('_det', None)
    subtle_header("Buscar producto")
    c1, c2 = st.columns([3,1])
    with c1:
        idx = productos.index(presel) if presel in productos else 0
        prod = st.selectbox("Producto", productos, index=idx, label_visibility="collapsed")
    with c2:
        default = st.session_state['stocks'].get(prod,
            int(df_agg[df_agg['Producto']==prod]['Cantidad'].mean()*2))
        stock = st.number_input("Stock actual", min_value=0, max_value=1000000, value=default, step=1)
        st.session_state['stocks'][prod] = stock

    info = estado_inventario(p, prod, stock)
    labels = {
        'critico':  ('Riesgo crítico de desabastecimiento', 'alert-danger'),
        'atencion': ('Requiere reposición pronto', 'alert-warning'),
        'exceso':   ('Sobrestock detectado', 'alert-warning'),
        'optimo':   ('Nivel óptimo', 'alert-success'),
    }
    txt, cls = labels[info['estado']]
    st.markdown(f'<div class="{cls}">{txt}</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Demanda estimada", f"{info['dem']:,}", "unidades próx. semana")
    with c2: metric_card("Stock mínimo", f"{info['seg']:,}", "stock de seguridad")
    with c3: metric_card("Reponer cuando queden", f"{info['repo']:,}", "unidades")
    with c4: metric_card("Cobertura actual", f"{info['dias']} días", "con el stock de hoy")

    if info['unis'] > 0:
        st.markdown(f'<div class="alert-warning">Recomendamos pedir <b>{info["unis"]} unidades</b> antes del <b>{info["fecha"]}</b>.</div>', unsafe_allow_html=True)

    section_header(f"Historial de ventas — {prod[:50]}")
    hist = df_agg[df_agg['Producto']==prod].sort_values(['Anio','Semana'])
    if len(hist) > 1:
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(range(len(hist)), hist['Cantidad'], color=C_ACCENT_SOFT, linewidth=2, marker='o', markersize=4)
        ax.fill_between(range(len(hist)), hist['Cantidad'], alpha=0.15, color=C_ACCENT_SOFT)
        ax.axhline(info['dem'], color=C_SUCCESS, linestyle='--', linewidth=2, label=f"Predicción: {info['dem']}")
        labels_x = [f"S{int(s)}" for s in hist['Semana']]
        step = max(1, len(labels_x)//12)
        ax.set_xticks(range(0, len(labels_x), step))
        ax.set_xticklabels(labels_x[::step], rotation=40, fontsize=8)
        ax.set_ylabel('Unidades')
        ax.spines[['top','right']].set_visible(False)
        leg = ax.legend(facecolor=C_CARD, edgecolor=C_CARD_BORDER, fontsize=9)
        for t in leg.get_texts(): t.set_color(C_TEXT)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Historial insuficiente para mostrar el gráfico.")

    section_header("Todos los productos")
    tabla = []
    for pp in productos[:200]:
        s = st.session_state['stocks'].get(pp, int(df_agg[df_agg['Producto']==pp]['Cantidad'].mean()*2))
        i = estado_inventario(p, pp, s)
        tabla.append({
            'Producto': pp[:50], 'Stock': s, 'Demanda estimada': i['dem'],
            'Estado': {'critico':'Crítico','atencion':'Atención','exceso':'Exceso','optimo':'Óptimo'}[i['estado']],
        })
    st.dataframe(pd.DataFrame(tabla), use_container_width=True, height=360)
    footer_credits()


# ── Página: Predicciones ──────────────────────────────────────

def page_predicciones():
    p = st.session_state['payload']
    df_agg = p['df_agg']

    if 'stocks' not in st.session_state:
        st.session_state['stocks'] = {}

    st.markdown("Planifica tus próximos pedidos antes de que falte stock. "
               "Los productos se ordenan por urgencia.")

    productos = df_agg['Producto'].unique()
    eventos = []
    for prod in productos[:80]:
        s = st.session_state['stocks'].get(prod)
        if s is None:
            h = df_agg[df_agg['Producto']==prod]['Cantidad']
            s = int(h.mean()*2) if len(h) else 50
            st.session_state['stocks'][prod] = s
        info = estado_inventario(p, prod, s)
        if info['unis'] > 0:
            info['producto'] = prod
            eventos.append(info)

    eventos = sorted(eventos, key=lambda x: x['fecha_dt'])

    c1,c2,c3 = st.columns(3)
    with c1: metric_card("Pedidos pendientes", str(len(eventos)), "productos a reponer", "warning")
    prox7 = [e for e in eventos if (e['fecha_dt']-datetime.now()).days <= 7]
    with c2: metric_card("Urgentes esta semana", str(len(prox7)), "requieren pedido pronto", "danger")
    total_u = sum(e['unis'] for e in eventos)
    with c3: metric_card("Unidades totales a pedir", f"{total_u:,}", "en todos los productos", "info")

    section_header("Calendario de reposición")

    MESES = {1:'ENE',2:'FEB',3:'MAR',4:'ABR',5:'MAY',6:'JUN',
             7:'JUL',8:'AGO',9:'SEP',10:'OCT',11:'NOV',12:'DIC'}

    if not eventos:
        st.markdown('<div class="alert-success">No hay pedidos pendientes. Tu inventario está en buen estado.</div>', unsafe_allow_html=True)
    else:
        for e in eventos[:25]:
            dias = (e['fecha_dt']-datetime.now()).days
            if dias <= 3: color, bg = C_DANGER, C_DANGER_BG
            elif dias <= 10: color, bg = C_WARNING, C_WARNING_BG
            else: color, bg = C_ACCENT_SOFT, C_BG_SOFT

            urgencia = "Urgente" if dias <= 3 else ("Próximamente" if dias <= 10 else "Planificado")

            ca, cb, cc = st.columns([0.9, 4, 1])
            with ca:
                st.markdown(f"""
                <div style="min-width:70px; text-align:center; padding:10px;
                            background:{bg}; border-radius:10px; color:{color};
                            font-weight:800; margin-top:4px;">
                    <div style="font-size:20px; line-height:1;">{e['fecha_dt'].day}</div>
                    <div style="font-size:10px; letter-spacing:0.5px;">{MESES[e['fecha_dt'].month]}</div>
                </div>""", unsafe_allow_html=True)
            with cb:
                st.markdown(f"""
                <div style="padding-top:6px;">
                    <b style="color:{C_TEXT};">{e['producto'][:55]}</b><br>
                    <span style="font-size:13px; color:{C_TEXT_DIM};">
                        {urgencia} · Pedir <b style="color:{C_TEXT};">{e['unis']} unidades</b> · Cobertura: {e['dias']} días
                    </span>
                </div>""", unsafe_allow_html=True)
            with cc:
                if st.button("Ver producto", key=f"r_{e['producto']}", use_container_width=True):
                    st.session_state['_det'] = e['producto']
                    st.session_state['_inv'] = True
                    st.rerun()
            st.markdown("<div style='height:2px;'></div>", unsafe_allow_html=True)

    footer_credits()


# ── Página: Reportes ──────────────────────────────────────────

def page_reportes():
    p = st.session_state['payload']
    df_agg = p['df_agg']

    if 'stocks' not in st.session_state:
        st.session_state['stocks'] = {}

    st.markdown("Descarga el estado completo de tu inventario con predicciones y alertas.")

    productos = df_agg['Producto'].unique()
    filas = []
    for prod in productos[:200]:
        s = st.session_state['stocks'].get(prod, int(df_agg[df_agg['Producto']==prod]['Cantidad'].mean()*2))
        i = estado_inventario(p, prod, s)
        filas.append({
            'Producto': prod, 'Stock actual': s,
            'Demanda (próx. semana)': i['dem'],
            'Stock mínimo': i['seg'],
            'Reponer en': i['repo'],
            'Días de cobertura': i['dias'],
            'Estado': {'critico':'Crítico','atencion':'Atención','exceso':'Exceso','optimo':'Óptimo'}[i['estado']],
            'Unidades a pedir': i['unis'],
        })

    df_rep = pd.DataFrame(filas)
    section_header("Vista previa del reporte")
    st.dataframe(df_rep, use_container_width=True, height=340)

    c1, c2 = st.columns(2)
    with c1:
        csv = df_rep.to_csv(index=False).encode('utf-8-sig')
        st.download_button("Descargar en Excel (CSV)", csv,
            file_name=f"reporte_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv', use_container_width=True, type="primary")
    with c2:
        criticos = df_rep[df_rep['Estado']=='Crítico']
        exceso   = df_rep[df_rep['Estado']=='Exceso']
        txt = f"""RESUMEN EJECUTIVO — {datetime.now().strftime('%d/%m/%Y %H:%M')}
{'='*50}
Ventas estimadas próxima semana: {df_rep['Demanda (próx. semana)'].sum():,.0f} unidades
Productos en riesgo crítico: {len(criticos)}
{chr(10).join('- '+str(r['Producto'])[:55]+f" (pedir {r['Unidades a pedir']})" for _,r in criticos.head(15).iterrows())}
Productos con sobrestock: {len(exceso)}
{chr(10).join('- '+str(r['Producto'])[:55] for _,r in exceso.head(15).iterrows())}
{'='*50}
Generado por StockSense"""
        st.download_button("Descargar resumen ejecutivo (TXT)", txt.encode('utf-8'),
            file_name=f"resumen_{datetime.now().strftime('%Y%m%d')}.txt",
            mime='text/plain', use_container_width=True)

    footer_credits()


# ── Main ──────────────────────────────────────────────────────

def main():
    if not st.session_state.get('logged_in'):
        show_login()
        return

    show_sidebar()
    show_topbar()

    if st.session_state.pop('_cfg', False):
        page_config()
        return

    fuerza_inv = st.session_state.pop('_inv', False)

    if 'payload' not in st.session_state:
        page_inicio()
        return

    tabs = st.tabs(["Inicio", "Mi Inventario", "Predicciones", "Reportes"])
    with tabs[0]: page_inicio()
    with tabs[1]:
        if fuerza_inv:
            page_inventario()
        else:
            page_inventario()
    with tabs[2]: page_predicciones()
    with tabs[3]: page_reportes()


if __name__ == "__main__":
    main()
