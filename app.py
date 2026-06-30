import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
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
#  ESTILOS CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
    .stApp .stMarkdown, .stApp .stText {
        color: #14213d !important;
    }

    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #e8edf5 !important;
    }

    .metric-card, .metric-card *,
    .section-header, .section-header *,
    .winner-badge, .winner-badge *,
    .alert-danger, .alert-danger *,
    .alert-warning, .alert-warning *,
    .alert-success, .alert-success *,
    .login-footer-box, .login-footer-box *,
    .product-row, .product-row *,
    .footer-credits, .footer-credits * {
        color: inherit !important;
    }
    .metric-card .label   { color: #8a93a3 !important; }
    .metric-card .value   { color: #14213d !important; }
    .metric-card .sub     { color: #9aa3b2 !important; }
    .section-header       { color: #ffffff !important; }
    .winner-badge         { color: #ffffff !important; }
    .alert-danger         { color: #9f1239 !important; }
    .alert-warning        { color: #92400e !important; }
    .alert-success        { color: #166534 !important; }
    .login-footer-box     { color: #8a93a3 !important; }
    .login-title          { color: #14213d !important; }
    .login-sub            { color: #8a93a3 !important; }
    .footer-credits       { color: #9aa3b2 !important; }

    .stButton > button, .stButton > button * { color: #ffffff !important; }
    section[data-testid="stSidebar"] .stButton > button { color: #ffffff !important; }

    input, textarea, select {
        color: #14213d !important;
        background-color: #ffffff !important;
    }

    .stApp {
        background: linear-gradient(180deg, #f4f7fb 0%, #eef2f8 100%) !important;
    }

    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; }

    h1 { color: #14213d; font-weight: 800 !important; letter-spacing: -0.5px; }
    h2, h3 { color: #1e293b; font-weight: 700 !important; }

    /* ───────── Tarjetas de métricas ───────── */
    .metric-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 22px 20px;
        box-shadow: 0 4px 16px rgba(20, 33, 61, 0.06);
        border: 1px solid #eef1f6;
        text-align: center;
        margin-bottom: 14px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(20, 33, 61, 0.10);
    }
    .metric-card.danger  { border-top: 4px solid #ef4444; }
    .metric-card.warning { border-top: 4px solid #f59e0b; }
    .metric-card.success { border-top: 4px solid #22c55e; }
    .metric-card.info     { border-top: 4px solid #2563eb; }
    .metric-card .label {
        font-size: 12px; color: #8a93a3; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.8px;
    }
    .metric-card .value {
        font-size: 30px; font-weight: 800; color: #14213d;
        margin: 8px 0 4px 0; line-height: 1.1;
    }
    .metric-card .sub { font-size: 12px; color: #9aa3b2; font-weight: 500; }

    /* ───────── Encabezado de sección ───────── */
    .section-header {
        background: linear-gradient(120deg, #0f2747 0%, #2563eb 100%);
        color: #ffffff; padding: 16px 22px; border-radius: 12px;
        font-size: 16px; font-weight: 700; margin: 26px 0 16px 0;
        letter-spacing: 0.2px; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.18);
    }

    .winner-badge {
        background: linear-gradient(120deg, #047857 0%, #10b981 100%);
        color: white; padding: 8px 20px; border-radius: 24px;
        font-size: 13px; font-weight: 700; display: inline-block;
        margin-bottom: 14px; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
    }

    /* ───────── Alertas ───────── */
    .alert-danger, .alert-warning, .alert-success {
        padding: 14px 18px; border-radius: 10px; margin: 10px 0;
        font-weight: 600; font-size: 14.5px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .alert-danger  { background: #fff1f2; border-left: 5px solid #ef4444; color: #9f1239; }
    .alert-warning { background: #fffbeb; border-left: 5px solid #f59e0b; color: #92400e; }
    .alert-success { background: #f0fdf4; border-left: 5px solid #22c55e; color: #166534; }

    /* ───────── Filas de producto (listas clicables) ───────── */
    .product-row {
        background: #ffffff;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
        border: 1px solid #eef1f6;
        border-left: 4px solid #cbd5e1;
        box-shadow: 0 2px 8px rgba(20,33,61,0.04);
    }
    .product-row.danger  { border-left-color: #ef4444; }
    .product-row.warning { border-left-color: #f59e0b; }
    .product-row.success { border-left-color: #22c55e; }

    /* ───────── Login ───────── */
    .login-card-icon { text-align: center; margin-bottom: 18px; }
    .login-title {
        text-align: center; font-size: 24px; font-weight: 800;
        color: #14213d; margin-bottom: 4px; letter-spacing: -0.3px;
    }
    .login-sub {
        text-align: center; font-size: 13.5px; color: #8a93a3;
        margin-bottom: 30px; font-weight: 500;
    }
    .login-footer-box {
        text-align: center; margin-top: 22px; padding: 14px;
        background: #f8fafc; border-radius: 10px;
        font-size: 12px; color: #8a93a3; line-height: 1.8;
    }

    /* ───────── Botones ───────── */
    .stButton > button {
        border-radius: 10px !important; font-weight: 700 !important;
        padding: 10px 20px !important; border: none !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(120deg, #1e40af 0%, #2563eb 100%) !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.4) !important;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox > div > div {
        border-radius: 10px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background: #eef2f8; padding: 6px; border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; font-weight: 600; color: #64748b; padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important; color: #1e40af !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    [data-testid="stDataFrame"] {
        border-radius: 12px; overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    /* ───────── Sidebar ───────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2747 0%, #14213d 100%);
    }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        padding: 8px 10px; border-radius: 8px; margin-bottom: 2px;
        transition: background 0.15s ease;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.08) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(239,68,68,0.85) !important;
        border-color: rgba(239,68,68,0.85) !important;
    }

    [data-testid="stAlert"] { border-radius: 12px; }

    .footer-credits {
        text-align: center; font-size: 11.5px; color: #9aa3b2;
        margin-top: 50px; padding-top: 18px;
        border-top: 1px solid #e6eaf1;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
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


def footer_credits():
    st.markdown("""
    <div class="footer-credits">
        StockSense © 2026 &nbsp;·&nbsp; Plataforma de predicción de demanda e inventario<br>
        Desarrollado por Sara Camila Mayorca Parra &amp; Luis Alejandro Palomino Acuña
    </div>
    """, unsafe_allow_html=True)


# ============================================================
#  MOTOR DE DATOS Y MODELO (sin exponer detalles técnicos al usuario)
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
    """Intenta mapear las columnas del archivo subido a los campos
    que el sistema necesita, sin que el usuario tenga que saber nombres técnicos."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    mapeo = {}
    faltantes = []

    for campo, alias in REQUIRED_COLS_MAP.items():
        encontrado = None
        for a in alias:
            if a in cols_lower:
                encontrado = cols_lower[a]
                break
        if encontrado:
            mapeo[campo] = encontrado
        else:
            faltantes.append(campo)

    for campo, alias in OPTIONAL_COLS_MAP.items():
        for a in alias:
            if a in cols_lower:
                mapeo[campo] = cols_lower[a]
                break

    return mapeo, faltantes


@st.cache_data(show_spinner=False)
def procesar_archivo(file_bytes, file_name):
    """Carga el archivo (csv, xlsx o zip con xlsx) y lo deja crudo."""
    try:
        if file_name.endswith('.zip'):
            with zipfile.ZipFile(file_bytes) as z:
                fname = [f for f in z.namelist()
                         if f.endswith(('.xlsx', '.xls'))][0]
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
def entrenar_sistema(df_raw, col_fecha, col_producto, col_cantidad,
                      col_precio, col_pais):
    """Limpieza, preparación y entrenamiento del modelo predictivo.
    Todo el detalle metodológico vive aquí, oculto al usuario final."""

    df = df_raw.copy()
    df = df.rename(columns={
        col_fecha: 'Fecha', col_producto: 'Producto',
        col_cantidad: 'Cantidad', col_precio: 'Precio',
    })
    if col_pais:
        df = df.rename(columns={col_pais: 'Pais'})
    else:
        df['Pais'] = 'General'

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
        Cantidad      = ('Cantidad', 'sum'),
        Precio        = ('Precio', 'mean'),
        Mes           = ('Mes', 'first'),
        Trimestre     = ('Trimestre', 'first'),
        DiaSemana     = ('DiaSemana', 'mean'),
        Pedidos       = ('Cantidad', 'count'),
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
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    modelo = RandomForestRegressor(
        n_estimators=150, max_depth=12, min_samples_split=4,
        random_state=42, n_jobs=-1
    )
    modelo.fit(X_train_s, y_train)
    y_pred = modelo.predict(X_test_s)

    precision_mae = mean_absolute_error(y_test, y_pred)
    precision_r2  = max(0, r2_score(y_test, y_pred))
    confiabilidad_pct = round(min(95, max(55, precision_r2 * 100)), 0)

    payload = {
        'df_raw':      df,
        'df_agg':      df_agg,
        'features':    features,
        'scaler':      scaler,
        'le_pais':     le_pais,
        'le_prod':     le_prod,
        'modelo':      modelo,
        'mae':         precision_mae,
        'confiabilidad': confiabilidad_pct,
        'fecha_proceso': datetime.now(),
    }
    return payload, None


def predecir_producto(payload, producto, pais=None, semanas_adelante=1):
    """Devuelve la predicción de demanda para un producto específico
    en la(s) próxima(s) semana(s)."""
    df_agg = payload['df_agg']
    le_pais = payload['le_pais']
    le_prod = payload['le_prod']
    scaler  = payload['scaler']
    modelo  = payload['modelo']
    features = payload['features']

    hist = df_agg[df_agg['Producto'] == producto]
    if len(hist) == 0:
        return None

    if pais is None:
        pais = hist['Pais'].mode().iloc[0]

    precio_prom   = hist['Precio'].mean()
    pedidos_prom  = hist['Pedidos'].mean()
    ultima_semana = hist['Semana'].max()
    anio_actual   = hist['Anio'].max()

    semana_obj = ultima_semana + semanas_adelante
    anio_obj   = anio_actual
    if semana_obj > 52:
        semana_obj -= 52
        anio_obj += 1
    mes_obj = min(12, max(1, round(semana_obj / 4.33)))
    trim_obj = ((mes_obj - 1) // 3) + 1

    try:
        pais_code = le_pais.transform([pais])[0]
    except Exception:
        pais_code = 0
    try:
        prod_code = le_prod.transform([producto])[0]
    except Exception:
        prod_code = 0

    entrada = pd.DataFrame([[
        precio_prom, pais_code, mes_obj, trim_obj,
        semana_obj, 3, anio_obj, pedidos_prom, prod_code
    ]], columns=features)

    entrada_s = scaler.transform(entrada)
    pred = max(0, round(modelo.predict(entrada_s)[0]))
    return pred


def calcular_estado_inventario(payload, producto, stock_actual):
    """Devuelve el estado (crítico, atención, óptimo) y las
    recomendaciones de reposición para un producto."""
    df_agg = payload['df_agg']
    hist = df_agg[df_agg['Producto'] == producto]

    demanda_pred = predecir_producto(payload, producto, semanas_adelante=1)
    if demanda_pred is None:
        return None

    std_hist = hist['Cantidad'].std()
    if pd.isna(std_hist) or std_hist == 0:
        std_hist = hist['Cantidad'].mean() * 0.3

    stock_seguridad = round(1.65 * std_hist)
    punto_reposicion = demanda_pred + stock_seguridad

    demanda_diaria = demanda_pred / 7 if demanda_pred > 0 else 0.01
    dias_cobertura = round(stock_actual / demanda_diaria)

    if stock_actual < stock_seguridad:
        estado = 'critico'
        unidades_sugeridas = max(0, punto_reposicion + demanda_pred - stock_actual)
    elif stock_actual < punto_reposicion:
        estado = 'atencion'
        unidades_sugeridas = max(0, punto_reposicion - stock_actual)
    elif stock_actual > punto_reposicion * 1.8:
        estado = 'exceso'
        unidades_sugeridas = 0
    else:
        estado = 'optimo'
        unidades_sugeridas = 0

    return {
        'demanda_predicha': demanda_pred,
        'stock_seguridad': stock_seguridad,
        'punto_reposicion': round(punto_reposicion),
        'dias_cobertura': dias_cobertura,
        'estado': estado,
        'unidades_sugeridas': round(unidades_sugeridas),
        'fecha_sugerida': (datetime.now() + timedelta(days=max(1, dias_cobertura - 2))).strftime('%d/%m/%Y'),
    }


# ============================================================
#  LOGIN
# ============================================================

def show_login():
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown('<br><br>', unsafe_allow_html=True)
        st.markdown("""
        <div class="login-card-icon"><span style="font-size:54px;">📦</span></div>
        <div class="login-title">StockSense</div>
        <div class="login-sub">Predicción inteligente de demanda e inventario</div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            usuario = st.text_input("👤 Usuario", placeholder="Ingresa tu usuario")
            password = st.text_input("🔒 Contraseña", type="password",
                                     placeholder="Ingresa tu contraseña")
            submit = st.form_submit_button("Ingresar →",
                                           use_container_width=True, type="primary")

        if submit:
            if usuario in USERS and USERS[usuario] == password:
                st.session_state['logged_in'] = True
                st.session_state['usuario'] = usuario
                st.rerun()
            else:
                st.error("❌ Usuario o contraseña incorrectos.")

        st.markdown("""
        <div class="login-footer-box">
            <b>Acceso de demostración</b><br>
            admin / admin123 &nbsp;·&nbsp;
            gerente / gerente2026 &nbsp;·&nbsp;
            analista / analista2026
        </div>
        """, unsafe_allow_html=True)


# ============================================================
#  SIDEBAR / NAVEGACIÓN
# ============================================================

def show_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:24px 0 16px;">
            <div style="font-size:42px;">📦</div>
            <div style="font-size:18px; font-weight:800; margin-top:8px; letter-spacing:-0.2px;">
                StockSense
            </div>
            <div style="font-size:11.5px; opacity:0.65; margin-top:4px; letter-spacing:0.3px;">
                INVENTARIO INTELIGENTE
            </div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.12); margin:8px 0 16px;">
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.06); padding:10px 14px;
                    border-radius:10px; margin-bottom:18px; font-size:13.5px;">
            👤 <b>{st.session_state.get('usuario','').upper()}</b>
        </div>
        """, unsafe_allow_html=True)

        tiene_datos = 'payload' in st.session_state
        opciones = ["🏠 Inicio"]
        if tiene_datos:
            opciones += ["📦 Mi Inventario", "📈 Predicciones", "📄 Reportes"]
        opciones += ["⚙️ Configuración"]

        pagina = st.radio("Navegación", opciones, label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:rgba(255,255,255,0.12);'>",
                    unsafe_allow_html=True)
        if st.button("🚪 Cerrar Sesión", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    return pagina


# ============================================================
#  PÁGINA: INICIO (Dashboard — las 3 preguntas clave)
# ============================================================

def page_inicio():
    st.title("🏠 Resumen General")

    if 'payload' not in st.session_state:
        st.markdown("""
        Bienvenido a **StockSense**, tu sistema de predicción de demanda
        y gestión inteligente de inventario.
        """)
        col1, col2 = st.columns([1.3, 1])
        with col1:
            section_header("¿Cómo empezar?")
            st.markdown("""
            **1.** Ve a **⚙️ Configuración** y sube tu historial de ventas
            (Excel o CSV).

            **2.** El sistema analizará tu información automáticamente.

            **3.** En segundos tendrás tu panel con alertas de inventario,
            predicciones de ventas y reportes listos para compartir.
            """)
            if st.button("📤 Subir mi historial de ventas ahora",
                        type="primary", use_container_width=True):
                st.session_state['_goto_config'] = True
                st.rerun()
        with col2:
            section_header("Lo que vas a obtener")
            st.markdown("""
            ✅ Qué productos se te van a acabar
            ✅ Qué productos te están sobrando
            ✅ Cuánto vas a vender la próxima semana
            ✅ Sugerencias de cuánto y cuándo reponer
            ✅ Reportes listos para tu equipo o proveedores
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
        stock_actual = st.session_state['stock_data'].get(p, None)
        if stock_actual is None:
            hist_prod = df_agg[df_agg['Producto'] == p]['Cantidad']
            stock_actual = int(hist_prod.mean() * 2) if len(hist_prod) else 50
            st.session_state['stock_data'][p] = stock_actual
        info = calcular_estado_inventario(payload, p, stock_actual)
        if info:
            info['producto'] = p
            info['stock_actual'] = stock_actual
            estados.append(info)

    criticos  = [e for e in estados if e['estado'] == 'critico']
    atencion  = [e for e in estados if e['estado'] == 'atencion']
    exceso    = [e for e in estados if e['estado'] == 'exceso']
    demanda_total_semana = sum(e['demanda_predicha'] for e in estados)

    st.caption(f"Última actualización: {payload['fecha_proceso'].strftime('%d/%m/%Y %H:%M')} · "
              f"Confiabilidad del modelo: {payload['confiabilidad']:.0f}%")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Por agotarse", f"{len(criticos)}",
                    "productos en riesgo crítico", "danger")
    with col2:
        metric_card("Necesitan atención", f"{len(atencion)}",
                    "productos bajo el nivel ideal", "warning")
    with col3:
        metric_card("Con exceso", f"{len(exceso)}",
                    "productos con sobrestock", "info")
    with col4:
        metric_card("Ventas próxima semana", f"{demanda_total_semana:,.0f}",
                    "unidades estimadas (todos los productos)", "success")

    section_header("🔴 ¿Qué productos se te van a acabar?")
    if criticos:
        criticos_sorted = sorted(criticos, key=lambda x: x['dias_cobertura'])[:8]
        for e in criticos_sorted:
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"""
                <div class="product-row danger">
                    <b>{e['producto'][:60]}</b><br>
                    <span style="font-size:13px;">
                        Se agota en aproximadamente <b>{e['dias_cobertura']} días</b> ·
                        Stock actual: {e['stock_actual']} ·
                        Sugerido pedir: <b>{e['unidades_sugeridas']} unidades</b>
                        antes del {e['fecha_sugerida']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                if st.button("Ver detalle", key=f"crit_{e['producto']}",
                            use_container_width=True):
                    st.session_state['producto_detalle'] = e['producto']
                    st.session_state['_force_inventario'] = True
                    st.rerun()
    else:
        st.markdown('<div class="alert-success">✅ No tienes productos en riesgo crítico de desabastecimiento.</div>',
                   unsafe_allow_html=True)

    section_header("🟡 ¿Qué productos te están sobrando?")
    if exceso:
        for e in exceso[:6]:
            st.markdown(f"""
            <div class="product-row warning">
                <b>{e['producto'][:60]}</b><br>
                <span style="font-size:13px;">
                    Stock actual: {e['stock_actual']} unidades ·
                    Demanda estimada próxima semana: {e['demanda_predicha']} unidades ·
                    Tienes inventario para varias semanas
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-success">✅ No se detecta sobrestock relevante en tu inventario.</div>',
                   unsafe_allow_html=True)

    section_header("📈 ¿Cuánto vas a vender la próxima semana?")
    top_demanda = sorted(estados, key=lambda x: -x['demanda_predicha'])[:5]
    df_top = pd.DataFrame([{
        'Producto': e['producto'][:40],
        'Unidades estimadas': e['demanda_predicha']
    } for e in top_demanda])
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.barh(df_top['Producto'][::-1], df_top['Unidades estimadas'][::-1],
            color='#2563eb', edgecolor='white')
    ax.set_title('Productos con mayor demanda estimada', fontweight='bold', fontsize=12)
    ax.set_xlabel('Unidades')
    for i, v in enumerate(df_top['Unidades estimadas'][::-1]):
        ax.text(v + 0.5, i, f'{v:.0f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    footer_credits()


# ============================================================
#  PÁGINA: MI INVENTARIO
# ============================================================

def page_inventario():
    st.title("📦 Mi Inventario")
    payload = st.session_state['payload']
    df_agg  = payload['df_agg']
    productos = sorted(df_agg['Producto'].unique())

    if 'stock_data' not in st.session_state:
        st.session_state['stock_data'] = {}

    preseleccionado = st.session_state.pop('producto_detalle', None)

    section_header("Buscar un producto")
    col1, col2 = st.columns([3, 1])
    with col1:
        idx = productos.index(preseleccionado) if preseleccionado in productos else 0
        producto = st.selectbox("Producto", productos, index=idx,
                                label_visibility="collapsed")
    with col2:
        stock_actual = st.number_input(
            "Stock actual",
            min_value=0, max_value=1000000,
            value=st.session_state['stock_data'].get(
                producto, int(df_agg[df_agg['Producto']==producto]['Cantidad'].mean()*2)),
            step=1
        )
        st.session_state['stock_data'][producto] = stock_actual

    info = calcular_estado_inventario(payload, producto, stock_actual)
    if info is None:
        st.warning("No hay suficiente historial para este producto.")
        return

    estado_labels = {
        'critico':  ('🔴 Riesgo crítico de desabastecimiento', 'alert-danger'),
        'atencion': ('🟡 Requiere reposición pronto', 'alert-warning'),
        'exceso':   ('🟠 Sobrestock detectado', 'alert-warning'),
        'optimo':   ('🟢 Nivel óptimo', 'alert-success'),
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
            💡 Te recomendamos pedir <b>{info['unidades_sugeridas']} unidades</b>
            antes del <b>{info['fecha_sugerida']}</b> para evitar quiebre de stock.
        </div>
        """, unsafe_allow_html=True)

    section_header(f"Histórico de ventas — {producto[:50]}")
    hist = df_agg[df_agg['Producto'] == producto].sort_values(['Anio', 'Semana'])
    if len(hist) > 1:
        fig, ax = plt.subplots(figsize=(12, 4))
        x_labels = [f"S{int(s)}-{int(a)}" for s, a in zip(hist['Semana'], hist['Anio'])]
        ax.plot(range(len(hist)), hist['Cantidad'], color='#2563eb',
                linewidth=2, marker='o', markersize=4)
        ax.fill_between(range(len(hist)), hist['Cantidad'], alpha=0.15, color='#2563eb')
        ax.axhline(info['demanda_predicha'], color='#10b981', linestyle='--',
                   linewidth=2, label=f"Predicción próxima semana: {info['demanda_predicha']}")
        step = max(1, len(x_labels)//12)
        ax.set_xticks(range(0, len(x_labels), step))
        ax.set_xticklabels(x_labels[::step], rotation=45, fontsize=8)
        ax.set_ylabel('Unidades vendidas')
        ax.legend()
        ax.grid(True, alpha=0.3)
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
                'Producto': p[:50],
                'Stock actual': s,
                'Demanda estimada': i['demanda_predicha'],
                'Estado': {'critico':'🔴 Crítico','atencion':'🟡 Atención',
                           'exceso':'🟠 Exceso','optimo':'🟢 Óptimo'}[i['estado']],
            })
    st.dataframe(pd.DataFrame(tabla), use_container_width=True, height=400)

    footer_credits()


# ============================================================
#  PÁGINA: PREDICCIONES
# ============================================================

def page_predicciones():
    st.title("📈 Predicciones de Venta")
    payload = st.session_state['payload']
    df_agg  = payload['df_agg']
    productos = sorted(df_agg['Producto'].unique())

    section_header("Simulador de demanda futura")
    col1, col2 = st.columns([2, 1])
    with col1:
        producto = st.selectbox("Selecciona un producto", productos)
    with col2:
        horizonte = st.selectbox("Horizonte", [1, 2, 3, 4],
                                 format_func=lambda x: f"Próximas {x} semana(s)")

    predicciones = []
    for s in range(1, horizonte + 1):
        pred = predecir_producto(payload, producto, semanas_adelante=s)
        predicciones.append({'semana': f"Semana +{s}", 'unidades': pred or 0})

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sems = [p['semana'] for p in predicciones]
        vals = [p['unidades'] for p in predicciones]
        ax.bar(sems, vals, color='#2563eb', edgecolor='white', width=0.5)
        ax.set_title('Demanda proyectada', fontweight='bold')
        ax.set_ylabel('Unidades')
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals)*0.02, f'{v:.0f}', ha='center', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col2:
        total = sum(vals)
        metric_card("Total proyectado", f"{total:,.0f}",
                    f"unidades en {horizonte} semana(s)", "info")
        precio_prom = df_agg[df_agg['Producto']==producto]['Precio'].mean()
        metric_card("Ingreso estimado", f"${total*precio_prom:,.2f}",
                    "basado en precio promedio histórico", "success")

    section_header("Comparativa entre tus productos principales")
    top_productos = df_agg.groupby('Producto')['Cantidad'].sum()\
                      .sort_values(ascending=False).head(10).index.tolist()
    comparativa = []
    for p in top_productos:
        pred = predecir_producto(payload, p, semanas_adelante=1)
        comparativa.append({'Producto': p[:45], 'Próxima semana': pred or 0})
    df_comp = pd.DataFrame(comparativa).sort_values('Próxima semana', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df_comp['Producto'], df_comp['Próxima semana'],
            color='#0f2747', edgecolor='white')
    ax.set_title('Top 10 productos — demanda estimada próxima semana', fontweight='bold')
    ax.set_xlabel('Unidades')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    footer_credits()


# ============================================================
#  PÁGINA: REPORTES
# ============================================================

def page_reportes():
    st.title("📄 Reportes")
    payload = st.session_state['payload']
    df_agg  = payload['df_agg']

    if 'stock_data' not in st.session_state:
        st.session_state['stock_data'] = {}

    section_header("Generar reporte de inventario")
    st.markdown("""
    Descarga un resumen con el estado de tu inventario, las alertas activas
    y las predicciones de demanda, listo para compartir con tu equipo de
    compras o tus proveedores.
    """)

    productos = df_agg['Producto'].unique()
    filas = []
    for p in productos[:200]:
        s = st.session_state['stock_data'].get(p)
        if s is None:
            s = int(df_agg[df_agg['Producto']==p]['Cantidad'].mean()*2)
        i = calcular_estado_inventario(payload, p, s)
        if i:
            filas.append({
                'Producto': p,
                'Stock actual': s,
                'Demanda estimada (próx. semana)': i['demanda_predicha'],
                'Stock de seguridad': i['stock_seguridad'],
                'Punto de reposición': i['punto_reposicion'],
                'Días de cobertura': i['dias_cobertura'],
                'Estado': {'critico':'Crítico','atencion':'Atención',
                           'exceso':'Exceso','optimo':'Óptimo'}[i['estado']],
                'Unidades sugeridas a pedir': i['unidades_sugeridas'],
            })

    df_reporte = pd.DataFrame(filas)
    st.dataframe(df_reporte, use_container_width=True, height=350)

    col1, col2 = st.columns(2)
    with col1:
        csv = df_reporte.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "⬇️ Descargar reporte en Excel (CSV)",
            data=csv,
            file_name=f"reporte_inventario_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
            use_container_width=True,
            type="primary"
        )
    with col2:
        resumen_txt = generar_resumen_texto(df_reporte)
        st.download_button(
            "⬇️ Descargar resumen ejecutivo (TXT)",
            data=resumen_txt.encode('utf-8'),
            file_name=f"resumen_ejecutivo_{datetime.now().strftime('%Y%m%d')}.txt",
            mime='text/plain',
            use_container_width=True
        )

    footer_credits()


def generar_resumen_texto(df_reporte):
    criticos = df_reporte[df_reporte['Estado'] == 'Crítico']
    exceso   = df_reporte[df_reporte['Estado'] == 'Exceso']
    total_demanda = df_reporte['Demanda estimada (próx. semana)'].sum()

    lineas_criticos = chr(10).join(
        '- ' + str(r['Producto'])[:60] + f" (pedir {r['Unidades sugeridas a pedir']} unidades)"
        for _, r in criticos.head(15).iterrows()
    )
    lineas_exceso = chr(10).join(
        '- ' + str(r['Producto'])[:60] for _, r in exceso.head(15).iterrows()
    )

    txt = f"""RESUMEN EJECUTIVO DE INVENTARIO
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
    return txt


# ============================================================
#  PÁGINA: CONFIGURACIÓN (carga de datos)
# ============================================================

def page_configuracion():
    st.title("⚙️ Configuración")

    tab1, tab2 = st.tabs(["📤 Cargar historial de ventas", "👥 Usuarios y acceso"])

    with tab1:
        section_header("Sube tu historial de ventas")
        st.markdown("""
        Acepta archivos en formato **Excel (.xlsx)**, **CSV (.csv)** o **ZIP**.

        Tu archivo debe incluir, como mínimo, estas columnas (los nombres
        pueden variar, el sistema los detecta automáticamente):

        - **Fecha** de la venta
        - **Producto** o artículo vendido
        - **Cantidad** de unidades vendidas
        - **Precio** unitario

        Opcionalmente puedes incluir una columna de **país** o **tienda**
        para predicciones más precisas por ubicación.
        """)

        uploaded = st.file_uploader(
            "Arrastra tu archivo aquí",
            type=['csv', 'xlsx', 'xls', 'zip'],
            help="Tamaño máximo recomendado: 50 MB"
        )

        if uploaded:
            df_raw, error = procesar_archivo(io.BytesIO(uploaded.read()), uploaded.name)

            if error:
                st.error(f"❌ {error}")
            else:
                mapeo, faltantes = detectar_columnas(df_raw)

                if faltantes:
                    st.warning(f"""
                    No pudimos identificar automáticamente estas columnas: 
                    **{', '.join(faltantes)}**

                    Por favor selecciona manualmente qué columna de tu archivo
                    corresponde a cada dato:
                    """)
                    cols_disponibles = df_raw.columns.tolist()
                    seleccion_manual = {}
                    for campo in faltantes:
                        seleccion_manual[campo] = st.selectbox(
                            f"¿Cuál columna es '{campo}'?",
                            cols_disponibles, key=f"map_{campo}"
                        )
                    mapeo.update(seleccion_manual)

                if st.button("✅ Confirmar y analizar mi información",
                            type="primary", use_container_width=True):
                    with st.spinner("🔍 Analizando tu historial de ventas... esto puede tardar unos segundos."):
                        payload, err = entrenar_sistema(
                            df_raw,
                            mapeo['fecha'], mapeo['producto'],
                            mapeo['cantidad'], mapeo['precio'],
                            mapeo.get('pais')
                        )
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        st.session_state['payload'] = payload
                        st.success("✅ ¡Listo! Tu información fue analizada correctamente.")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            metric_card("Registros procesados", f"{len(payload['df_raw']):,}", "ventas válidas")
                        with col2:
                            metric_card("Productos detectados", f"{payload['df_agg']['Producto'].nunique():,}", "artículos distintos")
                        with col3:
                            metric_card("Confiabilidad", f"{payload['confiabilidad']:.0f}%", "del modelo predictivo")
                        st.info("Ve a **🏠 Inicio** en el menú lateral para ver tu panel completo.")

        elif 'payload' in st.session_state:
            st.success("✅ Ya tienes un historial cargado y analizado. Puedes subir uno nuevo para reemplazarlo.")

    with tab2:
        section_header("Usuarios con acceso a esta plataforma")
        st.markdown("""
        Estos son los usuarios configurados para esta demostración.
        En una implementación real, cada empresa tendría su propio
        sistema de usuarios y permisos.
        """)
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

    if st.session_state.pop('_goto_config', False):
        st.session_state['_force_config'] = True

    pagina = show_sidebar()

    if st.session_state.pop('_force_config', False):
        pagina = "⚙️ Configuración"
    if st.session_state.pop('_force_inventario', False):
        pagina = "📦 Mi Inventario"

    if   pagina == "🏠 Inicio":            page_inicio()
    elif pagina == "📦 Mi Inventario":     page_inventario()
    elif pagina == "📈 Predicciones":      page_predicciones()
    elif pagina == "📄 Reportes":          page_reportes()
    elif pagina == "⚙️ Configuración":     page_configuracion()


if __name__ == "__main__":
    main()
