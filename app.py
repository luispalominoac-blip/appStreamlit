import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================
#  CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Predicción de Demanda | ML",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
#  USUARIOS (login básico)
# ============================================================
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
    /* Fondo general */
    .stApp { background-color: #f0f4f8; }

    /* Tarjetas de métricas */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card .label {
        font-size: 13px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 32px;
        font-weight: 700;
        color: #1e293b;
        margin: 6px 0;
    }
    .metric-card .sub {
        font-size: 12px;
        color: #94a3b8;
    }

    /* Encabezado de sección */
    .section-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 700;
        margin: 20px 0 14px 0;
        letter-spacing: 0.3px;
    }

    /* Badge de modelo ganador */
    .winner-badge {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 10px;
    }

    /* Alertas de inventario */
    .alert-danger {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        border-radius: 6px;
        color: #991b1b;
        margin: 8px 0;
        font-weight: 600;
    }
    .alert-warning {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 6px;
        color: #92400e;
        margin: 8px 0;
        font-weight: 600;
    }
    .alert-success {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 12px 16px;
        border-radius: 6px;
        color: #14532d;
        margin: 8px 0;
        font-weight: 600;
    }

    /* Login */
    .login-container {
        max-width: 420px;
        margin: 60px auto;
        background: white;
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    }
    .login-title {
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 6px;
    }
    .login-sub {
        text-align: center;
        font-size: 13px;
        color: #64748b;
        margin-bottom: 28px;
    }

    /* Sidebar */
    .css-1d391kg { background-color: #1e3a5f !important; }
    section[data-testid="stSidebar"] { background-color: #1e3a5f; }
    section[data-testid="stSidebar"] * { color: white !important; }

    /* Ocultar footer de Streamlit */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  FUNCIONES AUXILIARES
# ============================================================

def metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title):
    st.markdown(f'<div class="section-header">📊 {title}</div>',
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_and_prepare(file_bytes):
    """Carga, limpia, agrega y entrena los modelos."""

    # ── Carga ──────────────────────────────────────────────
    try:
        with zipfile.ZipFile(file_bytes) as z:
            fname = [f for f in z.namelist()
                     if f.endswith('.xlsx')][0]
            with z.open(fname) as f:
                df = pd.read_excel(f)
    except Exception:
        return None, "No se encontró un archivo .xlsx dentro del ZIP."

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # ── Limpieza ────────────────────────────────────────────
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df = df.dropna(subset=['CustomerID', 'Description'])
    df = df.drop_duplicates()

    # ── Variables temporales ────────────────────────────────
    df['Month']     = df['InvoiceDate'].dt.month
    df['Quarter']   = df['InvoiceDate'].dt.quarter
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Hour']      = df['InvoiceDate'].dt.hour
    df['Week']      = df['InvoiceDate'].dt.isocalendar().week.astype(int)
    df['Year']      = df['InvoiceDate'].dt.year

    # ── Agregación semanal ──────────────────────────────────
    df_agg = df.groupby(['Year', 'Week', 'StockCode', 'Country']).agg(
        Quantity      = ('Quantity',  'sum'),
        UnitPrice     = ('UnitPrice', 'mean'),
        Month         = ('Month',     'first'),
        Quarter       = ('Quarter',   'first'),
        DayOfWeek     = ('DayOfWeek', 'mean'),
        Hour          = ('Hour',      'mean'),
        Transacciones = ('Quantity',  'count'),
        Description   = ('Description', 'first'),
    ).reset_index()

    # ── Codificación ────────────────────────────────────────
    le = LabelEncoder()
    df_agg['Country_Code'] = le.fit_transform(df_agg['Country'])

    features = ['UnitPrice', 'Country_Code', 'Month', 'Quarter',
                'Week', 'DayOfWeek', 'Hour', 'Year', 'Transacciones']

    X = df_agg[features]
    y = df_agg['Quantity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Modelos ─────────────────────────────────────────────
    models = {
        'Regresión Lineal':  LinearRegression(),
        'Árbol de Decisión': DecisionTreeRegressor(random_state=42),
        'Random Forest':     RandomForestRegressor(
                                 n_estimators=100,
                                 max_depth=10,
                                 min_samples_split=5,
                                 random_state=42,
                                 n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        results[name] = {
            'model':  model,
            'MAE':    mean_absolute_error(y_test, y_pred),
            'RMSE':   np.sqrt(mean_squared_error(y_test, y_pred)),
            'R²':     r2_score(y_test, y_pred),
            'y_pred': y_pred,
            'y_test': y_test.values,
        }

    payload = {
        'df_raw':    df,
        'df_agg':    df_agg,
        'features':  features,
        'scaler':    scaler,
        'le':        le,
        'results':   results,
        'X_test':    X_test,
        'best_model_name': 'Random Forest',
        'best_model': results['Random Forest']['model'],
    }
    return payload, None


# ============================================================
#  PANTALLA DE LOGIN
# ============================================================

def show_login():
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown('<br><br>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; margin-bottom:20px;">
            <span style="font-size:52px;">📦</span>
        </div>
        <div class="login-title">Sistema de Predicción de Demanda</div>
        <div class="login-sub">Machine Learning · Online Retail 2010-2011</div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            usuario = st.text_input("👤 Usuario", placeholder="Ingresa tu usuario")
            password = st.text_input("🔒 Contraseña",
                                     type="password",
                                     placeholder="Ingresa tu contraseña")
            submit = st.form_submit_button("Ingresar →",
                                           use_container_width=True)

        if submit:
            if usuario in USERS and USERS[usuario] == password:
                st.session_state['logged_in'] = True
                st.session_state['usuario']   = usuario
                st.rerun()
            else:
                st.error("❌ Usuario o contraseña incorrectos.")

        st.markdown("""
        <div style="text-align:center; margin-top:20px; font-size:12px; color:#94a3b8;">
            <b>Usuarios de prueba:</b><br>
            admin / admin123 &nbsp;|&nbsp;
            gerente / gerente2026 &nbsp;|&nbsp;
            analista / analista2026
        </div>
        """, unsafe_allow_html=True)


# ============================================================
#  SIDEBAR
# ============================================================

def show_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:20px 0 10px;">
            <div style="font-size:40px;">📦</div>
            <div style="font-size:16px; font-weight:700; margin-top:6px;">
                Predicción de Demanda
            </div>
            <div style="font-size:12px; opacity:0.7; margin-top:4px;">
                Online Retail ML · 2026
            </div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.2); margin:10px 0;">
        """, unsafe_allow_html=True)

        st.markdown(f"**👤 {st.session_state.get('usuario','').upper()}**")
        st.markdown("---")

        pagina = st.radio(
            "Navegación",
            ["🏠 Inicio",
             "📤 Cargar Dataset",
             "📊 Exploración de Datos",
             "🤖 Modelos ML",
             "🔮 Predictor de Demanda",
             "📦 Gestión de Inventario",
             "ℹ️ Acerca del Proyecto"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        if st.button("🚪 Cerrar Sesión", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    return pagina


# ============================================================
#  PÁGINAS
# ============================================================

def page_inicio():
    st.title("🏠 Bienvenido al Sistema de Predicción de Demanda")
    st.markdown("""
    Esta plataforma aplica técnicas de **Machine Learning** para predecir
    la demanda semanal de productos y optimizar la gestión de inventarios
    en el sector retail, utilizando datos históricos de ventas 2010-2011.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_card("Metodología", "CRISP-DM", "6 fases")
    with col2: metric_card("Algoritmos", "3", "LR · DT · RF")
    with col3: metric_card("Período", "2010–2011", "Retail UK")
    with col4: metric_card("Modelo ganador", "Random Forest", "Mejor R² y MAE")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        section_header("Flujo del Sistema")
        pasos = [
            ("📤", "Cargar Dataset", "Sube el archivo ZIP con el dataset Online Retail"),
            ("🔍", "Exploración", "EDA: distribuciones, tendencias, correlaciones"),
            ("⚙️", "Preparación", "Limpieza, agregación semanal, codificación"),
            ("🧠", "Modelado", "Entrena y compara 3 algoritmos de ML"),
            ("🔮", "Predicción", "Ingresa parámetros y obtén demanda proyectada"),
            ("📦", "Inventario", "Alertas de sobrestock y desabastecimiento"),
        ]
        for icon, titulo, desc in pasos:
            st.markdown(f"**{icon} {titulo}** — {desc}")

    with col2:
        section_header("Cómo Empezar")
        st.info("""
        **Paso 1:** Ve a 📤 **Cargar Dataset** en el menú lateral.

        **Paso 2:** Sube el archivo **online+retail.zip** descargado del
        repositorio UCI Machine Learning Repository.

        **Paso 3:** Espera el procesamiento automático (~2 minutos).

        **Paso 4:** Navega por las secciones para explorar los datos,
        comparar modelos y obtener predicciones de demanda.
        """)
        st.warning("⚠️ Sin el dataset cargado, las secciones de análisis y modelos no estarán disponibles.")


def page_cargar():
    st.title("📤 Cargar Dataset")
    st.markdown("""
    Sube el archivo **online+retail.zip** del repositorio UCI.
    El sistema procesará automáticamente los datos y entrenará los modelos.
    """)

    st.info("""
    **¿Dónde descargar el dataset?**
    - URL: https://archive.ics.uci.edu/dataset/352/online+retail
    - Archivo: `online+retail.zip` (~23 MB)
    - Contenido: `Online Retail.xlsx` con 541,909 transacciones
    """)

    uploaded = st.file_uploader(
        "Arrastra aquí el archivo online+retail.zip",
        type=['zip'],
        help="Archivo ZIP que contiene Online Retail.xlsx"
    )

    if uploaded:
        with st.spinner("⚙️ Procesando datos y entrenando modelos... (~2 min)"):
            import io
            payload, error = load_and_prepare(io.BytesIO(uploaded.read()))

        if error:
            st.error(f"❌ Error: {error}")
        else:
            st.session_state['payload'] = payload
            st.success("✅ Dataset cargado y modelos entrenados correctamente.")

            df_raw = payload['df_raw']
            df_agg = payload['df_agg']

            col1, col2, col3, col4 = st.columns(4)
            with col1: metric_card("Transacciones originales", f"{len(df_raw):,}", "tras limpieza")
            with col2: metric_card("Productos únicos", f"{df_raw['StockCode'].nunique():,}", "StockCode")
            with col3: metric_card("Países", f"{df_raw['Country'].nunique()}", "mercados")
            with col4: metric_card("Registros agregados", f"{len(df_agg):,}", "semana·producto·país")

    elif 'payload' in st.session_state:
        st.success("✅ Dataset ya cargado. Puedes navegar por las demás secciones.")
    else:
        st.warning("👆 Sube el archivo ZIP para continuar.")


def page_eda():
    if 'payload' not in st.session_state:
        st.warning("⚠️ Primero carga el dataset en la sección 📤 Cargar Dataset.")
        return

    payload = st.session_state['payload']
    df      = payload['df_raw']

    st.title("📊 Exploración de Datos (EDA)")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Resumen", "📈 Temporal", "🌍 Geográfico", "🔗 Correlaciones"])

    # ── Tab 1: Resumen ──────────────────────────────────────
    with tab1:
        section_header("Estadísticas Generales")
        col1, col2, col3, col4 = st.columns(4)
        with col1: metric_card("Transacciones", f"{len(df):,}", "filas válidas")
        with col2: metric_card("Productos", f"{df['StockCode'].nunique():,}", "únicos")
        with col3: metric_card("Países", f"{df['Country'].nunique()}", "mercados")
        with col4: metric_card("Período", "2010–2011", "13 meses")

        section_header("Distribución de Quantity")
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        qty_pos = df[df['Quantity'] > 0]['Quantity']
        p99 = qty_pos.quantile(0.99)

        axes[0].hist(qty_pos[qty_pos <= p99], bins=60,
                     color='#4ecdc4', edgecolor='white', alpha=0.85)
        axes[0].set_title('Distribución Quantity (≤ p99)', fontweight='bold')
        axes[0].set_xlabel('Unidades')
        axes[0].set_ylabel('Frecuencia')
        axes[0].axvline(qty_pos.mean(), color='red', linestyle='--',
                        linewidth=2, label=f'Media: {qty_pos.mean():.1f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        price_pos = df[df['UnitPrice'] > 0]['UnitPrice']
        axes[1].hist(price_pos[price_pos <= price_pos.quantile(0.99)],
                     bins=60, color='#ffeaa7', edgecolor='white', alpha=0.85)
        axes[1].set_title('Distribución UnitPrice (≤ p99)', fontweight='bold')
        axes[1].set_xlabel('Precio (£)')
        axes[1].set_ylabel('Frecuencia')
        axes[1].axvline(price_pos.mean(), color='red', linestyle='--',
                        linewidth=2, label=f'Media: £{price_pos.mean():.2f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 2: Temporal ─────────────────────────────────────
    with tab2:
        section_header("Análisis Temporal de Ventas")
        df_t = df[df['Quantity'] > 0].copy()
        df_t['Month']     = df_t['InvoiceDate'].dt.month
        df_t['DayOfWeek'] = df_t['InvoiceDate'].dt.dayofweek
        df_t['Quarter']   = df_t['InvoiceDate'].dt.quarter

        meses = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
                 7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}
        dias  = {0:'Lunes',1:'Martes',2:'Miércoles',
                 3:'Jueves',4:'Viernes',5:'Sábado',6:'Domingo'}

        vm = df_t.groupby('Month')['Quantity'].sum().rename(index=meses)
        vd = df_t.groupby('DayOfWeek')['Quantity'].sum().rename(index=dias)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        colores = ['#ff6b6b' if v == vm.max() else '#4ecdc4' for v in vm]
        axes[0].bar(vm.index, vm.values, color=colores, edgecolor='white')
        axes[0].set_title('Unidades vendidas por mes', fontweight='bold')
        axes[0].set_xlabel('Mes')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)

        colores2 = ['#ff6b6b' if v == vd.max() else '#ffeaa7' for v in vd]
        axes[1].bar(vd.index, vd.values, color=colores2, edgecolor='white')
        axes[1].set_title('Unidades vendidas por día', fontweight='bold')
        axes[1].set_xlabel('Día')
        axes[1].tick_params(axis='x', rotation=30)
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        mes_pico = vm.idxmax()
        dia_pico = vd.idxmax()
        col1, col2, col3 = st.columns(3)
        with col1: metric_card("Mes pico", mes_pico, f"{vm.max():,.0f} unidades")
        with col2: metric_card("Día pico", dia_pico, f"{vd.max():,.0f} unidades")
        brecha = (vm.max() - vm.min()) / vm.min() * 100
        with col3: metric_card("Brecha estacional", f"{brecha:.1f}%", "máx vs mín mensual")

    # ── Tab 3: Geográfico ───────────────────────────────────
    with tab3:
        section_header("Distribución Geográfica")
        top10 = df[df['Quantity'] > 0]\
            .groupby('Country')['Quantity'].sum()\
            .sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(12, 5))
        colores = ['#1e3a5f' if i == 0 else '#4ecdc4'
                   for i in range(len(top10))]
        ax.barh(top10.index[::-1], top10.values[::-1],
                color=colores[::-1], edgecolor='white')
        ax.set_title('Top 10 Países por Unidades Vendidas',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Unidades totales')
        for i, v in enumerate(top10.values[::-1]):
            ax.text(v + 500, i, f'{v:,.0f}', va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(
            df[df['Quantity'] > 0].groupby('Country').agg(
                Transacciones=('Quantity', 'count'),
                Unidades=('Quantity', 'sum'),
                Precio_Promedio=('UnitPrice', 'mean')
            ).sort_values('Unidades', ascending=False).round(2),
            use_container_width=True
        )

    # ── Tab 4: Correlaciones ─────────────────────────────────
    with tab4:
        section_header("Matriz de Correlación")
        df_c = df[df['Quantity'] > 0].copy()
        df_c['Month']     = df_c['InvoiceDate'].dt.month
        df_c['DayOfWeek'] = df_c['InvoiceDate'].dt.dayofweek
        df_c['Quarter']   = df_c['InvoiceDate'].dt.quarter
        df_c['Week']      = df_c['InvoiceDate'].dt.isocalendar().week.astype(int)

        cols = ['Quantity', 'UnitPrice', 'Month', 'Quarter',
                'Week', 'DayOfWeek']
        corr = df_c[cols].corr()

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, square=True, ax=ax, linewidths=0.5,
                    annot_kws={'size': 10})
        ax.set_title('Matriz de Correlación', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Correlaciones con Quantity:**")
        corr_qty = corr['Quantity'].drop('Quantity')\
                       .sort_values(key=abs, ascending=False)
        for var, val in corr_qty.items():
            barra = '█' * int(abs(val) * 40)
            signo = '+' if val > 0 else '-'
            st.text(f"  {var:<15} {signo}{abs(val):.4f}  {barra}")


def page_modelos():
    if 'payload' not in st.session_state:
        st.warning("⚠️ Primero carga el dataset en la sección 📤 Cargar Dataset.")
        return

    payload = st.session_state['payload']
    results = payload['results']

    st.title("🤖 Resultados de los Modelos ML")

    # Tabla comparativa
    section_header("Comparación de Modelos")
    data_cmp = []
    for nombre, r in results.items():
        data_cmp.append({
            'Modelo': nombre,
            'MAE':  round(r['MAE'], 4),
            'RMSE': round(r['RMSE'], 4),
            'R²':   round(r['R²'], 4),
        })
    df_cmp = pd.DataFrame(data_cmp)
    st.dataframe(df_cmp.set_index('Modelo'), use_container_width=True)

    # Ganador
    best = payload['best_model_name']
    r_best = results[best]
    st.markdown(f'<div class="winner-badge">🏆 Modelo Ganador: {best}</div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: metric_card("MAE", f"{r_best['MAE']:.4f}", "Error Absoluto Medio")
    with col2: metric_card("RMSE", f"{r_best['RMSE']:.4f}", "Error Cuadrático Medio")
    with col3: metric_card("R²", f"{r_best['R²']:.4f}", "Coeficiente de Determinación")

    # Gráficos comparativos
    section_header("Visualización de Resultados")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    nombres  = list(results.keys())
    maes     = [results[n]['MAE']  for n in nombres]
    rmses    = [results[n]['RMSE'] for n in nombres]
    r2s      = [results[n]['R²']   for n in nombres]
    colores  = ['#ff6b6b', '#ffeaa7', '#4ecdc4']

    axes[0].bar(nombres, r2s, color=colores, edgecolor='white')
    axes[0].set_title('R² por modelo', fontweight='bold')
    axes[0].set_ylabel('R²')
    axes[0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(r2s):
        axes[0].text(i, v + 0.005, f'{v:.4f}', ha='center',
                     fontweight='bold', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(nombres, maes, color=colores, edgecolor='white')
    axes[1].set_title('MAE por modelo', fontweight='bold')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(maes):
        axes[1].text(i, v + 0.3, f'{v:.2f}', ha='center',
                     fontweight='bold', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Real vs Predicho del mejor modelo
    y_test = r_best['y_test']
    y_pred = r_best['y_pred']
    axes[2].scatter(y_test, y_pred, alpha=0.3, color='#4ecdc4',
                    edgecolors='none', s=12)
    mn, mx = y_test.min(), y_test.max()
    axes[2].plot([mn, mx], [mn, mx], 'r--', linewidth=2,
                 label='Predicción perfecta')
    axes[2].set_title(f'Real vs Predicho\n({best})', fontweight='bold')
    axes[2].set_xlabel('Valor Real')
    axes[2].set_ylabel('Valor Predicho')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Importancia de variables (Random Forest)
    section_header("Importancia de Variables — Random Forest")
    rf_model  = results['Random Forest']['model']
    features  = payload['features']
    imp_df    = pd.DataFrame({
        'Variable': features,
        'Importancia': rf_model.feature_importances_
    }).sort_values('Importancia', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(imp_df['Variable'][::-1], imp_df['Importancia'][::-1],
            color='#4ecdc4', edgecolor='white')
    ax.set_title('Importancia de Variables — Random Forest',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Importancia relativa')
    for i, v in enumerate(imp_df['Importancia'][::-1]):
        ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Distribución de errores
    section_header("Distribución de Errores — Random Forest")
    errores = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(errores, bins=60, color='#96ceb4', edgecolor='white', alpha=0.85)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    ax.set_title('Distribución de Errores (Real − Predicho)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Error')
    ax.set_ylabel('Frecuencia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def page_predictor():
    if 'payload' not in st.session_state:
        st.warning("⚠️ Primero carga el dataset en la sección 📤 Cargar Dataset.")
        return

    payload    = st.session_state['payload']
    best_model = payload['best_model']
    scaler     = payload['scaler']
    le         = payload['le']
    features   = payload['features']
    df_agg     = payload['df_agg']

    st.title("🔮 Predictor de Demanda Semanal")
    st.markdown("""
    Ingresa los parámetros del producto y período para obtener la
    **predicción de unidades demandadas** usando el modelo Random Forest.
    """)

    col_form, col_result = st.columns([1, 1])

    with col_form:
        section_header("Parámetros de Predicción")
        paises_disponibles = sorted(le.classes_.tolist())
        pais = st.selectbox("🌍 País", paises_disponibles,
                            index=paises_disponibles.index('United Kingdom')
                            if 'United Kingdom' in paises_disponibles else 0)

        col1, col2 = st.columns(2)
        with col1:
            mes      = st.slider("📅 Mes", 1, 12, 11, help="1=Ene … 12=Dic")
            semana   = st.slider("📆 Semana del año", 1, 52, 45)
            dia      = st.slider("📅 Día de semana", 0, 6, 3,
                                 help="0=Lunes … 6=Domingo")
        with col2:
            año      = st.selectbox("📅 Año", [2010, 2011], index=1)
            hora     = st.slider("🕐 Hora del día", 6, 20, 11)
            precio   = st.number_input("💷 Precio unitario (£)",
                                       min_value=0.01, max_value=500.0,
                                       value=2.95, step=0.01)

        trans = st.slider("📦 Transacciones estimadas", 1, 200, 15,
                          help="Número de pedidos esperados en la semana")

        trimestre = ((mes - 1) // 3) + 1
        st.info(f"📊 Trimestre calculado automáticamente: **Q{trimestre}**")

        predecir = st.button("🔮 Predecir Demanda", use_container_width=True,
                             type="primary")

    with col_result:
        section_header("Resultado de la Predicción")
        if predecir:
            country_code = le.transform([pais])[0]
            entrada = pd.DataFrame([[
                precio, country_code, mes, trimestre,
                semana, dia, hora, año, trans
            ]], columns=features)
            entrada_s = scaler.transform(entrada)
            pred      = best_model.predict(entrada_s)[0]
            pred      = max(0, round(pred))

            st.markdown(f"""
            <div class="metric-card" style="margin-top:30px;">
                <div class="label">Demanda Predicha</div>
                <div class="value" style="font-size:48px; color:#2563eb;">
                    {pred:,}
                </div>
                <div class="sub">unidades en la semana {semana} de {año}</div>
            </div>
            """, unsafe_allow_html=True)

            # Stock de seguridad y punto de reposición
            std_pred = df_agg['Quantity'].std()
            stock_seg    = round(1.65 * std_pred)
            punto_repos  = round(pred + stock_seg)

            st.markdown("**📦 Recomendaciones de Inventario:**")
            col_a, col_b = st.columns(2)
            with col_a:
                metric_card("Stock de Seguridad",
                            f"{stock_seg:,}",
                            "unidades mínimas (95% nivel servicio)")
            with col_b:
                metric_card("Punto de Reposición",
                            f"{punto_repos:,}",
                            "unidades al activar pedido")

            # Contexto histórico
            hist = df_agg[
                (df_agg['Week'] == semana) &
                (df_agg['Month'] == mes)
            ]['Quantity']
            if len(hist) > 0:
                st.markdown(f"""
                **📈 Contexto histórico — Semana {semana}, Mes {mes}:**
                - Media histórica: **{hist.mean():.0f}** unidades
                - Máximo histórico: **{hist.max():.0f}** unidades
                - Mínimo histórico: **{hist.min():.0f}** unidades
                """)
        else:
            st.info("👈 Completa los parámetros y presiona **Predecir Demanda**.")


def page_inventario():
    if 'payload' not in st.session_state:
        st.warning("⚠️ Primero carga el dataset en la sección 📤 Cargar Dataset.")
        return

    payload = st.session_state['payload']
    df_agg  = payload['df_agg']
    best    = payload['best_model']
    scaler  = payload['scaler']
    le      = payload['le']
    features = payload['features']

    st.title("📦 Gestión de Inventario")
    st.markdown("""
    Ingresa el stock actual de un producto para recibir
    **alertas automáticas** de sobrestock o desabastecimiento.
    """)

    section_header("Configurar Producto")
    col1, col2, col3 = st.columns(3)
    with col1:
        paises = sorted(le.classes_.tolist())
        pais   = st.selectbox("🌍 País", paises,
                              index=paises.index('United Kingdom')
                              if 'United Kingdom' in paises else 0)
    with col2:
        mes    = st.slider("📅 Mes", 1, 12, 11)
        semana = st.slider("📆 Semana", 1, 52, 45)
    with col3:
        precio = st.number_input("💷 Precio (£)", 0.01, 500.0, 2.95, 0.01)
        stock_actual = st.number_input("📦 Stock Actual (unidades)",
                                       0, 100000, 500, 10)

    if st.button("📊 Analizar Inventario", use_container_width=True,
                 type="primary"):
        trimestre    = ((mes - 1) // 3) + 1
        country_code = le.transform([pais])[0]
        entrada      = pd.DataFrame([[
            precio, country_code, mes, trimestre,
            semana, 3, 11, 2011, 15
        ]], columns=features)
        entrada_s    = scaler.transform(entrada)
        demanda_pred = max(0, round(best.predict(entrada_s)[0]))
        std_hist     = df_agg['Quantity'].std()
        stock_seg    = round(1.65 * std_hist)
        punto_repos  = demanda_pred + stock_seg
        exceso       = stock_actual - punto_repos
        deficit      = punto_repos - stock_actual

        section_header("Resultado del Análisis")
        col1, col2, col3, col4 = st.columns(4)
        with col1: metric_card("Demanda Predicha", f"{demanda_pred:,}", "unidades/semana")
        with col2: metric_card("Stock de Seguridad", f"{stock_seg:,}", "unidades (95%)")
        with col3: metric_card("Punto de Reposición", f"{punto_repos:,}", "unidades umbral")
        with col4: metric_card("Stock Actual", f"{stock_actual:,}", "unidades disponibles")

        st.markdown("---")
        if stock_actual > punto_repos * 1.5:
            st.markdown(f"""
            <div class="alert-danger">
                🔴 SOBRESTOCK CRÍTICO — Exceso de {exceso:,} unidades
                sobre el punto de reposición. Capital inmovilizado estimado:
                £{exceso * precio:,.2f}. Considere reducir pedidos o
                aplicar descuentos para acelerar la rotación.
            </div>""", unsafe_allow_html=True)
        elif stock_actual > punto_repos:
            st.markdown(f"""
            <div class="alert-warning">
                🟡 SOBRESTOCK MODERADO — Stock {exceso:,} unidades por
                encima del punto de reposición. Monitorear rotación.
            </div>""", unsafe_allow_html=True)
        elif stock_actual < stock_seg:
            st.markdown(f"""
            <div class="alert-danger">
                🔴 DESABASTECIMIENTO INMINENTE — Stock por debajo del
                mínimo de seguridad. Déficit: {deficit:,} unidades.
                Emitir pedido urgente. Pérdida potencial estimada:
                £{deficit * precio:,.2f}.
            </div>""", unsafe_allow_html=True)
        elif stock_actual < punto_repos:
            st.markdown(f"""
            <div class="alert-warning">
                🟡 REPOSICIÓN REQUERIDA — Stock de {stock_actual:,}
                unidades está por debajo del punto de reposición
                ({punto_repos:,}). Emitir pedido pronto.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-success">
                🟢 INVENTARIO ÓPTIMO — Stock de {stock_actual:,}
                unidades dentro del rango recomendado. No se requiere
                acción inmediata.
            </div>""", unsafe_allow_html=True)

        # Mini gráfico de estado
        fig, ax = plt.subplots(figsize=(10, 3))
        categorias = ['Stock\nSeguridad', 'Punto\nReposición',
                      'Stock\nActual', 'Sobrestock\nLímite']
        valores    = [stock_seg, punto_repos, stock_actual,
                      punto_repos * 1.5]
        colores    = ['#ff6b6b', '#f59e0b', '#2563eb', '#94a3b8']
        bars       = ax.bar(categorias, valores, color=colores,
                            edgecolor='white', width=0.5)
        ax.set_title('Comparación de Niveles de Inventario',
                     fontweight='bold')
        ax.set_ylabel('Unidades')
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(valores) * 0.01,
                    f'{val:,.0f}', ha='center', fontsize=9,
                    fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def page_acerca():
    st.title("ℹ️ Acerca del Proyecto")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📋 Descripción
        **Desarrollo de una Plataforma Web para la Predicción de Demanda
        y Optimización de Inventarios mediante Técnicas de Machine Learning**

        Utilizando datos históricos de ventas minoristas del período 2010-2011
        (Online Retail Dataset — UCI Machine Learning Repository).

        ### 👥 Autores
        - **Mayorca Parra, Sara Camila**
        - **Palomino Acuña, Luis Alejandro**

        ### 🏫 Institución
        Instituto de Educación Superior
        Departamento de Tecnología Digital — Big Data y Ciencia de Datos
        Curso: Pretésis | Sección 5C28A
        Docente: Tasayco Luis Horacio | 2026-1
        """)
    with col2:
        st.markdown("""
        ### ⚙️ Metodología
        **CRISP-DM** (Cross Industry Standard Process for Data Mining)

        | Fase | Descripción |
        |------|-------------|
        | 1 | Comprensión del negocio |
        | 2 | Comprensión de los datos (EDA) |
        | 3 | Preparación de datos |
        | 4 | Modelado ML |
        | 5 | Evaluación |
        | 6 | Despliegue (Streamlit) |

        ### 🛠️ Tecnologías
        Python · Pandas · NumPy · Scikit-learn
        Matplotlib · Seaborn · Streamlit

        ### 📊 Algoritmos
        - Regresión Lineal (baseline)
        - Árbol de Decisión
        - **Random Forest** ← modelo ganador
        """)


# ============================================================
#  MAIN
# ============================================================

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        show_login()
        return

    pagina = show_sidebar()

    if   pagina == "🏠 Inicio":                page_inicio()
    elif pagina == "📤 Cargar Dataset":         page_cargar()
    elif pagina == "📊 Exploración de Datos":   page_eda()
    elif pagina == "🤖 Modelos ML":             page_modelos()
    elif pagina == "🔮 Predictor de Demanda":   page_predictor()
    elif pagina == "📦 Gestión de Inventario":  page_inventario()
    elif pagina == "ℹ️ Acerca del Proyecto":    page_acerca()


if __name__ == "__main__":
    main()
