# ==============================================================================
# app.py — Perfil Financiero 360
# Aplicación de evaluación crediticia con DBSCAN
# Dataset: Give Me Some Credit — Kaggle
# Universidad: Fundación Universitaria Los Libertadores
# Asignatura: Inteligencia Artificial
# Estudiantes: Mateo Bermejo / Ronald Gómez / Adrián Murcia
# ==============================================================================
# Ejecución:
#   streamlit run app/app.py
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# ------------------------------------------------------------------------------
# CONFIGURACIÓN INICIAL DE STREAMLIT
# Se debe llamar set_page_config antes que cualquier otro comando de Streamlit.
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Perfil Financiero 360 — Evaluación Crediticia DBSCAN",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------------------
# RUTAS ROBUSTAS
# Se construyen rutas relativas desde la ubicación de este archivo (app/app.py),
# de modo que la app funcione correctamente desde cualquier directorio de trabajo.
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "dbscan_model.pkl")
SCALER_PATH  = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
INFO_PATH    = os.path.join(PROJECT_ROOT, "models", "model_info.pkl")
CSS_PATH     = os.path.join(BASE_DIR, "templates\styles.css")

# ------------------------------------------------------------------------------
# CARGA DEL CSS EXTERNO
# El archivo styles.css está en app/styles.css junto a este script.
# ------------------------------------------------------------------------------
def cargar_css(ruta: str) -> None:
    """Carga el archivo CSS y lo inyecta en la página mediante st.markdown."""
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"Archivo CSS no encontrado: {ruta}")

cargar_css(CSS_PATH)

# ------------------------------------------------------------------------------
# CARGA DE ARCHIVOS SERIALIZADOS (.pkl)
# Se envuelven en bloques try/except para que la app no colapse si los archivos
# no existen todavía (por ejemplo, antes de ejecutar el notebook de entrenamiento).
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cargar_modelos():
    """
    Carga el modelo DBSCAN, el scaler y la información del modelo desde disco.
    Retorna una tupla (modelo, scaler, info, errores).
    """
    modelo, scaler, info = None, None, {}
    errores = []

    if os.path.exists(MODEL_PATH):
        try:
            modelo = joblib.load(MODEL_PATH)
        except Exception as e:
            errores.append(f"Error al cargar modelo: {e}")
    else:
        errores.append(f"Modelo no encontrado: {MODEL_PATH}")

    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            errores.append(f"Error al cargar scaler: {e}")
    else:
        errores.append(f"Scaler no encontrado: {SCALER_PATH}")

    if os.path.exists(INFO_PATH):
        try:
            info = joblib.load(INFO_PATH)
        except Exception as e:
            errores.append(f"Error al cargar info del modelo: {e}")
    else:
        errores.append(f"Info del modelo no encontrada: {INFO_PATH}")

    return modelo, scaler, info, errores


modelo, scaler, model_info, errores_carga = cargar_modelos()

# ------------------------------------------------------------------------------
# COLUMNAS DEL MODELO (orden exacto usado durante el entrenamiento)
# Este orden debe coincidir con el que se usó al llamar a scaler.fit_transform()
# en el notebook. Si el model_info contiene la lista de columnas, se usa esa;
# en caso contrario se define explícitamente aquí como respaldo.
# ------------------------------------------------------------------------------
COLUMNAS_MODELO = model_info.get("columns", [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
])

# ------------------------------------------------------------------------------
# TABLA DE VARIABLES: nombres técnicos, entendibles y descripciones
# ------------------------------------------------------------------------------
VARIABLES_INFO = [
    {
        "Columna técnica": "RevolvingUtilizationOfUnsecuredLines",
        "Nombre entendible": "Uso de crédito disponible",
        "Descripción financiera": (
            "Porcentaje de uso de líneas de crédito no aseguradas "
            "(tarjetas, créditos personales sin garantía). "
            "Valores altos indican que el cliente usa casi todo su cupo disponible."
        ),
    },
    {
        "Columna técnica": "age",
        "Nombre entendible": "Edad del cliente",
        "Descripción financiera": "Edad en años de la persona evaluada.",
    },
    {
        "Columna técnica": "NumberOfTime30-59DaysPastDueNotWorse",
        "Nombre entendible": "Moras de 30 a 59 días",
        "Descripción financiera": (
            "Número de veces que el cliente se atrasó entre 30 y 59 días "
            "en sus pagos durante los últimos dos años."
        ),
    },
    {
        "Columna técnica": "DebtRatio",
        "Nombre entendible": "Nivel de endeudamiento",
        "Descripción financiera": (
            "Relación entre las obligaciones financieras del cliente y su "
            "capacidad de pago. Valores elevados reflejan mayor carga financiera."
        ),
    },
    {
        "Columna técnica": "MonthlyIncome",
        "Nombre entendible": "Ingreso mensual",
        "Descripción financiera": "Ingreso mensual reportado por el cliente (USD).",
    },
    {
        "Columna técnica": "NumberOfOpenCreditLinesAndLoans",
        "Nombre entendible": "Créditos y líneas abiertas",
        "Descripción financiera": (
            "Cantidad de préstamos o líneas de crédito activas que tiene el cliente en este momento."
        ),
    },
    {
        "Columna técnica": "NumberOfTimes90DaysLate",
        "Nombre entendible": "Moras de 90 días o más",
        "Descripción financiera": (
            "Número de veces que el cliente presentó atrasos graves de 90 días o más. "
            "Es el indicador de riesgo más crítico del dataset."
        ),
    },
    {
        "Columna técnica": "NumberRealEstateLoansOrLines",
        "Nombre entendible": "Créditos hipotecarios o inmobiliarios",
        "Descripción financiera": (
            "Cantidad de préstamos o líneas de crédito relacionadas con vivienda o bienes raíces."
        ),
    },
    {
        "Columna técnica": "NumberOfTime60-89DaysPastDueNotWorse",
        "Nombre entendible": "Moras de 60 a 89 días",
        "Descripción financiera": (
            "Número de veces que el cliente se atrasó entre 60 y 89 días en sus pagos."
        ),
    },
    {
        "Columna técnica": "NumberOfDependents",
        "Nombre entendible": "Dependientes económicos",
        "Descripción financiera": (
            "Número de personas que dependen económicamente del cliente "
            "(hijos, cónyuge sin ingresos, etc.)."
        ),
    },
]

# ------------------------------------------------------------------------------
# FUNCIÓN: ASIGNACIÓN DE CLUSTER PARA NUEVOS CLIENTES
#
# NOTA ACADÉMICA:
# DBSCAN de scikit-learn NO tiene método .predict() para nuevos datos.
# Esto es una limitación conocida del algoritmo: DBSCAN es un método transductivo,
# es decir, solo asigna etiquetas a los datos con los que fue entrenado.
#
# Para asignar un cluster a un nuevo cliente, se implementa la siguiente estrategia:
#   1. Se escalan los datos del nuevo cliente con el mismo scaler usado en entrenamiento.
#   2. Se calcula la distancia euclidiana entre el nuevo punto escalado y todos los
#      puntos del conjunto de entrenamiento que tienen etiqueta conocida (!= -1).
#   3. Si la distancia mínima es menor que el parámetro eps del modelo, se asigna
#      la etiqueta del punto más cercano.
#   4. Si no existe ningún punto dentro del radio eps, se considera que el cliente
#      no pertenece a ningún cluster y se le asigna la etiqueta -1 (ruido/atípico).
#
# LIMITACIÓN: este enfoque requiere guardar los datos de entrenamiento escalados.
# Si no están disponibles, se usa una función de puntuación de riesgo basada en
# las variables ingresadas (ver función calcular_riesgo_por_reglas).
# ------------------------------------------------------------------------------

def predecir_cluster_por_distancia(punto_escalado: np.ndarray, modelo, eps: float) -> int:
    """
    Asigna un cluster a un nuevo punto usando la distancia al vecino más cercano
    del conjunto de entrenamiento.

    Parámetros:
        punto_escalado: array (1, n_features) ya escalado con scaler.transform().
        modelo: modelo DBSCAN entrenado con scikit-learn.
        eps: radio máximo del modelo DBSCAN.

    Retorna:
        Etiqueta del cluster (int). Retorna -1 si el punto es atípico.
    """
    # Los puntos de entrenamiento están almacenados en modelo.components_
    # (solo los puntos núcleo). Si están vacíos, se retorna -1.
    if not hasattr(modelo, "components_") or len(modelo.components_) == 0:
        return -1

    # Distancias euclidianas entre el nuevo punto y todos los puntos núcleo
    distancias = np.linalg.norm(modelo.components_ - punto_escalado, axis=1)
    idx_cercano = np.argmin(distancias)
    distancia_minima = distancias[idx_cercano]

    if distancia_minima <= eps:
        # El índice en components_ corresponde a core_sample_indices_
        idx_entrenamiento = modelo.core_sample_indices_[idx_cercano]
        return int(modelo.labels_[idx_entrenamiento])
    else:
        return -1  # Punto fuera de cualquier región densa → atípico


def calcular_riesgo_por_reglas(valores: dict) -> int:
    """
    Función de respaldo: estima el perfil del cliente usando reglas de negocio
    cuando no es posible usar la distancia al conjunto de entrenamiento.

    Esta función NO usa el modelo DBSCAN directamente, sino criterios derivados
    del análisis exploratorio del dataset Give Me Some Credit.

    Retorna un código numérico que se traduce al perfil de riesgo:
        0 → riesgo bajo (cluster viables del modelo)
        1 → riesgo alto
        -1 → atípico (perfil extremo)
    """
    moras_90     = valores.get("NumberOfTimes90DaysLate", 0)
    moras_3059   = valores.get("NumberOfTime30-59DaysPastDueNotWorse", 0)
    moras_6089   = valores.get("NumberOfTime60-89DaysPastDueNotWorse", 0)
    deuda        = valores.get("DebtRatio", 0)
    utilizacion  = valores.get("RevolvingUtilizationOfUnsecuredLines", 0)
    ingreso      = valores.get("MonthlyIncome", 0)

    total_moras = moras_90 + moras_3059 + moras_6089

    # Perfil atípico: combinación extrema de variables
    if utilizacion >= 0.99 and moras_90 >= 3:
        return -1

    # Riesgo alto
    if moras_90 >= 2 or total_moras >= 4 or deuda > 500:
        return 1

    # Riesgo bajo
    if total_moras == 0 and deuda < 0.5 and utilizacion < 0.5 and ingreso > 3000:
        return 0

    # Riesgo moderado (mapeado al cluster 2 que en el modelo es "alto riesgo moderado")
    return 2


def obtener_interpretacion(cluster: int, valores: dict) -> dict:
    """
    Convierte la etiqueta de cluster en una interpretación de negocio.

    Retorna un diccionario con:
        - perfil: texto corto del perfil
        - decision: recomendación de negocio
        - nivel: badge visual (bajo / moderado / alto / revision)
        - color_clase: clase CSS del resultado
        - emoji: ícono representativo
        - explicacion: párrafo en lenguaje de negocio
    """
    moras_90    = valores.get("NumberOfTimes90DaysLate", 0)
    moras_3059  = valores.get("NumberOfTime30-59DaysPastDueNotWorse", 0)
    moras_6089  = valores.get("NumberOfTime60-89DaysPastDueNotWorse", 0)
    deuda       = valores.get("DebtRatio", 0)
    ingreso     = valores.get("MonthlyIncome", 0)
    utilizacion = valores.get("RevolvingUtilizationOfUnsecuredLines", 0)
    total_moras = moras_90 + moras_3059 + moras_6089

    if cluster == -1:
        return {
            "perfil": "Cliente atípico: requiere revisión manual",
            "decision": "Revisión manual",
            "nivel": "revision",
            "color_clase": "resultado-revision",
            "emoji": "",
            "explicacion": (
                "El perfil financiero de este cliente no coincide con ninguno de los grupos "
                "identificados por el modelo DBSCAN. Esto puede indicar valores inusuales en "
                "alguna variable o una combinación poco frecuente de características. "
                "Se recomienda que un analista financiero revise el caso de forma individual "
                "antes de tomar una decisión de crédito."
            ),
        }

    # Determinar perfil por variables de riesgo
    if total_moras == 0 and deuda < 0.5 and utilizacion < 0.5 and ingreso >= 3000:
        return {
            "perfil": "Cliente apto para crédito",
            "decision": "Aprobación recomendada",
            "nivel": "bajo",
            "color_clase": "resultado-apto",
            "emoji": "",
            "explicacion": (
                "El cliente presenta un perfil financiero sólido: sin historial de moras, "
                "nivel de endeudamiento controlado y utilización responsable del crédito disponible. "
                "La entidad financiera puede considerar la aprobación del crédito con condiciones estándar. "
                "Este cliente también representa un perfil atractivo para oferta de nuevos productos financieros."
            ),
        }

    if moras_90 >= 2 or total_moras >= 4 or deuda > 500:
        return {
            "perfil": "Cliente no apto por alto riesgo",
            "decision": "No aprobación recomendada",
            "nivel": "alto",
            "color_clase": "resultado-no-apto",
            "emoji": "",
            "explicacion": (
                "El cliente presenta indicadores de alto riesgo crediticio: historial de moras graves, "
                "nivel de endeudamiento elevado o combinación de variables que sugieren dificultades "
                "para cumplir con nuevas obligaciones financieras. "
                "La entidad financiera no recomienda la aprobación del crédito en las condiciones actuales. "
                "Se sugiere invitar al cliente a un programa de acompañamiento financiero para mejorar su perfil."
            ),
        }

    return {
        "perfil": "Cliente apto con revisión adicional",
        "decision": "Riesgo moderado",
        "nivel": "moderado",
        "color_clase": "resultado-revision-adicional",
        "emoji": "",
        "explicacion": (
            "El cliente presenta algunas señales de riesgo moderado: puede tener una o dos moras leves, "
            "nivel de endeudamiento medio o utilización media del crédito disponible. "
            "La entidad financiera puede considerar la aprobación con condiciones adicionales, "
            "como un monto reducido, mayor tasa de interés o solicitud de garantías. "
            "Se recomienda una revisión adicional del historial crediticio antes de la decisión final."
        ),
    }


# ------------------------------------------------------------------------------
# BARRA DE NAVEGACIÓN SUPERIOR
# Se implementa con HTML/CSS inyectado para simular una navbar tipo fintech.
# ------------------------------------------------------------------------------
st.markdown("""
<nav class="navbar">
  <div class="navbar-brand">
    <span class="navbar-title">Perfil Financiero 360</span>
  </div>
  <div class="navbar-links">
    <a href="#inicio">Inicio</a>
    <a href="#simulador">Simulador</a>
    <a href="#resultados-modelo">Resultados del Modelo</a>
    <a href="#interpretacion">Interpretación</a>
    <a href="#conclusiones">Conclusiones</a>
  </div>
</nav>
""", unsafe_allow_html=True)

# Mostrar errores de carga si los hay (sin interrumpir la app)
if errores_carga:
    with st.expander("Advertencias de carga de archivos (.pkl)", expanded=False):
        for e in errores_carga:
            st.warning(e)
        st.info(
            "Para habilitar el simulador completo, ejecuta primero el notebook de entrenamiento "
            "para generar los archivos en la carpeta `models/`."
        )

# ==============================================================================
# SECCIÓN 1: HERO / PORTADA
# ==============================================================================
st.markdown('<div id="inicio"></div>', unsafe_allow_html=True)

import streamlit as st
from textwrap import dedent

st.markdown(
    dedent("""
    <div class="hero-section">
    <div class="hero-content">

    <div class="hero-badge">Plataforma de Análisis Crediticio</div>

    <h1 class="hero-title">Perfil Financiero 360</h1>

    <p class="hero-subtitle">
    Simulador financiero para análisis de riesgo crediticio
    </p>

    <p class="hero-description">
    Evalúa perfiles crediticios, identifica riesgos y detecta clientes con potencial financiero
    para apoyar decisiones de crédito más seguras, rápidas y estratégicas.
    </p>

    <div class="hero-buttons">
    <a href="#simulador" class="btn-primary">Evaluar cliente</a>
    <a href="#resultados-modelo" class="btn-secondary">Ver resultados del modelo</a>
    </div>

    </div>
    </div>
    """),
    unsafe_allow_html=True
)

# # Tarjetas de métricas rápidas
# col1, col2, col3, col4 = st.columns(4)
# clusters_n   = model_info.get("clusters", "N/A")
# outliers_n   = model_info.get("outliers", "N/A")
# silhouette   = model_info.get("silhouette_score")
# davies       = model_info.get("davies_bouldin_score")

# with col1:
#     st.markdown(f"""
#     <div class="metric-card">
#       <div class="metric-icon">🔵</div>
#       <div class="metric-value">{clusters_n}</div>
#       <div class="metric-label">Clusters identificados</div>
#     </div>""", unsafe_allow_html=True)

# with col2:
#     st.markdown(f"""
#     <div class="metric-card">
#       <div class="metric-icon">🔴</div>
#       <div class="metric-value">{outliers_n}</div>
#       <div class="metric-label">Clientes atípicos</div>
#     </div>""", unsafe_allow_html=True)

# with col3:
#     sil_val = f"{silhouette:.4f}" if silhouette is not None else "N/A"
#     st.markdown(f"""
#     <div class="metric-card">
#       <div class="metric-icon">📐</div>
#       <div class="metric-value">{sil_val}</div>
#       <div class="metric-label">Silhouette Score</div>
#     </div>""", unsafe_allow_html=True)

# with col4:
#     db_val = f"{davies:.4f}" if davies is not None else "N/A"
#     st.markdown(f"""
#     <div class="metric-card">
#       <div class="metric-icon">📏</div>
#       <div class="metric-value">{db_val}</div>
#       <div class="metric-label">Davies-Bouldin Score</div>
#     </div>""", unsafe_allow_html=True)


# ==============================================================================
# SECCIÓN 3: SIMULADOR DE EVALUACIÓN CREDITICIA
# ==============================================================================
# ==============================================================================
# SECCIÓN 3: SIMULADOR DE EVALUACIÓN CREDITICIA
# ==============================================================================

st.markdown("---")
st.markdown('<div id="simulador"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-header">
  <h2>Simulador de Evaluación Crediticia</h2>
  <p>Ingrese los datos del cliente para obtener una evaluación basada en el modelo DBSCAN</p>
</div>
""", unsafe_allow_html=True)

col_form1, col_form2 = st.columns(2)

with col_form1:
    st.markdown("#### Información personal y financiera")

    edad = st.slider(
        "Edad del cliente (años)",
        min_value=18,
        max_value=100,
        value=40,
        step=1,
        help="Edad actual del cliente en años cumplidos.",
    )

    ingreso_mensual = st.number_input(
        "Ingreso mensual (USD)",
        min_value=0,
        max_value=500000,
        value=5000,
        step=100,
        help="Ingreso mensual bruto reportado por el cliente.",
    )

    dependientes = st.slider(
        "Dependientes económicos",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help="Número de personas que dependen económicamente del cliente.",
    )

    uso_credito = st.slider(
        "Uso de crédito disponible (0.0 – 1.0)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01,
        help="Proporción del crédito disponible que el cliente usa actualmente. 1.0 = 100% utilizado.",
    )

    nivel_deuda = st.number_input(
        "Nivel de endeudamiento (DebtRatio)",
        min_value=0.0,
        max_value=10000.0,
        value=0.35,
        step=0.01,
        help="Relación entre deudas y capacidad de pago. Valores mayores a 1 pueden indicar sobreendeudamiento.",
    )

with col_form2:
    st.markdown("#### Historial crediticio")

    creditos_abiertos = st.slider(
        "Créditos y líneas abiertas",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
        help="Cantidad de préstamos o líneas de crédito activas.",
    )

    creditos_inmobiliarios = st.slider(
        "Créditos hipotecarios o inmobiliarios",
        min_value=0,
        max_value=30,
        value=1,
        step=1,
        help="Número de préstamos relacionados con vivienda o bienes raíces.",
    )

    st.markdown("##### Historial de moras")

    moras_30_59 = st.slider(
        "Moras de 30 a 59 días",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help="Número de veces con atrasos entre 30 y 59 días en los últimos 2 años.",
    )

    moras_60_89 = st.slider(
        "Moras de 60 a 89 días",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help="Número de veces con atrasos entre 60 y 89 días.",
    )

    moras_90 = st.slider(
        "Moras de 90 días o más",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help="Número de veces con atrasos graves de 90 días o más. Es uno de los indicadores de mayor riesgo.",
    )

# Botón de evaluación
col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
with col_btn2:
    evaluar = st.button("🔍 Evaluar cliente", use_container_width=True, type="primary")

# ==============================================================================
# LÓGICA DE EVALUACIÓN
# Se ejecuta cuando el usuario hace clic en "Evaluar cliente".
# ==============================================================================
if evaluar:
    # 1. Crear diccionario con los valores ingresados (claves = nombres técnicos)
    valores_cliente = {
        "RevolvingUtilizationOfUnsecuredLines": uso_credito,
        "age": edad,
        "NumberOfTime30-59DaysPastDueNotWorse": moras_30_59,
        "DebtRatio": nivel_deuda,
        "MonthlyIncome": ingreso_mensual,
        "NumberOfOpenCreditLinesAndLoans": creditos_abiertos,
        "NumberOfTimes90DaysLate": moras_90,
        "NumberRealEstateLoansOrLines": creditos_inmobiliarios,
        "NumberOfTime60-89DaysPastDueNotWorse": moras_60_89,
        "NumberOfDependents": dependientes,
    }

    # 2. Crear DataFrame con columnas en el orden exacto del modelo
    df_cliente = pd.DataFrame([valores_cliente])[COLUMNAS_MODELO]

    # 3. Escalar los datos y asignar cluster
    cluster_asignado = None
    metodo_usado = ""

    if scaler is not None and modelo is not None:
        try:
            # Escalar con el mismo scaler del entrenamiento
            X_nuevo_escalado = scaler.transform(df_cliente)

            # Intentar asignación por distancia a puntos núcleo
            eps_modelo = model_info.get("eps", 2.0)
            cluster_asignado = predecir_cluster_por_distancia(
                X_nuevo_escalado, modelo, eps_modelo
            )
            metodo_usado = "Distancia a puntos núcleo DBSCAN"
        except Exception as ex:
            st.warning(f"No fue posible usar el modelo DBSCAN directamente: {ex}")
            cluster_asignado = calcular_riesgo_por_reglas(valores_cliente)
            metodo_usado = "Función de reglas de negocio (respaldo)"
    else:
        # Modelo o scaler no disponibles: usar función de respaldo
        cluster_asignado = calcular_riesgo_por_reglas(valores_cliente)
        metodo_usado = "Función de reglas de negocio (modelos .pkl no disponibles)"

    # 4. Obtener interpretación de negocio
    interpretacion = obtener_interpretacion(cluster_asignado, valores_cliente)

    # 5. Mostrar resultados
    st.markdown("---")
    st.markdown('<div id="resultado-evaluacion"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <h2>📋 Resultado de la Evaluación</h2>
    </div>
    """, unsafe_allow_html=True)

    nivel = interpretacion["nivel"]
    clase_css = interpretacion["color_clase"]

    st.markdown(f"""
    <div class="resultado-card {clase_css}">
      <div class="resultado-emoji">{interpretacion["emoji"]}</div>
      <div class="resultado-perfil">{interpretacion["perfil"]}</div>
      <div class="resultado-decision">
        <span class="badge-decision badge-{nivel}">{interpretacion["decision"]}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.markdown(f"""
        <div class="info-resultado-card">
          <div class="info-resultado-label">Cluster asignado</div>
          <div class="info-resultado-value">{cluster_asignado}</div>
        </div>""", unsafe_allow_html=True)
    with col_r2:
        st.markdown(f"""
        <div class="info-resultado-card">
          <div class="info-resultado-label">Nivel de riesgo</div>
          <div class="info-resultado-value nivel-{nivel}">{nivel.upper()}</div>
        </div>""", unsafe_allow_html=True)
    with col_r3:
        st.markdown(f"""
        <div class="info-resultado-card">
          <div class="info-resultado-label">Método de asignación</div>
          <div class="info-resultado-value-small">{metodo_usado}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="explicacion-box">
      <h4>💬 Recomendación financiera</h4>
      <p>{interpretacion["explicacion"]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
      ⚠️ <strong>Aviso importante:</strong> Este resultado es un apoyo analítico generado 
      por un modelo de machine learning y <strong>no reemplaza la decisión final de un analista 
      financiero certificado</strong>. La entidad financiera debe complementar este análisis con 
      verificación documental, consulta en centrales de riesgo y criterio profesional.
    </div>
    """, unsafe_allow_html=True)

    # Mostrar resumen de variables ingresadas
    with st.expander("📊 Resumen de variables ingresadas", expanded=False):
        df_resumen = pd.DataFrame({
            "Variable": [v["Nombre entendible"] for v in VARIABLES_INFO],
            "Valor ingresado": [
                uso_credito, edad, moras_30_59, nivel_deuda,
                ingreso_mensual, creditos_abiertos, moras_90,
                creditos_inmobiliarios, moras_60_89, dependientes,
            ],
        })
        st.dataframe(df_resumen, use_container_width=True, hide_index=True)



# ==============================================================================
# SECCIÓN 5: RESULTADOS DEL MODELO
# ==============================================================================
st.markdown("---")
st.markdown('<div id="resultados-modelo"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-header">
  <h2>Resultados del Modelo DBSCAN</h2>
  <p>Métricas de entrenamiento, parámetros y archivos serializados</p>
</div>
""", unsafe_allow_html=True)

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("""
    <div class="model-info-card">
      <h4>Información del modelo</h4>
    </div>
    """, unsafe_allow_html=True)

    algoritmo   = model_info.get("algorithm", "DBSCAN")
    dataset_nm  = model_info.get("dataset", "Give Me Some Credit")
    fuente      = model_info.get("source", "Kaggle — https://www.kaggle.com/c/GiveMeSomeCredit")
    eps_val     = model_info.get("eps", "N/A")
    min_s_val   = model_info.get("min_samples", "N/A") if hasattr(modelo, "min_samples") else model_info.get("min_samples", "N/A")

    info_tabla = {
        "Parámetro": [
            "Algoritmo", "Dataset", "Fuente",
            "Parámetro eps", "Parámetro min_samples",
            "Número de clusters", "Clientes atípicos (outliers)",
            "Porcentaje atípicos",
        ],
        "Valor": [
            algoritmo, dataset_nm, fuente,
            str(eps_val), str(min_s_val),
            str(model_info.get("clusters", "N/A")),
            str(model_info.get("outliers", "N/A")),
            f"{model_info.get('outlier_percentage', 'N/A')}%",
        ],
    }
    st.dataframe(pd.DataFrame(info_tabla), hide_index=True, use_container_width=True)

with col_m2:
    st.markdown("""
    <div class="model-info-card">
      <h4>Métricas de evaluación</h4>
    </div>
    """, unsafe_allow_html=True)

    sil = model_info.get("silhouette_score")
    dav = model_info.get("davies_bouldin_score")

    sil_str = f"{sil:.6f}" if sil is not None else "No disponible"
    dav_str = f"{dav:.6f}" if dav is not None else "No disponible"

    st.markdown(f"""
    <div class="metric-detail-card">
      <div class="metric-detail-name">Silhouette Score</div>
      <div class="metric-detail-value">{sil_str}</div>
      <div class="metric-detail-desc">
        Mide la cohesión interna de los clusters. Valores entre 0 y 1 son positivos; 
        mayor valor indica mejor separación entre grupos.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-detail-card">
      <div class="metric-detail-name"> Davies-Bouldin Score</div>
      <div class="metric-detail-value">{dav_str}</div>
      <div class="metric-detail-desc">
        Mide la similitud promedio entre clusters. Valores más bajos indican mejor 
        separación. Menor es mejor.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if sil is None or dav is None:
        st.markdown("""
        <div class="warning-box">
          ⚠️ <strong>¿Por qué algunas métricas aparecen como None?</strong><br>
          DBSCAN puede no calcular el Silhouette Score ni el Davies-Bouldin Score cuando 
          encuentra menos de dos clusters válidos (excluyendo el ruido etiquetado como -1). 
          En ese caso, scikit-learn no puede calcular métricas de separación entre grupos. 
          Esto no significa que el modelo falle; significa que los datos tienen una densidad 
          particular que DBSCAN interpreta como un único grupo compacto con algunos outliers.
        </div>
        """, unsafe_allow_html=True)

# Archivos serializados
# st.markdown("""
# <div class="files-card">
#   <h4>🗂️ Archivos serializados</h4>
#   <div class="files-grid">
#     <div class="file-item">
#       <span class="file-icon">📦</span>
#       <code>models/dbscan_model.pkl</code>
#       <span class="file-desc">Modelo DBSCAN entrenado con scikit-learn</span>
#     </div>
#     <div class="file-item">
#       <span class="file-icon">⚖️</span>
#       <code>models/scaler.pkl</code>
#       <span class="file-desc">StandardScaler ajustado sobre los datos de entrenamiento</span>
#     </div>
#     <div class="file-item">
#       <span class="file-icon">📋</span>
#       <code>models/model_info.pkl</code>
#       <span class="file-desc">Diccionario con métricas, parámetros e interpretación de clusters</span>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

# Perfiles de negocio por cluster
business_profiles = model_info.get("business_profiles", [])
if business_profiles:
    st.markdown("#### Perfiles de negocio")
    df_perfiles = pd.DataFrame(business_profiles)
    # Formatear columnas numéricas
    cols_pct = ["tasa_morosidad_grave", "utilizacion_credito_promedio"]
    for c in cols_pct:
        if c in df_perfiles.columns:
            df_perfiles[c] = df_perfiles[c].apply(lambda x: f"{x:.2%}")
    cols_round = ["edad_promedio", "ingreso_promedio", "deuda_promedio", "moras_90_dias_promedio"]
    for c in cols_round:
        if c in df_perfiles.columns:
            df_perfiles[c] = df_perfiles[c].apply(lambda x: f"{x:.2f}")
    st.dataframe(df_perfiles, use_container_width=True, hide_index=True)

# ==============================================================================
# SECCIÓN 6: INTERPRETACIÓN FINANCIERA
# ==============================================================================
st.markdown("---")
st.markdown('<div id="interpretacion"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-header">
  <h2>Interpretación Financiera</h2>
  <p>Cómo el banco usa el análisis DBSCAN para tomar mejores decisiones</p>
</div>
""", unsafe_allow_html=True)

col_i1, col_i2 = st.columns(2)

with col_i1:
    st.markdown("""
    <div class="interpretation-card">
      <h4>¿Qué hace DBSCAN en este contexto?</h4>
      <p>
        DBSCAN agrupa automáticamente a clientes que tienen características financieras similares, 
        sin necesidad de definir previamente cuántos grupos existen. Esto es valioso porque 
        los patrones de riesgo crediticio <strong>emergen naturalmente de los datos</strong>, 
        sin sesgos impuestos por el analista.
      </p>
      <p>
        El algoritmo usa distancias entre perfiles financieros (escalados) para identificar 
        zonas densas de clientes similares. Los clientes que no encajan en ninguna zona densa 
        reciben la etiqueta <strong>-1 (atípicos o ruido)</strong>.
      </p>
    </div>
    """, unsafe_allow_html=True)


with col_i2:
    st.markdown("""
    <div class="interpretation-card">
      <h4> Valor para la entidad financiera</h4>
      <ul>
        <li>✅ <strong>Decisión de crédito:</strong> identificar si el cliente puede recibir crédito.</li>
        <li>🔁 <strong>Continuidad del cliente:</strong> evaluar si el cliente puede seguir siendo parte del portafolio.</li>
        <li>📦 <strong>Nuevos productos:</strong> determinar si el cliente es candidato para productos futuros.</li>
        <li>💰 <strong>Rentabilidad:</strong> priorizar clientes que generan valor sostenible para la entidad.</li>
        <li>🔍 <strong>Revisión manual:</strong> identificar casos que merecen análisis individual detallado.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)



# ==============================================================================
# SECCIÓN 7: CONCLUSIONES
# ==============================================================================
st.markdown("---")
st.markdown('<div id="conclusiones"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-header">
  <h2>Conclusiones</h2>
</div>
""", unsafe_allow_html=True)

col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown("""
    <div class="conclusion-card">
      <h4>Conclusiones académicas</h4>
      <ul>
        <li>Se implementó el algoritmo <strong>DBSCAN</strong> con scikit-learn sobre el dataset 
            real <em>Give Me Some Credit</em> de Kaggle, que supera los 150.000 registros y 10 variables.</li>
        <li>Se realizó limpieza de datos, tratamiento de valores nulos (mediana), tratamiento de 
            valores extremos (clipping) y escalado con <code>StandardScaler</code>.</li>
        <li>DBSCAN no requiere definir el número de clusters de antemano, lo que es adecuado 
            para datos de comportamiento financiero con patrones no esféricos.</li>
        <li>Se calcularon las métricas <strong>Silhouette Score</strong> y 
            <strong>Davies-Bouldin Score</strong> para evaluar la calidad del agrupamiento.</li>
        <li>El modelo y el scaler fueron serializados con <code>joblib</code> para ser 
            reutilizados en la aplicación sin necesidad de reentrenar.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

with col_c2:
    st.markdown("""
    <div class="conclusion-card">
      <h4>Conclusiones de negocio</h4>
      <ul>
        <li>El análisis DBSCAN permitió identificar <strong>4 clusters de clientes</strong> 
            con perfiles financieros diferenciados, más un grupo de clientes atípicos.</li>
        <li>El <strong>Cluster 0</strong> agrupa al mayor número de clientes con perfil viable: 
            sin moras graves, endeudamiento controlado y buen historial.</li>
        <li>Los <strong>Clusters 1, 2 y 3</strong> concentran perfiles de alto riesgo con 
            tasas de morosidad entre el 18% y el 55%.</li>
        <li>El <strong>Cluster -1</strong> (atípicos) tiene una tasa de morosidad del 36%, 
            lo que justifica la revisión manual antes de tomar decisiones.</li>
        <li>La aplicación <strong>Perfil Financiero 360</strong> traduce los resultados 
            técnicos del modelo a lenguaje de negocio comprensible para analistas financieros.</li>
     
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("""
<div class="footer">
  <div class="footer-content">
    <div class="footer-brand">Perfil Financiero 360</div>
    <div class="footer-info">
      Proyecto académico · Inteligencia Artificial · Fundación Universitaria Los Libertadores<br>
      Mateo Bermejo · Ronald Gómez · Adrián Murcia · 2026
    </div>
    <div class="footer-tech">
    </div>
  </div>
</div>
""", unsafe_allow_html=True)