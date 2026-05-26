import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# Configuración inicial de la aplicación
# ============================================================

st.set_page_config(
    page_title="Segmentación de Clientes con DBSCAN",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Segmentación de Clientes de Tarjetas de Crédito")
st.subheader("Modelo de Machine Learning con DBSCAN")

st.write("""
Esta aplicación permite analizar el comportamiento financiero de un cliente
y asignarlo a un grupo usando un modelo de clustering DBSCAN entrenado con el dataset CCData.
""")


# ============================================================
# Rutas de archivos
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "dbscan_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
INFO_PATH = os.path.join(BASE_DIR, "models", "model_info.pkl")


# ============================================================
# Carga del modelo, scaler e información
# ============================================================

@st.cache_resource
def cargar_archivos():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_info = joblib.load(INFO_PATH)
    return model, scaler, model_info


try:
    model, scaler, model_info = cargar_archivos()
    st.success("Modelo DBSCAN cargado correctamente.")
except Exception as e:
    st.error("No se pudieron cargar los archivos del modelo.")
    st.error(f"Detalle del error: {e}")
    st.stop()


# ============================================================
# Información general del modelo
# ============================================================

with st.expander("Ver información del modelo"):
    st.write("**Algoritmo:**", model_info.get("algorithm", "DBSCAN"))
    st.write("**Dataset:**", model_info.get("dataset", "CCData"))
    st.write("**Clusters encontrados:**", model_info.get("clusters", "No disponible"))
    st.write("**Clientes atípicos detectados:**", model_info.get("outliers", "No disponible"))
    st.write("**Silhouette Score:**", model_info.get("silhouette_score", "No disponible"))
    st.write("**Davies-Bouldin Score:**", model_info.get("davies_bouldin_score", "No disponible"))


# ============================================================
# Columnas utilizadas en el entrenamiento
# ============================================================

columnas_modelo = model_info.get("columns")

if columnas_modelo is None:
    columnas_modelo = [
        "BALANCE",
        "BALANCE_FREQUENCY",
        "PURCHASES",
        "ONEOFF_PURCHASES",
        "INSTALLMENTS_PURCHASES",
        "CASH_ADVANCE",
        "PURCHASES_FREQUENCY",
        "ONEOFF_PURCHASES_FREQUENCY",
        "PURCHASES_INSTALLMENTS_FREQUENCY",
        "CASH_ADVANCE_FREQUENCY",
        "CASH_ADVANCE_TRX",
        "PURCHASES_TRX",
        "CREDIT_LIMIT",
        "PAYMENTS",
        "MINIMUM_PAYMENTS",
        "PRC_FULL_PAYMENT",
        "TENURE"
    ]


# ============================================================
# Función para asignar cluster a un nuevo cliente
# ============================================================

def asignar_cluster_dbscan(model, nuevo_dato_escalado):
    """
    DBSCAN no tiene método predict() como los modelos supervisados.
    Por eso, esta función asigna el nuevo cliente al cluster del punto núcleo
    más cercano si está dentro del radio eps. Si no cumple, se marca como ruido (-1).
    """

    if not hasattr(model, "components_"):
        return -1

    if len(model.components_) == 0:
        return -1

    # Puntos núcleo encontrados por DBSCAN durante el entrenamiento
    core_samples = model.components_

    # Etiquetas de los puntos núcleo
    core_labels = model.labels_[model.core_sample_indices_]

    # Distancia entre el nuevo cliente y cada punto núcleo
    distancias = np.linalg.norm(core_samples - nuevo_dato_escalado, axis=1)

    # Punto núcleo más cercano
    indice_mas_cercano = np.argmin(distancias)
    distancia_minima = distancias[indice_mas_cercano]

    # Si está dentro del radio eps, se asigna al cluster correspondiente
    if distancia_minima <= model.eps:
        return core_labels[indice_mas_cercano]
    else:
        return -1


# ============================================================
# Formulario de entrada de datos
# ============================================================

st.markdown("---")
st.header("Ingresar información del cliente")

st.write("""
Complete los valores financieros del cliente para identificar su segmento.
""")

balance = st.number_input("Balance actual del cliente", min_value=0.0, value=1000.0)
balance_frequency = st.number_input("Frecuencia de balance", min_value=0.0, max_value=1.0, value=1.0)
purchases = st.number_input("Total de compras", min_value=0.0, value=500.0)
oneoff_purchases = st.number_input("Compras de una sola vez", min_value=0.0, value=200.0)
installments_purchases = st.number_input("Compras en cuotas", min_value=0.0, value=300.0)
cash_advance = st.number_input("Avances en efectivo", min_value=0.0, value=0.0)

purchases_frequency = st.number_input("Frecuencia de compras", min_value=0.0, max_value=1.0, value=0.5)
oneoff_purchases_frequency = st.number_input("Frecuencia de compras únicas", min_value=0.0, max_value=1.0, value=0.2)
purchases_installments_frequency = st.number_input("Frecuencia de compras en cuotas", min_value=0.0, max_value=1.0, value=0.3)
cash_advance_frequency = st.number_input("Frecuencia de avances en efectivo", min_value=0.0, max_value=1.0, value=0.0)

cash_advance_trx = st.number_input("Número de transacciones de avance", min_value=0, value=0)
purchases_trx = st.number_input("Número de transacciones de compra", min_value=0, value=10)
credit_limit = st.number_input("Límite de crédito", min_value=0.0, value=5000.0)
payments = st.number_input("Pagos realizados", min_value=0.0, value=800.0)
minimum_payments = st.number_input("Pagos mínimos", min_value=0.0, value=300.0)
prc_full_payment = st.number_input("Porcentaje de pago completo", min_value=0.0, max_value=1.0, value=0.0)
tenure = st.number_input("Meses de permanencia", min_value=1, max_value=12, value=12)


# ============================================================
# Preparación de los datos de entrada
# ============================================================

input_data = pd.DataFrame({
    "BALANCE": [balance],
    "BALANCE_FREQUENCY": [balance_frequency],
    "PURCHASES": [purchases],
    "ONEOFF_PURCHASES": [oneoff_purchases],
    "INSTALLMENTS_PURCHASES": [installments_purchases],
    "CASH_ADVANCE": [cash_advance],
    "PURCHASES_FREQUENCY": [purchases_frequency],
    "ONEOFF_PURCHASES_FREQUENCY": [oneoff_purchases_frequency],
    "PURCHASES_INSTALLMENTS_FREQUENCY": [purchases_installments_frequency],
    "CASH_ADVANCE_FREQUENCY": [cash_advance_frequency],
    "CASH_ADVANCE_TRX": [cash_advance_trx],
    "PURCHASES_TRX": [purchases_trx],
    "CREDIT_LIMIT": [credit_limit],
    "PAYMENTS": [payments],
    "MINIMUM_PAYMENTS": [minimum_payments],
    "PRC_FULL_PAYMENT": [prc_full_payment],
    "TENURE": [tenure]
})

# Asegurar el mismo orden de columnas usado en entrenamiento
input_data = input_data[columnas_modelo]


# ============================================================
# Predicción / asignación de cluster
# ============================================================

st.markdown("---")

if st.button("Analizar cliente"):
    input_scaled = scaler.transform(input_data)

    cluster = asignar_cluster_dbscan(model, input_scaled)

    st.subheader("Resultado del análisis")

    if cluster == -1:
        st.warning("El cliente fue identificado como atípico o fuera de los grupos principales.")
        st.write("""
        Esto significa que el comportamiento financiero ingresado no se parece lo suficiente
        a los grupos encontrados por DBSCAN durante el entrenamiento.
        """)
    else:
        st.success(f"El cliente pertenece al Cluster {cluster}.")
        st.write("""
        Esto significa que el cliente presenta un comportamiento similar al de otros clientes
        agrupados por el modelo DBSCAN.
        """)

    st.write("Datos ingresados:")
    st.dataframe(input_data)