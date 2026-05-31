import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Configuración general de Streamlit
# ============================================================

st.set_page_config(
    page_title="Riesgo Crediticio con DBSCAN",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Segmentación de Clientes para Análisis de Riesgo Crediticio")
st.write(
    """
    Aplicación web basada en el modelo **DBSCAN** entrenado en el notebook del proyecto.
    El objetivo es analizar el perfil financiero de un cliente y traducir el resultado
    del cluster en una conclusión útil para negocio.
    """
)


# ============================================================
# Rutas del proyecto
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "dbscan_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
INFO_PATH = os.path.join(BASE_DIR, "models", "model_info.pkl")


# ============================================================
# Carga de archivos serializados
# ============================================================

@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_info = joblib.load(INFO_PATH)
    return modelo, scaler, model_info


try:
    model, scaler, model_info = cargar_modelo()
    st.success("Modelo, scaler e información cargados correctamente.")
except Exception as error:
    st.error("No se pudieron cargar los archivos del modelo.")
    st.write("Verifica que existan estos archivos:")
    st.code(
        """
models/dbscan_model.pkl
models/scaler.pkl
models/model_info.pkl
        """
    )
    st.error(f"Detalle del error: {error}")
    st.stop()


# ============================================================
# Diccionario de nombres entendibles
# ============================================================

nombres_entendibles = model_info.get(
    "column_descriptions",
    {
        "RevolvingUtilizationOfUnsecuredLines": "Uso de crédito disponible",
        "age": "Edad del cliente",
        "NumberOfTime30-59DaysPastDueNotWorse": "Moras de 30 a 59 días",
        "DebtRatio": "Nivel de endeudamiento",
        "MonthlyIncome": "Ingreso mensual",
        "NumberOfOpenCreditLinesAndLoans": "Créditos y líneas abiertas",
        "NumberOfTimes90DaysLate": "Moras de 90 días o más",
        "NumberRealEstateLoansOrLines": "Créditos hipotecarios o inmobiliarios",
        "NumberOfTime60-89DaysPastDueNotWorse": "Moras de 60 a 89 días",
        "NumberOfDependents": "Dependientes económicos"
    }
)

descripciones = {
    "RevolvingUtilizationOfUnsecuredLines": "Porcentaje de uso de líneas de crédito no aseguradas, como tarjetas de crédito o créditos personales.",
    "age": "Edad del cliente.",
    "NumberOfTime30-59DaysPastDueNotWorse": "Cantidad de veces que el cliente tuvo retrasos de pago entre 30 y 59 días.",
    "DebtRatio": "Relación entre deuda y capacidad de pago del cliente.",
    "MonthlyIncome": "Ingreso mensual reportado por el cliente.",
    "NumberOfOpenCreditLinesAndLoans": "Cantidad de préstamos o líneas de crédito activas.",
    "NumberOfTimes90DaysLate": "Cantidad de veces que el cliente tuvo retrasos graves de 90 días o más.",
    "NumberRealEstateLoansOrLines": "Cantidad de préstamos inmobiliarios o líneas de crédito relacionadas con vivienda.",
    "NumberOfTime60-89DaysPastDueNotWorse": "Cantidad de veces que el cliente tuvo retrasos entre 60 y 89 días.",
    "NumberOfDependents": "Número de personas que dependen económicamente del cliente."
}

columnas_modelo = model_info.get(
    "columns",
    [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents"
    ]
)


# ============================================================
# Función para asignar cluster a un nuevo cliente
# ============================================================

def asignar_cluster_dbscan(modelo, dato_escalado):
    """
    DBSCAN en scikit-learn no tiene método predict().
    Esta función compara el nuevo cliente con los puntos núcleo del modelo.
    Si el cliente está dentro del radio eps de un punto núcleo, se asigna al cluster
    correspondiente. Si no, se clasifica como -1, es decir, cliente atípico.
    """

    if not hasattr(modelo, "components_") or len(modelo.components_) == 0:
        return -1

    puntos_nucleo = modelo.components_
    etiquetas_nucleo = modelo.labels_[modelo.core_sample_indices_]

    distancias = np.linalg.norm(puntos_nucleo - dato_escalado, axis=1)
    indice_cercano = np.argmin(distancias)
    distancia_minima = distancias[indice_cercano]

    if distancia_minima <= modelo.eps:
        return int(etiquetas_nucleo[indice_cercano])

    return -1


def obtener_interpretacion(cluster, model_info):
    """
    Obtiene la interpretación de negocio del cluster usando la información
    generada en el notebook.
    """

    perfiles = model_info.get("business_profiles", [])

    for perfil in perfiles:
        if int(perfil.get("Cluster")) == int(cluster):
            return {
                "interpretacion": perfil.get("interpretacion_negocio", "Sin interpretación disponible"),
                "tasa_morosidad": perfil.get("tasa_morosidad_grave", None),
                "edad_promedio": perfil.get("edad_promedio", None),
                "ingreso_promedio": perfil.get("ingreso_promedio", None),
                "deuda_promedio": perfil.get("deuda_promedio", None),
                "uso_credito_promedio": perfil.get("utilizacion_credito_promedio", None),
                "moras_90_promedio": perfil.get("moras_90_dias_promedio", None),
                "registros": perfil.get("registros", None)
            }

    if cluster == -1:
        return {
            "interpretacion": "Cliente atípico: requiere revisión manual",
            "tasa_morosidad": None,
            "edad_promedio": None,
            "ingreso_promedio": None,
            "deuda_promedio": None,
            "uso_credito_promedio": None,
            "moras_90_promedio": None,
            "registros": None
        }

    return {
        "interpretacion": "Cluster identificado, pero sin interpretación registrada",
        "tasa_morosidad": None,
        "edad_promedio": None,
        "ingreso_promedio": None,
        "deuda_promedio": None,
        "uso_credito_promedio": None,
        "moras_90_promedio": None,
        "registros": None
    }


# ============================================================
# Panel lateral con información del modelo
# ============================================================

with st.sidebar:
    st.header("Información del modelo")
    st.write("**Algoritmo:**", model_info.get("algorithm", "DBSCAN"))
    st.write("**Dataset:**", model_info.get("dataset", "Give Me Some Credit"))
    st.write("**Fuente:** Kaggle")
    st.write("**eps:**", model_info.get("eps", "No disponible"))
    st.write("**min_samples:**", model_info.get("min_samples", "No disponible"))
    st.write("**Clusters encontrados:**", model_info.get("clusters", "No disponible"))
    st.write("**Clientes atípicos:**", model_info.get("outliers", "No disponible"))
    st.write("**Porcentaje atípicos:**", f"{model_info.get('outlier_percentage', 'No disponible')} %")
    st.write("**Silhouette Score:**", round(model_info.get("silhouette_score", 0), 4))
    st.write("**Davies-Bouldin Score:**", round(model_info.get("davies_bouldin_score", 0), 4))

    st.markdown("---")
    st.write("**Archivos usados:**")
    st.code(
        """
models/dbscan_model.pkl
models/scaler.pkl
models/model_info.pkl
        """
    )


# ============================================================
# Explicación de atributos
# ============================================================

with st.expander("Ver descripción de los atributos"):
    tabla_atributos = pd.DataFrame({
        "Dato original": columnas_modelo,
        "Nombre entendible": [nombres_entendibles.get(col, col) for col in columnas_modelo],
        "Descripción": [descripciones.get(col, "Sin descripción registrada") for col in columnas_modelo]
    })
    st.dataframe(tabla_atributos, use_container_width=True)


# ============================================================
# Formulario de datos del cliente
# ============================================================

st.header("Ingresar información financiera del cliente")

st.write(
    """
    Complete los datos financieros del cliente. La aplicación escalará los valores con el mismo
    `StandardScaler` usado en el notebook y asignará el perfil al cluster más cercano del modelo DBSCAN.
    """
)

col1, col2 = st.columns(2)

with col1:
    uso_credito = st.number_input(
        "Uso de crédito disponible",
        min_value=0.0,
        max_value=5.0,
        value=0.30,
        step=0.01,
        help=descripciones["RevolvingUtilizationOfUnsecuredLines"]
    )

    edad = st.number_input(
        "Edad del cliente",
        min_value=18,
        max_value=110,
        value=45,
        step=1,
        help=descripciones["age"]
    )

    moras_30_59 = st.number_input(
        "Moras de 30 a 59 días",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help=descripciones["NumberOfTime30-59DaysPastDueNotWorse"]
    )

    nivel_endeudamiento = st.number_input(
        "Nivel de endeudamiento",
        min_value=0.0,
        max_value=5000.0,
        value=0.35,
        step=0.01,
        help=descripciones["DebtRatio"]
    )

    ingreso_mensual = st.number_input(
        "Ingreso mensual",
        min_value=0.0,
        max_value=300000.0,
        value=5500.0,
        step=100.0,
        help=descripciones["MonthlyIncome"]
    )

with col2:
    creditos_abiertos = st.number_input(
        "Créditos y líneas abiertas",
        min_value=0,
        max_value=80,
        value=8,
        step=1,
        help=descripciones["NumberOfOpenCreditLinesAndLoans"]
    )

    moras_90 = st.number_input(
        "Moras de 90 días o más",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help=descripciones["NumberOfTimes90DaysLate"]
    )

    creditos_hipotecarios = st.number_input(
        "Créditos hipotecarios o inmobiliarios",
        min_value=0,
        max_value=20,
        value=1,
        step=1,
        help=descripciones["NumberRealEstateLoansOrLines"]
    )

    moras_60_89 = st.number_input(
        "Moras de 60 a 89 días",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help=descripciones["NumberOfTime60-89DaysPastDueNotWorse"]
    )

    dependientes = st.number_input(
        "Dependientes económicos",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help=descripciones["NumberOfDependents"]
    )


# ============================================================
# Construcción del dataframe de entrada
# ============================================================

input_data = pd.DataFrame({
    "RevolvingUtilizationOfUnsecuredLines": [uso_credito],
    "age": [edad],
    "NumberOfTime30-59DaysPastDueNotWorse": [moras_30_59],
    "DebtRatio": [nivel_endeudamiento],
    "MonthlyIncome": [ingreso_mensual],
    "NumberOfOpenCreditLinesAndLoans": [creditos_abiertos],
    "NumberOfTimes90DaysLate": [moras_90],
    "NumberRealEstateLoansOrLines": [creditos_hipotecarios],
    "NumberOfTime60-89DaysPastDueNotWorse": [moras_60_89],
    "NumberOfDependents": [dependientes]
})

input_data = input_data[columnas_modelo]
input_data_presentacion = input_data.rename(columns=nombres_entendibles)


# ============================================================
# Resultado del análisis
# ============================================================

st.markdown("---")

if st.button("Analizar cliente"):
    input_scaled = scaler.transform(input_data)
    cluster = asignar_cluster_dbscan(model, input_scaled)

    resultado = obtener_interpretacion(cluster, model_info)
    interpretacion = resultado["interpretacion"]

    st.subheader("Resultado del análisis")

    if cluster == -1:
        st.warning("Cliente atípico: requiere revisión manual.")
    elif "riesgo bajo" in interpretacion.lower() or "viable" in interpretacion.lower():
        st.success(interpretacion)
    elif "moderado" in interpretacion.lower():
        st.info(interpretacion)
    else:
        st.error(interpretacion)

    st.write(f"**Cluster asignado:** {cluster}")

    st.markdown("### Interpretación para negocio")

    if cluster == -1:
        st.write(
            """
            El cliente no se parece lo suficiente a los grupos principales encontrados por DBSCAN.
            Esto no significa rechazo automático. En una entidad financiera, este caso debería pasar
            a revisión manual para analizar con más detalle su comportamiento financiero.
            """
        )
    else:
        st.write(
            """
            El cliente fue asignado a un grupo de comportamiento financiero similar dentro del dataset.
            La interpretación se basa en el perfil promedio del cluster y en la tasa de morosidad grave
            usada como variable de referencia.
            """
        )

    metricas = {
        "Interpretación": interpretacion,
        "Tasa de morosidad grave del cluster": resultado["tasa_morosidad"],
        "Edad promedio del cluster": resultado["edad_promedio"],
        "Ingreso promedio del cluster": resultado["ingreso_promedio"],
        "Nivel de endeudamiento promedio": resultado["deuda_promedio"],
        "Uso de crédito promedio": resultado["uso_credito_promedio"],
        "Moras de 90 días promedio": resultado["moras_90_promedio"],
        "Clientes en el cluster": resultado["registros"]
    }

    st.dataframe(
        pd.DataFrame(metricas.items(), columns=["Indicador", "Valor"]),
        use_container_width=True
    )

    st.markdown("### Datos ingresados")
    st.dataframe(input_data_presentacion, use_container_width=True)

    st.markdown("### Recomendación")
    if cluster == -1:
        st.write("Recomendación: enviar a revisión manual antes de tomar una decisión financiera.")
    elif "riesgo bajo" in interpretacion.lower() or "viable" in interpretacion.lower():
        st.write("Recomendación: cliente potencialmente viable para evaluación de producto financiero.")
    elif "moderado" in interpretacion.lower():
        st.write("Recomendación: cliente con riesgo moderado. Se recomienda evaluar condiciones, cupo o garantías.")
    else:
        st.write("Recomendación: cliente de alto riesgo. Se recomienda análisis adicional antes de aprobar crédito.")
