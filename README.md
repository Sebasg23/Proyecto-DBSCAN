# Segmentación de Clientes para Riesgo Crediticio con DBSCAN

## Descripción

Esta aplicación permite analizar perfiles financieros de clientes con el objetivo de apoyar procesos de evaluación de riesgo crediticio en una entidad financiera.

El proyecto busca identificar grupos de clientes con comportamientos similares y detectar clientes atípicos que podrían requerir revisión manual antes de aprobar, rechazar o ampliar una línea de crédito.

Para esto se utiliza un modelo de clustering no supervisado basado en **DBSCAN**, aplicado sobre variables financieras como ingreso mensual, nivel de endeudamiento, uso de crédito disponible, historial de mora y número de créditos activos.

## Demostración

[Enlace a la aplicación desplegada](https://proyecto-dbscan-mppggsw7ci9xtk3rnvxpru.streamlit.app/)


## Algoritmo utilizado

* **Nombre del algoritmo:** DBSCAN — Density-Based Spatial Clustering of Applications with Noise.

* **Por qué es apropiado para este problema:**

DBSCAN es apropiado porque permite agrupar clientes según la densidad de sus características financieras sin necesidad de definir previamente la cantidad de clusters.

Además, DBSCAN identifica registros atípicos mediante la etiqueta `-1`, lo cual es útil en un contexto de riesgo crediticio, ya que algunos clientes pueden presentar comportamientos financieros poco comunes que requieren revisión adicional.

Este algoritmo es adecuado para el caso de negocio porque permite:

* Segmentar clientes con perfiles financieros similares.

* Detectar clientes atípicos.

* Apoyar el análisis de riesgo crediticio.

* Traducir los clusters a perfiles de negocio como cliente viable, riesgo moderado, alto riesgo o revisión manual.

* **Métricas de desempeño obtenidas:**

Como DBSCAN es un algoritmo no supervisado, no se evalúa con métricas como accuracy, precision o recall. En su lugar, se usan métricas de clustering.

| Métrica                              | Resultado |
| ------------------------------------ | --------: |
| Cantidad de clusters encontrados     |         4 |
| Clientes atípicos detectados         |       630 |
| Porcentaje de clientes atípicos      |     6.3 % |
| Silhouette Score                     |    0.2012 |
| Davies-Bouldin Score                 |    1.2135 |
| Parámetro `eps` seleccionado         |       2.0 |
| Parámetro `min_samples` seleccionado |        20 |

El modelo también utiliza la variable `SeriousDlqin2yrs` únicamente como referencia de negocio para interpretar los clusters, pero esta variable no se usa durante el entrenamiento porque el enfoque del proyecto es no supervisado.

## Dataset

* **Fuente de los datos:**

El dataset utilizado es **Give Me Some Credit**, disponible en Kaggle.

Fuente:
https://www.kaggle.com/c/GiveMeSomeCredit

* **Archivo utilizado:**

```text
data/raw/cs-training.csv
```

* **Tamaño del dataset original:**

```text
150000 registros
12 columnas
```

* **Tamaño utilizado para entrenamiento:**

Para mejorar el rendimiento de DBSCAN, se trabajó con una muestra controlada de:

```text
10000 registros
10 features financieras
```

* **Features utilizadas:**

| Columna original                       | Descripción                           |
| -------------------------------------- | ------------------------------------- |
| `RevolvingUtilizationOfUnsecuredLines` | Uso de crédito disponible             |
| `age`                                  | Edad del cliente                      |
| `NumberOfTime30-59DaysPastDueNotWorse` | Moras de 30 a 59 días                 |
| `DebtRatio`                            | Nivel de endeudamiento                |
| `MonthlyIncome`                        | Ingreso mensual                       |
| `NumberOfOpenCreditLinesAndLoans`      | Créditos y líneas abiertas            |
| `NumberOfTimes90DaysLate`              | Moras de 90 días o más                |
| `NumberRealEstateLoansOrLines`         | Créditos hipotecarios o inmobiliarios |
| `NumberOfTime60-89DaysPastDueNotWorse` | Moras de 60 a 89 días                 |
| `NumberOfDependents`                   | Dependientes económicos               |

La columna `SeriousDlqin2yrs` se conserva únicamente como referencia para interpretar los resultados, pero no se incluye en el entrenamiento del modelo.

## Preprocesamiento

Antes de entrenar el modelo se realizaron los siguientes pasos:

1. Eliminación de la columna identificadora `Unnamed: 0`.
2. Separación de la variable `SeriousDlqin2yrs` como referencia de negocio.
3. Selección de variables financieras relevantes.
4. Imputación de valores nulos usando la mediana.
5. Tratamiento de valores extremos mediante clipping entre percentiles 1 y 99.
6. Selección de una muestra de 10000 registros.
7. Escalado de variables usando `StandardScaler`.

El escalado es necesario porque DBSCAN trabaja con distancias. Si las variables tienen escalas muy diferentes, el modelo puede dar más peso a columnas con valores grandes como ingreso mensual o nivel de endeudamiento.

## Resultados del modelo

El modelo encontró 4 clusters principales y un grupo de clientes atípicos representado con la etiqueta `-1`.

Distribución de clientes por cluster:

| Cluster | Cantidad de clientes | Interpretación general                    |
| ------: | -------------------: | ----------------------------------------- |
|      -1 |                  630 | Cliente atípico, requiere revisión manual |
|       0 |                 9016 | Cliente viable o de menor riesgo          |
|       1 |                  174 | Cliente de alto riesgo                    |
|       2 |                  160 | Cliente con riesgo medio                  |
|       3 |                   20 | Cliente de alto riesgo                    |

Los clientes marcados como `-1` no deben interpretarse automáticamente como rechazados. Son clientes cuyo comportamiento financiero no se parece al de los grupos principales, por lo que deben ser revisados con mayor detalle.

## Archivos generados

El notebook serializa los siguientes archivos en la carpeta `models/`:

```text
models/dbscan_model.pkl
models/scaler.pkl
models/model_info.pkl
```

Descripción de cada archivo:

| Archivo            | Descripción                                                                  |
| ------------------ | ---------------------------------------------------------------------------- |
| `dbscan_model.pkl` | Modelo DBSCAN entrenado                                                      |
| `scaler.pkl`       | Escalador utilizado para normalizar los datos                                |
| `model_info.pkl`   | Información del modelo, columnas, métricas, parámetros y perfiles de negocio |

## Instalación local

### Requisitos

* Python 3.11+
* pip
* Git

### Pasos

```bash
git clone https://github.com/usuario/proyecto.git
cd proyecto
pip install -r requirements.txt
```

> Reemplazar `https://github.com/usuario/proyecto.git` por la URL real del repositorio.

## Ejecución del notebook

Para ejecutar el notebook, abrir el archivo en Jupyter Notebook o Visual Studio Code:

```text
notebooks/Grupo2_Notebook_DBSCAN.ipynb
```

El dataset debe estar ubicado en:

```text
data/raw/cs-training.csv
```

## Ejecución de la aplicación

Si el proyecto cuenta con una aplicación en Streamlit, se puede ejecutar con:

```bash
streamlit run app/app.py
```

## Estructura del proyecto

```text
Proyecto-DBSCAN/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── cs-training.csv
│   └── processed/
│       └── give_me_some_credit_processed.csv
│
├── models/
│   ├── dbscan_model.pkl
│   ├── scaler.pkl
│   └── model_info.pkl
│
├── notebooks/
│   └── Grupo2_Notebook_DBSCAN.ipynb
│
├── requirements.txt
└── README.md
```

## Tecnologías utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit

## Conclusiones

El proyecto implementa un modelo DBSCAN para segmentar clientes en un contexto de riesgo crediticio. El modelo permite identificar perfiles financieros similares y detectar clientes atípicos que requieren revisión manual.

El dataset utilizado es real, contiene más de 500 registros y más de 5 variables relevantes, cumpliendo con los requisitos del proyecto. Además, el modelo fue entrenado, evaluado y serializado correctamente para su posterior integración con una aplicación web.

## Referencias

* Kaggle. Give Me Some Credit.
  https://www.kaggle.com/c/GiveMeSomeCredit

* Scikit-learn. DBSCAN.
  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

* Scikit-learn. StandardScaler.
  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

* Scikit-learn. Clustering performance evaluation.
  https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

* Scikit-learn. PCA.
  https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
