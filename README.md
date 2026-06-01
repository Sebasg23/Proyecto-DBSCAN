# Segmentación de Clientes para Análisis de Riesgo Crediticio con DBSCAN

## Descripción

Este proyecto desarrolla una aplicación de Machine Learning no supervisado para segmentar clientes según su perfil financiero y comportamiento crediticio.

El objetivo principal es apoyar a una entidad financiera en la identificación de perfiles de clientes similares, detectando clientes viables, clientes con posible riesgo crediticio y clientes atípicos que requieren revisión manual.

El modelo no se limita a mostrar un número de cluster, sino que busca traducir el resultado en una interpretación útil para el negocio, como:

- Perfil viable: riesgo bajo.
- Perfil de alto riesgo.
- Cliente atípico: requiere revisión manual.

## Demostración

[Enlace a la aplicación desplegada](https://url-de-la-app.com)

> Nota: El enlace será actualizado cuando la aplicación sea desplegada en Streamlit Cloud.

## Algoritmo utilizado

- **Nombre del algoritmo:** DBSCAN  
  DBSCAN significa *Density-Based Spatial Clustering of Applications with Noise*. Es un algoritmo de clustering basado en densidad que permite encontrar grupos de datos similares e identificar registros atípicos o ruido.

- **Por qué es apropiado para este problema:**  
  DBSCAN es adecuado porque el proyecto busca segmentar clientes financieros según variables como ingreso mensual, nivel de endeudamiento, uso de crédito disponible, historial de mora y cantidad de créditos abiertos. Además, permite detectar clientes atípicos mediante la etiqueta `-1`, lo cual es útil en un contexto de riesgo crediticio porque algunos clientes pueden presentar comportamientos financieros inusuales que requieren revisión manual.

- **Parámetros seleccionados:**
  - `eps`: 2.0
  - `min_samples`: 20

- **Métricas de desempeño obtenidas:**
  - Cantidad de clusters encontrados: 4
  - Cantidad de clientes atípicos: 630
  - Porcentaje de clientes atípicos: 6.3 %
  - Silhouette Score: 0.2012
  - Davies-Bouldin Score: 1.2135

## Dataset

- **Fuente de los datos:**  
  Kaggle - Give Me Some Credit  
  https://www.kaggle.com/c/GiveMeSomeCredit

- **Archivo utilizado:**  
  `data/raw/cs-training.csv`

- **Tamaño del dataset original:**  
  150.000 registros y 12 columnas.

- **Tamaño utilizado para entrenamiento:**  
  10.000 registros y 10 variables financieras.

- **Variable de referencia:**  
  `SeriousDlqin2yrs`  
  Esta variable indica si el cliente presentó morosidad grave de 90 días o más en los siguientes dos años. No se utiliza para entrenar DBSCAN, ya que el algoritmo es no supervisado. Se usa únicamente para interpretar los clusters desde una perspectiva de negocio.

- **Features utilizadas:**

| Dato original | Nombre |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Uso de crédito disponible |
| `age` | Edad del cliente |
| `NumberOfTime30-59DaysPastDueNotWorse` | Moras de 30 a 59 días |
| `DebtRatio` | Nivel de endeudamiento |
| `MonthlyIncome` | Ingreso mensual |
| `NumberOfOpenCreditLinesAndLoans` | Créditos y líneas abiertas |
| `NumberOfTimes90DaysLate` | Moras de 90 días o más |
| `NumberRealEstateLoansOrLines` | Créditos hipotecarios o inmobiliarios |
| `NumberOfTime60-89DaysPastDueNotWorse` | Moras de 60 a 89 días |
| `NumberOfDependents` | Dependientes económicos |

## Instalación local

### Requisitos

- Python 3.11+
- pip
- Git

### Pasos

```bash
git clone https://github.com/Sebasg23/Proyecto-DBSCAN.git
cd Proyecto-DBSCAN
pip install -r requirements.txt