Grupo 2 - Actividad 3 - Asesoría de Avances

URL Repositorio: https://github.com/Sebasg23/Proyecto-DBSCAN

Algoritmo: DBSCAN

Problema:
Segmentación de clientes para análisis de riesgo crediticio mediante DBSCAN.

Descripción del problema:
Una entidad financiera necesita analizar clientes o solicitantes de crédito para identificar perfiles financieros similares y detectar comportamientos atípicos. El objetivo del proyecto no es solamente mostrar a qué cluster pertenece una persona, sino traducir ese resultado en una conclusión útil para el negocio, como cliente viable, cliente de alto riesgo o cliente atípico que requiere revisión manual.

Este enfoque permite apoyar procesos de análisis crediticio, ya que al banco no solo le interesa saber si una persona ha presentado mora, sino también si puede seguir siendo cliente financiero, solicitar nuevos productos, mantener capacidad de pago y representar una oportunidad rentable con un nivel de riesgo controlado.

Dataset:
Give Me Some Credit - Kaggle

Fuente:
https://www.kaggle.com/c/GiveMeSomeCredit

Descripción del dataset:
El dataset contiene información financiera de clientes, incluyendo variables relacionadas con edad, ingreso mensual, nivel de endeudamiento, uso de crédito disponible, líneas de crédito abiertas, historial de mora y dependientes económicos.

El objetivo original del dataset es mejorar el scoring crediticio mediante la predicción de la probabilidad de que una persona experimente dificultades financieras en los próximos dos años. En este proyecto, el dataset se utiliza con un enfoque no supervisado para segmentar clientes y detectar perfiles atípicos mediante DBSCAN.

Tamaño del dataset:
- Dataset original: 150.000 registros y 12 columnas.
- Dataset utilizado para entrenamiento: 10.000 registros y 10 variables financieras.

Variable de referencia:
- SeriousDlqin2yrs

Esta variable indica si el cliente presentó morosidad grave de 90 días o más en los siguientes dos años. No se utiliza para entrenar DBSCAN, porque el algoritmo es no supervisado. Se utiliza únicamente para interpretar los clusters desde una perspectiva de negocio.

Features utilizadas:
- RevolvingUtilizationOfUnsecuredLines: Uso de crédito disponible.
- age: Edad del cliente.
- NumberOfTime30-59DaysPastDueNotWorse: Moras de 30 a 59 días.
- DebtRatio: Nivel de endeudamiento.
- MonthlyIncome: Ingreso mensual.
- NumberOfOpenCreditLinesAndLoans: Créditos y líneas abiertas.
- NumberOfTimes90DaysLate: Moras de 90 días o más.
- NumberRealEstateLoansOrLines: Créditos hipotecarios o inmobiliarios.
- NumberOfTime60-89DaysPastDueNotWorse: Moras de 60 a 89 días.
- NumberOfDependents: Dependientes económicos.

Estado del Modelo:
- Modelo entrenado: SÍ
- Algoritmo utilizado: DBSCAN
- Parámetros seleccionados:
  - eps: 2.0
  - min_samples: 20
- Dataset utilizado: data/raw/cs-training.csv
- Dataset procesado: data/processed/give_me_some_credit_processed.csv
- Archivo del modelo serializado: models/dbscan_model.pkl
- Archivo del scaler serializado: models/scaler.pkl
- Archivo de información del modelo: models/model_info.pkl
- Cantidad de clusters encontrados: 4
- Cantidad de clientes atípicos: 630
- Porcentaje de clientes atípicos: 6.3 %
- Silhouette Score: 0.2012
- Davies-Bouldin Score: 1.2135
- Técnica de preprocesamiento: limpieza de datos, tratamiento de valores nulos, eliminación de la columna identificadora, separación de la variable de referencia, selección de variables financieras y estandarización con StandardScaler.
- Objetivo del modelo: identificar perfiles financieros similares y detectar clientes atípicos que puedan representar un posible riesgo crediticio.

Interpretación de los clusters:
- Cluster -1: Cliente atípico, requiere revisión manual.
- Cluster 0: Perfil viable, riesgo bajo.
- Cluster 1: Perfil de alto riesgo.
- Cluster 2: Perfil de alto riesgo.
- Cluster 3: Perfil de alto riesgo.

Estado de la Aplicación:

- Framework utilizado: Streamlit.
- Archivo principal de la aplicación:
  app/app.py

- Estado actual:
  La aplicación web se encuentra funcionando localmente en versión inicial. Actualmente permite cargar el modelo DBSCAN entrenado y serializado, junto con el scaler utilizado durante el preprocesamiento de los datos.

  La aplicación permite ingresar información financiera de un cliente y obtener una interpretación útil del resultado. En lugar de mostrar únicamente “Cluster 0” o “Cluster 1”, la salida se traduce en perfiles de negocio como cliente viable, cliente de alto riesgo o cliente atípico que requiere revisión manual.

- Funcionalidades implementadas:
  - Carga del modelo DBSCAN serializado.
  - Carga del scaler utilizado en el entrenamiento.
  - Carga de información del modelo desde model_info.pkl.
  - Formulario para ingresar variables financieras del cliente.
  - Estandarización de los datos ingresados.
  - Asignación del cliente a un cluster.
  - Interpretación del cluster en lenguaje de negocio.
  - Recomendación general para análisis crediticio.

Despliegue Planificado:
- Servicio: Streamlit Cloud
- Fecha estimada: 03/05/2026

Investigación y enfoque aplicado:
El proyecto se orienta a un caso real de negocio dentro del sector financiero. El uso de técnicas de Machine Learning en análisis crediticio permite apoyar la toma de decisiones sobre evaluación de clientes, asignación de productos financieros, detección de perfiles de riesgo y revisión de casos atípicos.

DBSCAN es apropiado en este contexto porque permite descubrir grupos de clientes con comportamientos financieros similares y detectar registros que no encajan con los perfiles comunes. Estos casos atípicos pueden ser importantes para revisión manual, especialmente cuando una entidad financiera necesita tomar decisiones responsables sobre crédito, continuidad del cliente y rentabilidad futura.