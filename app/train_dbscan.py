import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib

# Cargar datos
df = pd.read_csv("data/Mall_Customers.csv")

# Seleccionar columnas
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear modelo DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Entrenar modelo
clusters = dbscan.fit_predict(X_scaled)

# Guardar clusters
df['Cluster'] = clusters

# Guardar modelo
joblib.dump(dbscan, 'models/dbscan_model.joblib')

# Mostrar gráfico
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set1'
)

plt.title("DBSCAN Clustering")
plt.show()

print("MODELO GUARDADO CORRECTAMENTE")