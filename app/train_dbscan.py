import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import joblib

# Cargar dataset
df = pd.read_csv("data/CC GENERAL.csv")

# Ver primeras filas
print(df.head())

# Eliminar ID
df = df.drop("CUST_ID", axis=1)

# Eliminar valores nulos
df = df.dropna()

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Crear modelo DBSCAN
model = DBSCAN(eps=1.5, min_samples=5)

# Entrenar modelo
clusters = model.fit_predict(X_scaled)

# Agregar clusters
df["Cluster"] = clusters

# Mostrar clusters encontrados
print(df["Cluster"].value_counts())

# Guardar modelo
joblib.dump(model, "models/dbscan_model.joblib")

print("MODELO GUARDADO")