import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# CARGAR DATASET
df = pd.read_csv("data/Raw/CC GENERAL.csv")

# MOSTRAR PRIMERAS FILAS
print(df.head())

# ELIMINAR VALORES NULOS
df = df.dropna()

# ELIMINAR COLUMNA NO NUMÉRICA
df = df.drop("CUST_ID", axis=1)

# ESCALAR DATOS
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# ENTRENAR MODELO DBSCAN
model = DBSCAN(eps=1.5, min_samples=5)

# CREAR CLUSTERS
clusters = model.fit_predict(scaled_data)

# AGREGAR CLUSTERS AL DATASET
df["Cluster"] = clusters

# MOSTRAR RESULTADOS
print(df["Cluster"].value_counts())

# GUARDAR MODELO
joblib.dump(model, "models/dbscan_model.joblib")

print("MODELO GUARDADO CORRECTAMENTE")

# GRAFICAR CLUSTERS
# GRAFICAR CLUSTERS

plt.figure(figsize=(12,7))

plt.scatter(
    df["BALANCE"],
    df["PURCHASES"],
    c=clusters,
    cmap='viridis'
)

plt.xlabel("Balance de la Tarjeta")
plt.ylabel("Compras Realizadas")
plt.title("Segmentación de Clientes con DBSCAN")

plt.grid(True)

plt.show()