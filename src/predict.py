import joblib

def cargar_modelo():

    modelo = joblib.load("models/dbscan_model.joblib")

    return modelo