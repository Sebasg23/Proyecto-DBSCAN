import pandas as pd

def limpiar_datos(df):

    df = df.drop(columns=["CUST_ID"])

    df.fillna(df.mean(), inplace=True)

    return df