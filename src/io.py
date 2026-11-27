"""Carga de datos y separación de predictores y objetivo."""
import pandas as pd

from src.config import DATA_PATH, TARGET


def load_data():
    """Lee housing.csv y elimina las filas con valores nulos.

    Devuelve: un DataFrame con registros completos.
    """
    datos_originales = pd.read_csv(DATA_PATH)
    datos = datos_originales.dropna().copy()
    eliminadas = len(datos_originales) - len(datos)
    print(f"Filas cargadas: {len(datos_originales):,} | "
          f"eliminadas por nulos: {eliminadas} ({eliminadas / len(datos_originales):.2%})")
    return datos


def separar_X_y(df):
    """Separa los predictores (X) de la variable objetivo (y)."""
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return X, y