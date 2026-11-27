"""Creación de variables y preprocesamiento de los datos."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL, NUMERIC


def crear_variables(tabla):
    """Agrega rooms_per_household y bedrooms_per_household.

    Son divisiones fila por fila, así que no aprenden nada de otros datos
    y se pueden aplicar igual a entrenamiento, validación y prueba.
    Se usa .replace(0, 1) en el denominador para evitar dividir entre cero.

    Devuelve: una copia del DataFrame con las columnas nuevas.
    """
    resultado = tabla.copy()
    hogares = resultado["households"].replace(0, 1)
    resultado["bedrooms_per_household"] = resultado["total_bedrooms"] / hogares
    resultado["rooms_per_household"] = resultado["total_rooms"] / hogares
    return resultado


def build_preprocessor():
    """Devuelve un ColumnTransformer SIN ajustar.

    Estandariza las variables numéricas y codifica ocean_proximity.
    Se ajusta dentro del Pipeline, únicamente con los datos de entrenamiento.
    """
    return ColumnTransformer([
        ("numericas", StandardScaler(), NUMERIC),
        ("categoricas",
         OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL),
    ])