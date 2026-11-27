"""Configuración central del proyecto: rutas, variables y semilla."""
from pathlib import Path

# Raíz del proyecto (la carpeta que contiene src/, datos/ y train.py).
# Así las rutas funcionan sin importar desde dónde se ejecute el código.
RAIZ = Path(__file__).resolve().parent.parent
DATA_PATH = RAIZ / "datos" / "housing.csv"

# Variable objetivo
TARGET = "median_house_value"

# Predictores originales
NUMERIC_ORIGINAL = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
CATEGORICAL = ["ocean_proximity"]

# Variables creadas en preprocessor.crear_variables
NUMERIC_DERIVED = ["rooms_per_household", "bedrooms_per_household"]

# Columnas numéricas que recibe el preprocesador (originales + derivadas)
NUMERIC = NUMERIC_ORIGINAL + NUMERIC_DERIVED

# Semilla y proporciones de la división
RANDOM_STATE = 42
TEST_SIZE = 0.20    # 20% del total para prueba
VALID_SIZE = 0.25   # 25% del 80% restante = 20% del total para validación