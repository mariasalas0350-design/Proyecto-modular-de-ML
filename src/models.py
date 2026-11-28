"""Modelos de regresión y pipeline que los une con el preprocesamiento."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.config import RANDOM_STATE
from src.preprocessor import build_preprocessor


def get_models():
    """Devuelve un diccionario {nombre: estimador} con los tres modelos a comparar."""
    return {
        "Regresión lineal": LinearRegression(),
        "Árbol de decisión": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
        "Bosque aleatorio": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }


def construir_pipeline(estimador):
    """Devuelve un Pipeline sin ajustar: preprocesamiento + modelo.

    Se crea un preprocesador nuevo en cada llamada para que ningún modelo
    comparta estado ajustado con otro.
    """
    return Pipeline([
        ("preprocesamiento", build_preprocessor()),
        ("modelo", estimador),
    ])