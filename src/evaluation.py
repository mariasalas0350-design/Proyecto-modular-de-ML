"""Métricas, comparación de modelos en validación y evaluación final en prueba."""

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def calcular_metricas(y_real, y_pred):
    """Devuelve un diccionario con MAE y RMSE."""
    return {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": root_mean_squared_error(y_real, y_pred),
    }


def comparar_modelos(modelos, X_train, y_train, X_valid, y_valid):
    """Entrena cada modelo con entrenamiento y lo evalúa en validación.

    modelos: diccionario {nombre: pipeline sin ajustar}.
    Devuelve: un DataFrame con MAE de entrenamiento, MAE de validación
    y RMSE de validación de cada modelo, más una referencia con la mediana.
    """
    # La referencia (mediana de y_train) sirve para saber si los modelos aportan algo
    candidatos = dict(modelos)
    candidatos["Referencia (mediana)"] = DummyRegressor(strategy="median")

    filas = {}
    for nombre, modelo in candidatos.items():
        # El ajuste usa solo entrenamiento (el preprocesamiento va dentro del pipeline)
        modelo.fit(X_train, y_train)
        metricas_train = calcular_metricas(y_train, modelo.predict(X_train))
        metricas_valid = calcular_metricas(y_valid, modelo.predict(X_valid))
        filas[nombre] = {
            "MAE entrenamiento": metricas_train["MAE"],
            "MAE validación": metricas_valid["MAE"],
            "RMSE validación": metricas_valid["RMSE"],
        }

    return pd.DataFrame.from_dict(filas, orient="index")


def evaluar_en_prueba(pipeline, X_train, y_train, X_test, y_test):
    """Entrena con entrenamiento y evalúa una sola vez en prueba.

    Devuelve: un diccionario con MAE y RMSE.
    """
    pipeline.fit(X_train, y_train)
    return calcular_metricas(y_test, pipeline.predict(X_test))