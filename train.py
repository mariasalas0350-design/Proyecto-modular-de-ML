from src.evaluation import comparar_modelos, evaluar_en_prueba
from src.io import load_data, separar_X_y
from src.models import construir_pipeline, get_models
from src.preprocessor import crear_variables
from src.train_test_split import split_data


def main():
    # 1. Cargar datos y separar predictores (X) y objetivo (y)
    df = load_data()
    X, y = separar_X_y(df)

    # 2. Dividir en entrenamiento, validación y prueba (misma partición para todos)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)

    # 3. Crear las variables nuevas en los tres conjuntos
    X_train = crear_variables(X_train)
    X_valid = crear_variables(X_valid)
    X_test = crear_variables(X_test)

    # 4. Comparar los tres modelos en validación (cada uno dentro de su pipeline)
    pipelines = {
        nombre: construir_pipeline(estimador)
        for nombre, estimador in get_models().items()
    }
    tabla = comparar_modelos(pipelines, X_train, y_train, X_valid, y_valid)
    print("Comparación en validación:")
    print(tabla.round(2))

    # 5. Elegir el de menor MAE de validación (sin considerar la referencia)
    mejor = tabla.loc[list(pipelines), "MAE validación"].idxmin()
    print(f"\nModelo elegido: {mejor}")

    # 6. Evaluar solo el elegido, una vez, en prueba (con un pipeline nuevo)
    pipeline_final = construir_pipeline(get_models()[mejor])
    metricas = evaluar_en_prueba(pipeline_final, X_train, y_train, X_test, y_test)
    print(f"Resultado en prueba -> MAE: {metricas['MAE']:.2f} | RMSE: {metricas['RMSE']:.2f}")


if __name__ == "__main__":
    main()