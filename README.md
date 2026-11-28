# Proyecto modular de ML: precios de vivienda

## 1. Descripción del problema y del conjunto de datos

### Problema

El objetivo es predecir el valor mediano de las viviendas de una zona censal de California a partir de las características de esa zona. Es un problema de **regresión**, porque la variable a predecir es numérica y continua.

Cada fila del conjunto de datos describe una **zona censal**, no una vivienda individual. Por eso una predicción es una estimación del valor mediano de las viviendas de esa zona, no debe interpretarse como el precio de una casa específica.

### Conjunto de datos

Se utiliza `housing.csv`, con 20,640 filas y 10 columnas:

| Columna | Descripción |
|---|---|
| `longitude`, `latitude` | Ubicación geográfica de la zona |
| `housing_median_age` | Edad mediana de las viviendas |
| `total_rooms` | Total de habitaciones en la zona |
| `total_bedrooms` | Total de recámaras en la zona (tiene 207 valores faltantes) |
| `population` | Población de la zona |
| `households` | Número de hogares |
| `median_income` | Ingreso mediano de los hogares |
| `ocean_proximity` | Cercanía al océano (variable categórica) |
| `median_house_value` | **Variable objetivo:** valor mediano de las viviendas, en dólares |

### Preparación de los datos

- Se eliminan las filas con valores faltantes (207 filas, 1% del total), por lo que se trabaja con 20,433 registros.
- Se crean dos variables a partir de las originales: `rooms_per_household` y `bedrooms_per_household`.
- Los datos se dividen en 60% entrenamiento, 20% validación y 20% prueba, con la semilla 42.
- El preprocesamiento (estandarización de las variables numéricas y codificación de `ocean_proximity`) se ajusta únicamente con los datos de entrenamiento.

## 2. Integrantes

| Nombre | Usuario de GitHub | Aporte |
|---|---|---|
| Maria Gabriela Salas Romo | mariasalas0350-design | Datos y preprocesamiento |
| Santiago Olivares Legaspi | santiagoolivares4510-hash | Modelos y evaluación |

## 3. Organización de los archivos

```
Proyecto-moular-de-ML/
├── datos/
│   └── housing.csv            # Conjunto de datos
├── notebooks/
│   └── actividad_01_housing.py  # Notebook original de Marimo (referencia, sin respuestas)
├── src/
│   ├── __init__.py
│   ├── config.py              # Rutas, variables, semilla y proporciones
│   ├── io.py                  # Carga de datos y separación de X / y
│   ├── train_test_split.py    # División en entrenamiento, validación y prueba
│   ├── preprocessor.py        # Variables nuevas y preprocesamiento
│   ├── models.py              # Modelos y pipelines
│   └── evaluation.py          # Métricas, comparación y evaluación final
├── train.py                   # Archivo principal que coordina el flujo
├── requirements.txt           # Dependencias
└── README.md
```

El proyecto parte de la actividad `actividad_01_housing.py` del repositorio del curso (carpeta `01_ml_recap`), cuyo notebook de Marimo se reorganizó en módulos.

## 4. Instalación de dependencias y ubicación de los datos

### Requisitos

- Python 3.11 o superior
- Git

### Instalación

1. Clonar el repositorio en la carpeta donde se guardará el proyecto:

```bash
git clone git clone https://github.com/mariasalas0350-design/Proyecto-modularde-ML.git
```

2. Crear y activar un entorno virtual: (Opcional pero recomendado) 

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

3. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Las dependencias son `pandas`, `scikit-learn` y `matplotlib`.

### Ubicación de los datos

El conjunto de datos ya está incluido en el repositorio, en `datos/housing.csv`, y no hace falta descargarlo. La ruta se define en `src/config.py`.

Si el archivo faltara, se puede descargar desde el repositorio del curso y colocarlo en la carpeta `datos/` con el nombre `housing.csv`:

https://raw.githubusercontent.com/IvTole/Intro_IA_ML_CUGDL/refs/heads/main/data/housing/housing.csv

## 5. Comandos para ejecutar el proyecto

Con las dependencias instaladas y desde la raíz del repositorio:

```bash
python train.py
```

El script carga `datos/housing.csv`, elimina los nulos, separa X / y, divide los datos en entrenamiento (60%), validación (20%) y prueba (20%) y crea las variables nuevas. Después compara los tres modelos en validación, elige el de menor MAE de validación y lo evalúa una sola vez en prueba. Imprime la tabla de comparación, el modelo elegido y su MAE y RMSE en prueba.

## 6. Modelos utilizados, resultados e interpretación

### Modelos

| Modelo | Configuración |
|---|---|
| Regresión lineal | Valores por defecto |
| Árbol de decisión | `max_depth=6`, `random_state=42` |
| Bosque aleatorio | `n_estimators=100`, `random_state=42` |

Se incluye además una referencia (`DummyRegressor` con la mediana de entrenamiento) para saber si los modelos aportan algo. Cada modelo va dentro de un `Pipeline` junto con el preprocesador, que se ajusta solo con entrenamiento. Todos usan la misma partición y la misma semilla. Las métricas son MAE y RMSE, en dólares.

### Resultados en validación

| Modelo | MAE entrenamiento | MAE validación | RMSE validación |
|---|---|---|---|
| Regresión lineal | [ ] | [ ] | [ ] |
| Árbol de decisión | [ ] | [ ] | [ ] |
| Bosque aleatorio | [ ] | [ ] | [ ] |
| Referencia (mediana) | [ ] | [ ] | [ ] |

Modelo elegido (menor MAE de validación): **[ ]**

Resultado en prueba (una sola evaluación): MAE = [ ], RMSE = [ ]

### Interpretación

[2 o 3 frases: qué modelo ganó y por cuánto superó a la referencia; si el MAE de entrenamiento es mucho menor que el de validación (señal de sobreajuste); si el resultado en prueba se parece al de validación; y por qué el RMSE sale mayor que el MAE.]


## 7. Limitaciones y problemas conocidos

- La comparación usa una sola partición; los resultados podrían cambiar con otra semilla o con validación cruzada.
- No se ajustaron hiperparámetros: se usaron los valores fijos del enunciado.
- Solo se usaron MAE y RMSE; no se analizan los errores por zona ni por rango de precio.
- Cada fila es una zona censal, así que las predicciones no representan el precio de una casa individual.
- [Añade cualquier problema que encuentres al ejecutar.]

## 8. Tabla de contribuciones

(pendiente: Ambos)