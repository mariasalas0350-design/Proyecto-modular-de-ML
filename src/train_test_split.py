"""División de los datos en entrenamiento, validación y prueba."""
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE, VALID_SIZE


def split_data(X, y):
    """Divide en 60% entrenamiento, 20% validación y 20% prueba.

    Primero se reserva el 20% para prueba; del 80% restante se toma el 25%
    para validación (25% de 80% = 20% del total).

    Devuelve: X_train, X_valid, X_test, y_train, y_valid, y_test
    """
    X_desarrollo, X_test, y_desarrollo, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_desarrollo, y_desarrollo, test_size=VALID_SIZE, random_state=RANDOM_STATE
    )

    # Comprobaciones: los conjuntos no deben compartir filas
    assert set(X_train.index).isdisjoint(X_valid.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_valid.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_valid) + len(X_test) == len(X)

    return X_train, X_valid, X_test, y_train, y_valid, y_test