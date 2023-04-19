import numpy as np


def treshold_numeric(data: float, reference_value: float) -> bool:
    """
    The goal of this function is to compare
    a given data to a reference value to check
    whether it should go left or right node

    Arguments:
        -data: float: The data to be attributed
        left or right Node
        -reference_value: float: The treshold it
        is compared with
    Returns:
        -bool: True if data is inferior to reference
        value, False otherwise
    """
    if data < reference_value:
        return True
    else:
        return False


def split_categorical(data: str, reference_value: str) -> bool:
    """
    The goal of this function is to compare
    a given data to a reference value to check
    whether it should go left or right node

    Arguments:
        -data: str: The data to be attributed
        left or right Node
        -reference_value: str: The categorical value
        it is compared with
    Returns:
        -bool: True if data is equal to reference value,
        False otherwise
    """
    return data == reference_value


def is_float(x) -> bool:
    """
    The goal of this function is
    to determine wheter a column of
    a given array is float type or
    not

    Arguments:
        -x: The element that needs to
        be determined (float or not)
    Returns:
        -bool: True if x is float, False
        otherwise
    """
    try:
        x = float(x)
        return True
    except ValueError:
        return False


def get_random_set(
    row_size_test_dataset: np.array, objective: str = "classification"
) -> np.array:
    """
    The goal of this function is to
    get random dataset to perform multiple
    tests

    Arguments:
        -row_size_test_dataset: np.array:
        The dataset from which to build a
        test set from
        -objective: str: To know what type
        of target column it must be
    Returns:
        -X: np.array: The feature dataset
        computed
        -y: np.array: The targett column
        computed
    """

    assert objective in [
        "classification",
        "regression",
    ], f"The objective\
              must be classification or regression but got {objective}"

    categorical_value_1 = ["retraités", "actifs", "étudiant"]
    categorical_value_2 = ["a", "b", "c", "d", "e"]

    X_numeric_normal = np.random.normal(scale=30, size=(row_size_test_dataset, 1))
    X_numeric_geometric = np.random.geometric(p=0.1, size=(row_size_test_dataset, 1))
    X_numeric_poisson = np.random.poisson(size=(row_size_test_dataset, 1))
    X_numeric = np.hstack((X_numeric_normal, X_numeric_geometric, X_numeric_poisson))

    X_categorical_1 = np.random.choice(
        categorical_value_1, size=(row_size_test_dataset, 1)
    )
    X_categorical_2 = np.random.choice(
        categorical_value_2, size=(row_size_test_dataset, 1)
    )
    X_categorical = np.hstack((X_categorical_1, X_categorical_2))

    X = np.hstack((X_categorical, X_numeric))

    if objective == "classification":
        target_value = ["target_1", "target_2", "target_3"]
        y = np.random.choice(target_value, size=(row_size_test_dataset, 1))
    else:
        y = np.random.normal(scale=30, size=(row_size_test_dataset, 1))

    return X, y
