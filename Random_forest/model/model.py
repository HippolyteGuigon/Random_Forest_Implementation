import numpy as np
import logging
from sklearn.utils import check_random_state
from Random_forest.decision_tree.decision_tree import Decision_Tree
from Random_forest.decision_tree.array_functions import is_float
from Random_forest.configs.confs import load_conf
from Random_forest.logs.logs import main
from joblib import Parallel, delayed
from multiprocessing import cpu_count

main_params = load_conf("configs/main.yml", include=True)
max_depth = main_params["model_hyperparameters"]["max_depth"]
min_sample_split = main_params["model_hyperparameters"]["min_sample_split"]

main()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")


class RandomForest:
    """
    The goal of this class is the
    implementation of the final model

    Arguments:
        -random_state: int: The random_state
        used for data generation
        -max_depth: int: The maximum depth of
        each decision Tree composing the forest
        -min_sample_split: The minimum number of
        samples required for having a split in the
        Node

    Returns:
        -None
    """

    def __init__(
        self,
        objective="regression",
        random_state=42,
        max_depth: int = max_depth,
        n_estimators=100,
        min_sample_split: int = min_sample_split,
        max_features: str = "sqrt",
        **kwargs,
    ) -> None:
        self.rng = check_random_state(random_state)
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_samples_split = min_sample_split
        self.max_features = max_features
        self.objective = objective

        if self.objective not in ["classification", "regression"]:
            raise ValueError(
                f"The objective of the Random Forest is classification\
                              or regression got the argument {self.objective}"
            )

        for param, value in kwargs.items():
            if param not in main_params.keys():
                raise AttributeError(
                    f"The Random Forest model has\
                                      no attribute {param}"
                )
            else:
                setattr(self, param, value)

        for param, value in main_params["model_hyperparameters"].items():
            if not hasattr(self, param):
                setattr(self, param, value)

        if self.max_features not in ["sqrt", "log2", None]:
            raise ValueError(
                f"The max_features hyperparameter must be sqrt,\
                              log2 or None, got {self.max_features}"
            )

        logging.info("Model initialized")

    def data_bootstrap(self, X: np.array, y: np.array) -> np.array:
        """
        The goal of this function is to build a
        bootstraped dataset from the original one

        Arguments:
            -X: np.array: The original dataset
            -y: np.array: The target column
        Returns:
            -bootstraped_data: np.array: The
            bootstraped dataset
        """

        full_data = np.hstack((X, y))
        n = full_data.shape[0]
        choosed_sample_index = sorted(np.random.choice(np.arange(0, n), size=n))
        full_data_bootstraped = full_data[choosed_sample_index, :]
        X_bootstraped, y_bootstraped = (
            full_data_bootstraped[:, :-1],
            full_data_bootstraped[:, -1],
        )

        return X_bootstraped, y_bootstraped.reshape(-1, 1)

    def fit(self, X: np.array, y: np.array) -> None:
        """
        The goal  of this function is to fit the
        model, build a forest of Decision Trees
        that will be used afterwhise for the prediction
        of new values

        Arguments:
            -X: np.array: The array of values on which
            the model will be fitted
            -y: np.array: The target column
        Returns:
            -None
        """

        self.X = X
        self.y = y

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows, X\
                              contains {self.X.shape[0]} and y {self.y.shape[0]}"
            )
        if self.y.shape[1] > 1:
            raise ValueError(
                f"The target dataset must only contain\
                              one column, it contains {self.y.shape[1]}"
            )
        if self.X.shape[0] <= self.min_samples_split:
            raise AssertionError(
                f"The number of rows\
            of X, {self.X.shape[0]} must be superior to min_sample_split\
                 hyperparameter {self.min_samples_split}"
            )

        if self.objective == "regression" and (not all([is_float(x) for x in self.y])):
            raise AssertionError(
                "The objective here is regression but categorical values were\
                                 found in the target column"
            )
        elif self.objective == "classification" and (
            any([is_float(x) for x in self.y])
        ):
            raise AssertionError(
                "The objective here is classification but numerical\
                    values were found in the target column, all values\
                        must be categorical for classification"
            )

        logging.warning("Model fitting...")

        bootstraped_set = Parallel(n_jobs=int(cpu_count()))(
            delayed(self.data_bootstrap)(self.X, self.y)
            for _ in range(0, self.n_estimators)
        )

        logging.info("Bootstraped set initialized")

        self.out_of_bag_values = np.array(
            [
                np.unique(dataset[0].astype("<U32"), axis=0)
                for dataset in bootstraped_set
            ]
        )

        logging.info("Out of bag values determined")

        for i in range(self.out_of_bag_values.shape[0]):
            self.out_of_bag_values[i] = np.array(
                [
                    x
                    for x in self.X.tolist()
                    if x not in self.out_of_bag_values[i].tolist()
                ]
            )

        model_set = Parallel(n_jobs=int(cpu_count()))(
            delayed(Decision_Tree)(x[0], x[1]) for x in bootstraped_set
        )

        model_set = Parallel(n_jobs=int(cpu_count()))(
            delayed(x.grow_node)(x.node) for x in model_set
        )

        self.model_set = model_set

        logger.warning("Model has finished fitting !")

    def to_predict_data_allocation(self, data: np.array, node) -> float:
        """
        The goal of this function is to
        allocate the data to be predicted
        in a given decision tree in order
        to make a prediction after

        Arguments:
        -data: The data to be allocated
        -node: The node in which the data
        will be splitted

        Returns:
            -decision: The outcome of the tree
        """

        self.current_node = node
        if node.left or node.right:
            if is_float(data[node.split_column]):
                if node.check_condition(data[node.split_column].astype(float)):
                    self.to_predict_data_allocation(data, self.current_node.left)
                else:
                    self.to_predict_data_allocation(data, self.current_node.right)
            else:
                if node.check_condition(data[node.split_column]):
                    self.to_predict_data_allocation(data, self.current_node.left)
                else:
                    self.to_predict_data_allocation(data, self.current_node.right)

            self.current_node.y = self.current_node.y.flatten()

            if len(self.current_node.y) == 0:
                print(self.current_node.y)

            if is_float(self.current_node.y[0]):
                return np.mean(self.current_node.y.astype(float))
            else:
                values, counts = np.unique(self.current_node.y, return_counts=True)
                return values[counts.argmax()]

    def individual_predict(self, X_to_predict: np.array) -> float:
        """
        The goal of this function is to
        return the predictions made for
        a given array

        Arguments:
            -X_to_predict: np.array: The
            array of values to be predicted
        Returns:
            -predictions: np.array: The predictions
            made for a given array
        """

        predictions = []

        for decision_tree in self.model_set:
            predictions.append(
                self.to_predict_data_allocation(X_to_predict, decision_tree.node)
            )

        if isinstance(predictions[0], (float, int)):
            prediction = np.mean(predictions)
        else:
            values, counts = np.unique(predictions, return_counts=True)
            prediction = values[counts.argmax()]
        return prediction

    def predict(self, full_X_to_predict: np.array) -> np.array:
        """
        The goal of this function is to
        predict a all array

        Arguments:
            -X: np.array: The array to be
            predicted

        Returns:
            -predicted: np.array: The prediction
            made by the model
        """

        if full_X_to_predict.dtype == "O":
            full_X_to_predict = full_X_to_predict.astype("<U32")

        if not hasattr(self, "model_set"):
            raise AssertionError(
                "The model needs to be fitted\
                                 first before predicting"
            )

        if full_X_to_predict.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"The array to predict must be the\
                             same size as the one used to fit the model, but\
                             got respectively {full_X_to_predict.shape[1]} and\
                              {self.X.shape[1]} columns"
            )

        predicted = []

        for x in full_X_to_predict:
            prediction = self.individual_predict(x)
            predicted.append(prediction)

        predicted = np.array(predicted)

        return predicted

    def score(self, y_pred: np.array, y_test: np.array) -> float:
        """
        The goal of this function is, once
        the model has been fitted, to compute
        its score regarding its objective (regression
        or classification)

        Arguments:
            -y_pred: np.array: The predictions made
            by the model
            -y_test: np.array: The real values
        """

        if self.objective == "classification":
            return np.mean(y_pred == y_test)
        else:
            rss = ((y_pred - y_test) ** 2).sum()
            tss = ((y_test - y_test.mean()) ** 2).sum()
            r_squared = 1 - rss / tss
            return r_squared
