import unittest
import numpy as np
import warnings
import logging
from Random_forest.criterion.criterion import (
    gini_impurity_categorical,
    full_gini_compute,
    variance_reduction_categorical,
    variance_reduction_numerical,
)
from Random_forest.decision_tree.decision_tree import (
    Node,
    Decision_Tree,
    get_bottom_values,
)
from Random_forest.decision_tree.array_functions import get_random_set, is_float
from Random_forest.configs.confs import load_conf
from Random_forest.model.model import RandomForest
from Random_forest.logs.logs import main

warnings.filterwarnings("ignore")

main_params = load_conf("configs/main.yml", include=True)
row_size_test_dataset = main_params["pytest_configs"]["row_size_test_dataset"]

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")


class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    logging.info("Performing unitest")

    def test_categorical_gini_criterion(self) -> None:
        """
        The goal of this function is to check wheter the
        gini function works for categorical values

        Arguments:
            None

        Returns:
            None
        """
        random_categorical_values = ["a", "b", "c"]
        target = ["survivor", "non_survivor"]
        X = np.random.choice(random_categorical_values, size=(row_size_test_dataset, 5))
        y_categorical = np.random.choice(target, size=(row_size_test_dataset, 1))

        selected_column, gini_score = full_gini_compute(X, y_categorical)

        self.assertEqual(gini_impurity_categorical(y_categorical, y_categorical)[0], 0)
        self.assertTrue(selected_column in range(X.shape[1]))
        self.assertTrue(((gini_score[0] <= 1) and (gini_score[0] >= 0)))

    def test_variance_reduction(self) -> None:
        """ "
        The goal of this function is to check wheter
        the variance reduction technique works with
        categorical values used to split numerical
        values

        Arguments:
            None

        Returns:
            None
        """
        random_categorical_values = ["a", "b", "c"]
        random_numerical_values = np.random.normal(
            size=(row_size_test_dataset, 1), scale=50
        )
        random_numerical_values = np.array(sorted(random_numerical_values))
        X = np.random.choice(random_categorical_values, size=(row_size_test_dataset, 1))
        y = np.random.uniform(size=(row_size_test_dataset, 1))
        best_candidate_categorical = variance_reduction_categorical(X, y)
        best_candidate_numerical = variance_reduction_numerical(
            random_numerical_values, y
        )
        split_values = np.array(
            [
                (
                    np.float(random_numerical_values[i])
                    + np.float(random_numerical_values[i + 1])
                )
                / 2
                for i in range(row_size_test_dataset - 1)
            ]
        )

        self.assertTrue((best_candidate_categorical[1] in np.unique(X)))
        self.assertTrue((best_candidate_numerical[1] in split_values))

    def test_check_split(self) -> None:
        """
        The goal of this function is to check if,
        from a given numpy array composed of multiple
        data types, the Tree class is able to choose the
        best value to perform a split

        Arguments:
            None

        Returns:
            None
        """

        target = ["category_a", "category_b", "category_c"]
        X_numeric = np.random.normal(size=(row_size_test_dataset, 3), scale=30)
        X_categorical = np.random.choice(target, size=(row_size_test_dataset, 3))
        full_data = np.hstack((X_numeric, X_categorical))
        y = np.random.randint(5, size=(row_size_test_dataset, 1))

        Tree = Node(full_data, y)
        Tree.compute_condition()
        split_value = Tree.split_value

        if isinstance(split_value, (np.float_, np.int_)):
            data_test_true = split_value - 1
            data_test_false = split_value + 1
            self.assertTrue(Tree.condition(data_test_true, reference_value=split_value))
            self.assertFalse(
                Tree.condition(data_test_false, reference_value=split_value)
            )
        else:
            data_test_true = split_value
            self.assertTrue(Tree.condition(data_test_true, reference_value=split_value))

    def test_tree_values_allocation(self):
        """
        The goal of this function is to check if
        the values given to a binary tree are well
        and all allocated

        Arguments:
            None

        Returns:
            None
        """

        X, y = get_random_set(
            row_size_test_dataset=row_size_test_dataset, objective="regression"
        )

        Tree = Decision_Tree(X, y)
        Tree.grow_node(Tree.node)
        bottom_values = get_bottom_values(Tree.node)
        bottom_sum = np.sum(bottom_values)

        self.assertEqual(bottom_sum, row_size_test_dataset)

    def test_model_prediction(self):
        """
        The goal of this function is
        to check if the model can deliver
        accuracte predictions, both for
        classification and regression

        Arguments:
            None
        Returns:
            None
        """

        X, y = get_random_set(
            row_size_test_dataset=row_size_test_dataset, objective="classification"
        )
        X_predict, _ = get_random_set(
            row_size_test_dataset=30, objective="classification"
        )

        model_classification = RandomForest(objective="classification")
        model_classification.fit(X, y)

        classification_predictions = model_classification.predict(X_predict)
        classification_predictions = [x in y for x in classification_predictions]

        X, y = get_random_set(
            row_size_test_dataset=row_size_test_dataset, objective="regression"
        )
        X_predict, _ = get_random_set(row_size_test_dataset=30, objective="regression")

        model_regression = RandomForest(objective="regression")
        model_regression.fit(X, y)

        regression_predictions = model_regression.predict(X_predict)
        regression_predictions = [is_float(x) for x in regression_predictions]

        self.assertTrue(np.all(classification_predictions))
        self.assertTrue(np.all(regression_predictions))


if __name__ == "__main__":
    main()
    unittest.main()
