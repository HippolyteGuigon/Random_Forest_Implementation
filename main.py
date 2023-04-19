import argparse
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from Random_forest.model.model import RandomForest
from Random_forest.decision_tree.array_functions import get_random_set
from Random_forest.configs.confs import load_conf

main_params = load_conf("configs/main.yml", include=True)
row_size_test_dataset = main_params["pytest_configs"]["row_size_test_dataset"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "objective",
    help="The objective to test the Random Forest, do you prefer classification\
        or regression ?",
    nargs="?",
    const="classification",
    type=str,
)

args = parser.parse_args()

X, y = get_random_set(
    row_size_test_dataset=row_size_test_dataset, objective="classification"
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

if __name__ == "__main__":
    model = RandomForest(objective="classification")
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    if args.objective == "classification":
        score = np.mean(np.array(prediction) == np.array(y_test))
        print(f"The accuracy of the model is {score}")
    else:
        print(
            f"The mean absolute error of the model\
               is {mean_absolute_error(y_test, prediction)}"
        )
