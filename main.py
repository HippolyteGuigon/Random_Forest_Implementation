from Random_forest.model.model import RandomForest
from Random_forest.decision_tree.array_functions import get_random_set, is_float
from Random_forest.configs.confs import load_conf

main_params = load_conf("configs/main.yml", include=True)
row_size_test_dataset = main_params["pytest_configs"]["row_size_test_dataset"]

X, y = get_random_set(
            row_size_test_dataset=row_size_test_dataset, objective="classification"
        )

if __name__=="__main__":
    model=RandomForest(objective="classification")
    model.fit(X, y)