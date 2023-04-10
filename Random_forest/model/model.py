import numpy as np
import sys
from sklearn.utils import check_random_state
from Random_forest.decision_tree.decision_tree import Decision_Tree
from Random_forest.configs.confs import load_conf

main_params=load_conf("configs/main.yml", include=True)
max_depth=main_params["model_hyperparameters"]["max_depth"]
min_sample_split=main_params["model_hyperparameters"]["min_sample_split"]

class RandomForest:
    def __init__(self, random_state=42, max_depth: int=max_depth, 
                 min_sample_split: int = min_sample_split, **kwargs) -> None:
        
        self.rng = check_random_state(random_state)
        self.max_depth=max_depth
        self.min_samples_split=min_sample_split

        for param, value in kwargs.items():
            if param not in main_params.keys():
                raise AttributeError(f"The Random Forest model has\
                                      no attribute {param}")
            elif hasattr(self, param):
                pass
            else:
                setattr(self, param, value)

        for param, value in main_params["model_hyperparameters"].items():
            if not hasattr(self, param):
                setattr(self, param, value)

    def data_bootstrap(self, X: np.array, y: np.array)->np.array:
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

        full_data=np.hstack((X,  y))
        n=full_data.shape[0]
        choosed_sample_index=np.random.choice(np.arange(0, n), size=(n, 1), replace=True)
        full_data_bootstraped=full_data[choosed_sample_index.flatten(),:]
        X_bootstraped,  y_bootstraped = full_data_bootstraped[:, :-1], full_data_bootstraped[:, -1]

        return X_bootstraped, y_bootstraped.reshape(-1, 1)
    
    def fit(self, X: np.array, y: np.array)->None:
        self.X=X
        self.y=y

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f"X and y must have the same number of rows, X\
                              contains {self.X.shape[0]} and y {self.y.shape[0]}")
        if self.y.shape[1]>1:
            raise ValueError(f"The target dataset must only contain\
                              one column, it contains {self.y.shape[1]}")
        if self.X.shape[0]<=self.min_samples_split:
            raise AssertionError(f"The number of rows\
            of X, {self.X.shape[0]} must be superior to min_sample_split\
                 hyperparameter {self.min_samples_split}")
        
        bootstraped_set=[self.data_bootstrap(self.X, self.y) for _ in range(0,self.n_estimators)]
        model_set=[Decision_Tree(x[0], x[1]) for x in bootstraped_set]
        for x in model_set:
            x.grow_node(x.node)
        
        return model_set