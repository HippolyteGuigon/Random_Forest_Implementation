import numpy as np
import sys
from sklearn.utils import check_random_state



class RandomForest:

    def __init__(self, random_state) -> None:
        self.rng = check_random_state(random_state)
        self.X = self.rng.normal(size=(50, 50))
        random_categorical_values = ["a", "b", "c"]
        self.categorical_values = self.rng.choice(random_categorical_values, size=(50, 10))
        self.X = np.hstack((self.X, self.categorical_values))
        print(self.X)

if __name__=="__main__":
    a=RandomForest(42)       