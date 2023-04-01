import numpy as np
import sys
from sklearn.utils import check_random_state



class RandomForest:

    def __init__(self, random_state, max_depth=None, min_samples_split=2) -> None:
        self.rng = check_random_state(random_state)
        self.X = self.rng.normal(size=(50, 50))
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split

