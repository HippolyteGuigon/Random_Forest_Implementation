import numpy as np
import sys
from sklearn.utils import check_random_state
from Random_forest.decision_tree.decision_tree import Decision_Tree

class RandomForest(Decision_Tree):

    def __init__(self, random_state) -> None:
        self.rng = check_random_state(random_state)
        self.X = self.rng.normal(size=(50, 50))
        self.max_depth=Decision_Tree.max_depth
        self.min_samples_split=Decision_Tree.min_samples_split
        print(self.max_depth)

if __name__=="__main__":
    test=RandomForest(42)

