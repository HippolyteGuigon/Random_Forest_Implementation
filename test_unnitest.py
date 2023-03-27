import unittest
import numpy as np
from Random_forest.criterion.criterion import gini_impurity, full_gini_compute


class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    def test_categorical_gini_criterion(self) -> bool:
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
        X=np.random.choice(random_categorical_values,size=(50, 5))
        y=np.random.choice(target, size=(50, 1))   
        selected_column, gini_score =full_gini_compute(X, y)

        self.assertEqual(gini_impurity(y, y), 0) 
        self.assertTrue(selected_column in range(X.shape[1]))
        self.assertTrue(((gini_score<=1) and (gini_score>=0)))

if __name__ == "__main__":
    unittest.main()
