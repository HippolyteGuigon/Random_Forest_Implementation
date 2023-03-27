import unittest
import numpy as np
from Random_forest.criterion.criterion import gini_impurity, full_gini_compute, variance_reduction


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
        X_numerical_check=np.random.choice(random_categorical_values, size=(50,1))
        y_categorical=np.random.choice(target, size=(50, 1)) 
        
        selected_column, gini_score =full_gini_compute(X, y_categorical)
        
        self.assertEqual(gini_impurity(y_categorical, y_categorical), 0) 
        self.assertTrue(selected_column in range(X.shape[1]))
        self.assertTrue(((gini_score<=1) and (gini_score>=0)))
        
        

    def test_variance_reduction(self):
        """"
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
        X=np.random.choice(random_categorical_values, size=(50,1))
        y=np.random.uniform(size=(50, 1))  
        best_candidate=variance_reduction(X,y)
        
        self.assertTrue((best_candidate[0] in np.unique(X)))

if __name__ == "__main__":
    unittest.main()
