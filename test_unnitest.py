import unittest
import numpy as np
from Random_forest.criterion.criterion import gini_impurity_categorical, full_gini_compute, variance_reduction_categorical, variance_reduction_numerical


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
        
        self.assertEqual(gini_impurity_categorical(y_categorical, y_categorical), 0) 
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
        random_numerical_values = np.random.normal(size=(50, 1), scale=50)
        random_numerical_values = np.array(sorted(random_numerical_values))
        X=np.random.choice(random_categorical_values, size=(50,1))
        y=np.random.uniform(size=(50, 1))  
        best_candidate_categorical=variance_reduction_categorical(X,y)
        best_candidate_numerical=variance_reduction_numerical(random_numerical_values, y)
        tresholds=np.array([(np.float(random_numerical_values[i])+np.float(random_numerical_values[i+1]))/2 for i in range(random_numerical_values.shape[0]-1)])

        self.assertTrue((best_candidate_categorical[0] in np.unique(X)))
        self.assertTrue((best_candidate_numerical[0] in tresholds))

if __name__ == "__main__":
    unittest.main()
