import numpy as np
import sys 

sys.path.insert(0, "Random_forest/criterion")
from criterion import gini_impurity_categorical, compute_gini_numerical, variance_reduction_numerical, variance_reduction_categorical

def treshold_numeric(data, reference_value):
    if data<reference_value:
        return True
    else:
        return False
    
def split_categorical(data, reference_value):
    if data==reference_value:
        return True
    else:
        return False

def is_float(x):
    try:
        x=float(x)
        return True
    except ValueError:
        return False
    
class Node:
    """
    The goal of this class is to compute a Node 
    with appropriate conditions for the construction
    of a decision Tree
    """
    def __init__(self, data) -> None:
        self.left=None
        self.right=None 
        self.data=data

        #if isinstance(self.data, (float, int)):
        #    self.condition = staticmethod(treshold_numeric)
        #else:
        #    self.condition = staticmethod(split_categorical)

        #if self.condition(self.data, reference_value=3):
        #    print("Ok")
        #else:
        #    print("Not ok")
    
    def compute_condition(self, X:np.array, y: np.array)->None:
        """
        The goal of this function is to compute the appropriate
        condition for a given Node
        
        Arguments:
            -X: np.array: The numpy array with the sets of columns
            from which the appropriate condition will be deducted
            -y: np.array: The target column
            
        Returns:
            None
        """
        variance_reduction=[]
        gini_scores=[]
        
        X.sort()

        for col in range(X.shape[1]):
            if is_float(X[0, col]):
                print("Numerique")
                if isinstance(y[0], (np.int_, np.float_)):
                    variance=variance_reduction_numerical(X[:, col].reshape(-1, 1), y)
                    variance_reduction.append(variance)
                else:
                    gini=compute_gini_numerical(X[:, col].reshape(-1, 1), y)
                    gini_scores.append(gini)
            else:
                print("Categorique")
                if isinstance(y[0], (np.int_, np.float_)):
                    variance=variance_reduction_categorical(X[:, col].reshape(-1, 1), y)
                    variance_reduction.append(X, y)
                else:
                    gini=gini_impurity_categorical(X[:, col].reshape(-1, 1), y)
                    gini_scores.append(gini)
        return variance_reduction, gini_scores



