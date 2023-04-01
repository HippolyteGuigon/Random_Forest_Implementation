import numpy as np
import sys 
import warnings

sys.path.insert(0, "Random_forest/criterion")
from criterion import gini_impurity_categorical, compute_gini_numerical,\
variance_reduction_numerical, variance_reduction_categorical

warnings.filterwarnings("ignore")

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
    def __init__(self) -> None:
        self.left=None
        self.right=None 
    
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
                if isinstance(y.flatten()[0], (np.int_, np.float_)):
                    variance=variance_reduction_numerical(X[:, col].reshape(-1, 1), y)
                    variance_reduction.append(variance)
                else:
                    gini=compute_gini_numerical(X[:, col].reshape(-1, 1), y)
                    gini_scores.append(gini)
            else:
                if isinstance(y.flatten()[0], (np.int_, np.float_)):
                    variance=variance_reduction_categorical(X[:, col].reshape(-1, 1), y)
                    variance_reduction.append(variance)
                else:
                    gini=gini_impurity_categorical(X[:, col].reshape(-1, 1), y)
                    gini_scores.append(gini)

        if len(variance_reduction)>len(gini_scores):
            criterion_scores= variance_reduction
        else:
            criterion_scores= gini_scores

        criterion_scores=sorted(criterion_scores, key=lambda x: x[1])
        chosen_criteria=criterion_scores[0][0]

        if isinstance(chosen_criteria, (float, int)):
            self.condition = staticmethod(treshold_numeric)
        else:
            self.condition = staticmethod(split_categorical)
        self.treshold=chosen_criteria

    def check_condition(self, data)->bool:
        if self.condition(data, reference_value=self.treshold):
            return True
        else:
            return False
