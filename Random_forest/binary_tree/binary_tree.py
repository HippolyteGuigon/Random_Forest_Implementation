import numpy as np
import sys 
import warnings

sys.path.insert(0, "Random_forest/criterion")
from criterion import gini_impurity_categorical, compute_gini_numerical,\
variance_reduction_numerical, variance_reduction_categorical
from sklearn.exceptions import NotFittedError

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
    def __init__(self, X: np.array, y: np.array) -> None:
        self.left=None
        self.right=None 
        self.X=X
        self.y=y

    def compute_condition(self)->None:
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
        self.X.sort()

        for col in range(self.X.shape[1]):
            if is_float(self.X[0, col]):
                if isinstance(self.y.flatten()[0], (np.int_, np.float_)):
                    variance=variance_reduction_numerical(self.X[:, col].reshape(-1, 1),self.y)
                    variance_reduction.append(variance)
                else:
                    gini=compute_gini_numerical(self.X[:, col].reshape(-1, 1), self.y)
                    gini_scores.append(gini)
            else:
                if isinstance(self.y.flatten()[0], (np.int_, np.float_)):
                    variance=variance_reduction_categorical(self.X[:, col].reshape(-1, 1), self.y)
                    variance_reduction.append(variance)
                else:
                    gini=gini_impurity_categorical(self.X[:, col].reshape(-1, 1), self.y)
                    gini_scores.append(gini)

        if len(variance_reduction)>len(gini_scores):
            criterion_scores= variance_reduction
        else:
            criterion_scores= gini_scores

        split_column=np.argmin(np.array(a[1] for a in criterion_scores))
        criterion_scores=sorted(criterion_scores, key=lambda x: x[1])
        chosen_criteria=criterion_scores[0][0]

        if isinstance(chosen_criteria, (float, int)):
            self.condition = staticmethod(treshold_numeric)
        else:
            self.condition = staticmethod(split_categorical)
        self.split_value=chosen_criteria
        self.split_column=split_column

    def check_condition(self, data)->bool:
        if self.condition(data, reference_value=self.split_value):
            return True
        else:
            return False

    def get_data_subsets(self):
        """
        The goal of this function is, once the condition
        has been computed, to get the data both for left 
        and right node

        Arguments:
            None
        Returns:
            -X_left_node: np.array: The data that will go 
            in the left node
            -X_right_node: np.array: The data that will go 
            in the right node
        """

        if not hasattr(self, "split_value"):
            raise NotFittedError("The condition for this Node needs to be computed first")
        
        vf = np.vectorize(self.condition)
        if self.condition.__func__==treshold_numeric:
            
            X_left_node=self.X[vf(self.X[:, self.split_column].astype(float), reference_value=self.split_value)]
            X_right_node=self.X[~vf(self.X[:, self.split_column].astype(float), reference_value=self.split_value)]
        else:
            X_left_node=self.X[vf(self.X[:, self.split_column], reference_value=self.split_value)]
            X_right_node=self.X[vf(self.X[:, self.split_column], reference_value=self.split_value)]

        return X_left_node, X_right_node