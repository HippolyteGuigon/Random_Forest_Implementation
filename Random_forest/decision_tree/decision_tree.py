import numpy as np
import warnings

from Random_forest.criterion.criterion import gini_impurity_categorical,\
compute_gini_numerical, variance_reduction_numerical,\
variance_reduction_categorical
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")

def treshold_numeric(data: float, reference_value: float)->bool:
    """
    The goal of this function is to compare 
    a given data to a reference value to check
    whether it should go left or right node
    
    Arguments:
        -data: float: The data to be attributed
        left or right Node
        -reference_value: float: The treshold it
        is compared with
    """
    if data<reference_value:
        return True
    else:
        return False
    
def split_categorical(data: str, reference_value: str)->bool:
    """
    The goal of this function is to compare 
    a given data to a reference value to check
    whether it should go left or right node
    
    Arguments:
        -data: str: The data to be attributed
        left or right Node
        -reference_value: str: The categorical value
        it is compared with
    """
    return data==reference_value

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
            criterion_scores=[(a[1], a[0]) for a in criterion_scores]

        split_column=np.argmin(np.array(a[1] for a in criterion_scores))
        
        if isinstance(criterion_scores[0], (float, int)):
            chosen_criteria=sorted(criterion_scores)[0]
        else:
            criterion_scores=sorted(criterion_scores, key=lambda x: x[0])
            chosen_criteria=criterion_scores[0][1]

        if isinstance(chosen_criteria, (float, int)):
            self.condition = staticmethod(treshold_numeric).__func__
        else:
            self.condition = staticmethod(split_categorical).__func__
        self.split_value=chosen_criteria
        self.split_column=split_column

    def check_condition(self, data)->bool:
        """
        The goal of this function is to check if 
        a given data verifies the condition of the 
        Node and as a consequence, is sent in the left
        Node or right Node
        
        Arguments: 
            -data: (int, float, str): The data to be checked
        Returns: 
            -bool: True or False wheter the condition is respected
            or not
        """

        return self.condition(data, reference_value=self.split_value)

    def get_data_subsets(self)->None:
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
        
        self.X=np.hstack((self.X, self.y))
        if self.condition==treshold_numeric:
            self.X_left_node=self.X[vf(self.X[:, self.split_column].astype(float), reference_value=self.split_value)]
            self.X_right_node=self.X[~vf(self.X[:, self.split_column].astype(float), reference_value=self.split_value)]
            self.y_left_node=self.X_left_node[:, -1]
            self.y_right_node=self.X_right_node[:, -1]
        else:
            self.X_left_node=self.X[vf(self.X[:, self.split_column], reference_value=self.split_value)]
            self.X_right_node=self.X[vf(self.X[:, self.split_column], reference_value=self.split_value)]
            self.y_left_node=self.X_left_node[:, -1]
            self.y_right_node=self.X_right_node[:, -1]    


class Decision_Tree:

    def __init__(self, max_depth: int) -> None:
        pass