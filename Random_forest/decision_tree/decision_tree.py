import numpy as np
import warnings

from Random_forest.criterion.criterion import gini_impurity_categorical,\
compute_gini_numerical, variance_reduction_numerical,\
variance_reduction_categorical
from Random_forest.decision_tree.array_functions import float_array_converter, treshold_numeric, \
split_categorical, is_float
from Random_forest.configs.confs import load_conf
from sklearn.exceptions import NotFittedError
from typing import List

warnings.filterwarnings("ignore")

main_params = load_conf("configs/main.yml", include=True)

max_depth=main_params["model_hyperparameters"]["max_depth"]
min_sample_split=main_params["model_hyperparameters"]["min_sample_split"]

bottom_values=[]

def get_bottom_values(node)->List[int]:
    """
    The goal of this function is to get 
    all the values that are at the last 
    level oof the binary tree, in the 
    lowest leaf
    
    Arguments:
        -node: The parent node from which 
        the operation will beegin
    Returns:
        -bottom_values: List[int]: The list
        of values of all lowest leaf nodes
    """
    
    global bottom_values
    if (node.left is None) and (node.right is None):
        bottom_values.append(node.X.shape[0])
    elif (node.left is None) and node.right:
        get_bottom_values(node.right)
    elif node.left and (node.right is None):
        get_bottom_values(node.left)
    else:
        get_bottom_values(node.left)
        get_bottom_values(node.right)
    return bottom_values


class Node:
    """
    The goal of this class is to compute a Node 
    with appropriate conditions for the construction
    of a decision Tree

    Arguments:
        -X: np.array: The feature values of the node
        -y: np.array: The target values of the nodes

    Returns:
        None
    """
    def __init__(self, X: np.array, y: np.array) -> None:
        self.left=None
        self.right=None 
        self.X=X
        self.y=y
        self.profondeur=0
        self.data=[]

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
        
        for score_couple_index in range(len(criterion_scores)):
            if isinstance(criterion_scores[score_couple_index][0], str):
                criterion_scores[score_couple_index][0], criterion_scores[score_couple_index][1]\
                =criterion_scores[score_couple_index][1], criterion_scores[score_couple_index][0]
        
        criterion_scores=[tuple(x) for x in criterion_scores]
        split_column=np.argmin([float(x[0]) for x in criterion_scores])
        if isinstance(criterion_scores[0], (float, int)):
            min_index=np.argmin(criterion_scores)
            chosen_criteria=criterion_scores[min_index]
        else:
            min_index=np.argmin([x[0] for x in criterion_scores])
            chosen_criteria=criterion_scores[min_index][1]
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
        
        if self.condition==treshold_numeric:
            X_left_node=self.X[vf(self.X[:, self.split_column].astype(float), reference_value=self.split_value)]
            X_right_node=self.X[~(vf(self.X[:, self.split_column].astype(float), reference_value=self.split_value))]
            y_left_node=self.y[vf(self.X[:, self.split_column].astype(float), reference_value=self.split_value)].reshape(-1,1)
            y_right_node=self.y[~(vf(self.X[:, self.split_column].astype(float), reference_value=self.split_value))].reshape(-1,1)
            
        else:
            X_left_node=self.X[vf(self.X[:, self.split_column], reference_value=self.split_value)]
            X_right_node=self.X[~(vf(self.X[:, self.split_column], reference_value=self.split_value))]
            y_left_node=self.y[vf(self.X[:, self.split_column], reference_value=self.split_value)].reshape(-1,1)
            y_right_node=self.y[~(vf(self.X[:, self.split_column], reference_value=self.split_value))].reshape(-1,1)
        
        self.X_left_node=X_left_node
        self.X_right_node=X_right_node
        self.y_left_node=y_left_node
        self.y_right_node=y_right_node  

class Decision_Tree:
    """
    The goal of this class is to elaborate, 
    from the Node class computed above, a 
    Decision Tree consisting of multiple nodes
    with different conditions

    Arguments:
        -X: np.array: The array to be fitted on
        -y: np.array: The target array
        -max_depth: int: The maximum depth the Tree
        can reach 
        -min_samples_split: int: The minimum number 
        of samples required to split an internal node

    Returns:
        None
    """

    def __init__(self, X: np.array, y: np.array, max_depth: int=max_depth, 
    min_samples_split: int = min_sample_split) -> None:
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.X=X
        self.y=y
        self.node=Node(X, y)

    def grow_node(self, node)->None:
        """
        The goal of this this function is to
        build an entire Decision Tree performing
        recursion method 

        Arguments:
            -node

        Returns:
            None
        """

        if (node.X.shape[0]>=self.min_samples_split) and (node.profondeur<self.max_depth):
            
            node.compute_condition()
            node.get_data_subsets()
            node.left=Node(node.X_left_node, node.y_left_node)
            node.right=Node(node.X_right_node, node.y_right_node)
            node.left.profondeur=node.profondeur+1
            node.right.profondeur=node.profondeur+1

            if node.left.X.shape[0]>=self.min_samples_split:
                self.grow_node(node.left)
            if node.right.X.shape[0]>=self.min_samples_split:
                self.grow_node(node.right)

    