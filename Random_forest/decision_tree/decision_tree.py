import numpy as np
import warnings

from Random_forest.criterion.criterion import gini_impurity_categorical,\
compute_gini_numerical, variance_reduction_numerical,\
variance_reduction_categorical
from Random_forest.decision_tree.array_functions import float_array_converter, treshold_numeric, \
split_categorical, is_float
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")

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
            X_left_node=self.X[vf(self.X[:, self.split_column].astype(float), reference_value=self.split_value)]
            X_right_node=self.X[~(vf(self.X[:, self.split_column].astype(float), reference_value=self.split_value))]
            y_left_node=X_left_node[:, -1].reshape(-1, 1)
            y_right_node=X_right_node[:, -1].reshape(-1, 1)
            X_left_node=X_left_node[:, :-1]
            X_right_node=X_right_node[:, :-1]
            
        else:
            X_left_node=self.X[vf(self.X[:, self.split_column], reference_value=self.split_value)]
            X_right_node=self.X[~(vf(self.X[:, self.split_column], reference_value=self.split_value))]
            y_left_node=X_left_node[:, -1].reshape(-1, 1)
            y_right_node=X_right_node[:, -1].reshape(-1, 1)
            X_left_node=X_left_node[:, :-1]
            X_right_node=X_right_node[:, :-1]
        
        self.X=self.X[:, :-1]
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

    def __init__(self, X: np.array, y: np.array, max_depth: int=10, 
    min_samples_split: int = 10) -> None:
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.X=X
        self.y=y
        self.node=Node(X, y)

    def depth(self, node)->int:
        """
        The goal of this function is to
        compute the depth of the Tree 
        
        Arguments:
            None
        
        Returns:
            -tree_depth: int: The computed 
            depth of the binary tree
        """
        
        if node is None:
            return 0
    
        else:
    
            # Compute the depth of each subtree
            lDepth = self.depth(node.left)
            rDepth = self.depth(node.right)
    
            # Use the larger one
            if (lDepth > rDepth):
                return lDepth+1
            else:
                return rDepth+1

    def grow_node(self, node):
        """
        Le but de cette fonction est de passer 
        d'un noeud simple à un noeud avec une feuille
        gauche et une feuille droite
        """
        if node.X.shape[0]>=self.min_samples_split and self.depth(self.node)<self.max_depth:
            
            node.compute_condition()
            node.get_data_subsets()
            node.left=Node(node.X_left_node, node.y_left_node)
            node.right=Node(node.X_right_node, node.y_right_node)
            if node.left.X.shape[0]>=self.min_samples_split:
                self.grow_node(node.left)
            if node.right.X.shape[0]>=self.min_samples_split:
                self.grow_node(node.right)

    def iterate(self, node)->None:
        """
        The goal of this function is, for a given node
        of the Decision Tree, to build both left and right
        nodes provided conditions are respected
        
        Arguments:
            -node: 
        Returns:
            None
        """
        print("Node",node)
        print("Right node", node.right)
        print("Left node", node.left)
        if node.X.shape[0]>=self.min_samples_split:
            node.compute_condition()
            node.get_data_subsets()

            if node.left.X.shape[0]>=self.min_samples_split:

                self.left=Node(node.X_left_node, node.y_left_node)
                self.left.compute_condition()
                self.left.get_data_subsets()
                node.left=self.iterate(self.left)

            if self.right.X.shape[0]>=self.min_samples_split:

                self.right=Node(node.X_right_node, node.y_right_node)
                self.right.compute_condition()
                self.right.get_data_subsets()
                node.right=self.iterate(self.right)


           