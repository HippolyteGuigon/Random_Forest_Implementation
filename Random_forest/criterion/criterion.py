"""https://enjoymachinelearning.com/blog/gini-index-vs-entropy/"""
import numpy as np
from typing import Tuple

def gini_impurity_categorical(X: np.array, y: np.array)->tuple:
    """
    The goal of this function is to take two numpy arrays, 
    and to compute the gini impurity of one related to the 
    other
    
    Arguments: 
        -X: np.array: The numpy array to be assessed
        -y: np.array: The target column to be compared with 
        the other columns
    
    Returns:
        -gini: np.array: The gini impurity of the column related
        to the other 
    """
    full_data=np.hstack((X, y))
    n=X.shape[0]
    gini=0
    unique_target=np.unique(y, return_counts=False)

    for value in np.unique(X, return_counts=False):
        gini_impurity=1
        X_filtered=full_data[full_data[:, 0]==value, :]
        n_value=X_filtered.shape[0]
        for unique_subtarget in unique_target:
            gini_impurity-=(X_filtered[X_filtered[:,1]==unique_subtarget].shape[0]/n_value)**2

        gini+=(n_value/n)*gini_impurity
    
    return gini

def compute_gini_numerical(X: np.array, y: np.array)->tuple:
    """
    The goal of this function is to compute the gini score 
    of a numerical array to perform the split of a categorical
    array
    
    Arguments:
        -X: np.array: The numerical column which gini score 
        will be computed 
        -y: np.array: The target categorical column which 
        will be compared to the numerical column
    Returns:
        -best_candidate: tuple: The gini score of the X 
        column
    """

    gini_values=np.array([])
    unique_targets=np.unique(y, return_counts=False)
    
    full_data=np.hstack((X, y))
    n=full_data.shape[0]
    full_data=full_data[full_data[:, 0].astype(int).argsort()]
    tresholds=np.array([(np.float(full_data[i, 0])+np.float(full_data[i+1, 0]))/2 for i in range(full_data.shape[0]-1)])
    tresholds=np.unique(tresholds, return_counts=False)
    
    for treshold in tresholds:
        left_node=full_data[full_data[:, 0].astype(int)<treshold][:, 1]
        right_node=full_data[full_data[:, 0].astype(int)>=treshold][:, 1]
        n_left_node=left_node.shape[0]
        n_right_node=right_node.shape[0]
        impurity_right=1
        impurity_left=1

        for value in unique_targets:
            impurity_right-=(right_node[right_node==value].shape[0]/n_right_node)**2
            impurity_left-=(left_node[left_node==value].shape[0]/n_left_node)**2
        total_impurity=impurity_right*(n_right_node/n)+impurity_left*(n_left_node/n)
        gini_values=np.append(gini_values, total_impurity)
    best_treshold_split=tresholds[gini_values.argmin()]

    return best_treshold_split

def variance_reduction_numerical(X: np.array, y: np.array)->Tuple[float, float]:
    """
    The goal of this function is to decide which value 
    in a numerical column is the best treshold for splitting 
    by calculating the variance of each subset 
    
    Arguments:
        -X: np.array: The column composed of multiple numerical 
        variables 
        -y: The target column 
    
    Returns:
        -best_split: tuple: The best collumn with lowest variance 
    """
    
    variance_candidates = []

    full_data=np.hstack((X, y))
    n=len(full_data)
    full_data=full_data[full_data[:, 0].astype(int).argsort()]
    tresholds=np.array([(np.float(full_data[i, 0])+np.float(full_data[i+1, 0]))/2 for i in range(full_data.shape[0]-1)])
    tresholds=np.unique(tresholds, return_counts=False)

    for treshold_candidate in tresholds:
        left_node=full_data[full_data[:, 0].astype(int)<treshold_candidate][:, 1]
        right_node=full_data[full_data[:, 0].astype(int)>=treshold_candidate][:, 1]
        n_left_node=len(left_node)
        n_right_node=len(right_node)
        total_variance = (n_left_node/n)*np.var(left_node)+(n_right_node/n)*np.var(right_node)
        variance_candidates.append((treshold_candidate, total_variance))

    variance_candidates=sorted(variance_candidates,key=lambda x: x[1])
    best_candidate=variance_candidates[0]

    return best_candidate

def variance_reduction_categorical(X: np.array, y: np.array)->tuple:
    """
    The goal of this function is to decide which value 
    in a categorical column to choose for splitting 
    by calculating the variance of each subset 
    
    Arguments:
        -X: np.array: The column composed of multiple categorical 
        variables 
        -y: The target column 
    
    Returns:
        -best_split: tuple: The best collumn with lowest variance 
    """

    variance_candidates = []
    split_candidate=np.unique(X, return_counts=False)
    full_data=np.hstack((X, y))
    for candidate in split_candidate:
        first_subset=full_data[full_data[:, 0]==candidate, -1].astype(float)
        second_subset=full_data[full_data[:,0]!=candidate, -1].astype(float)
        n=full_data.shape[0]
        n_first_subset=len(first_subset)
        n_second_subset=len(second_subset)
        total_variance = (n_first_subset/n)*np.var(first_subset)+(n_second_subset/n)*np.var(second_subset)
        variance_candidates.append((candidate, total_variance))
    variance_candidates=sorted(variance_candidates,key=lambda x: x[1])
    best_candidate=variance_candidates[0]

    return best_candidate

def full_gini_compute(X: np.array, y: np.array)->tuple:
    """
    The goal of this function is to compute the 
    column with the lowest gini impurity score in 
    order to select the best candidate for splitting

    Arguments:
        -X: np.array: The set of columns to be compared
        with the target column 
        -y: np.array: The target column 

    Returns: 
        -best_split_candidate: tuple: The tuple containing
        the column number and the gini impurity of the best 
        column to perform a split
    """

    full_gini_impurities=[(col, gini_impurity_categorical(X[:, col].reshape(-1, 1), y)) for col in range(X.shape[1])]
    split_pair = sorted(full_gini_impurities, key=lambda x: x[1])[0]
    return split_pair[0], split_pair[1]

def full_variance_reduction_compute(X: np.array, y: np.array)->tuple:
    """
    The goal of this function is to compute the best categorical 
    column to split the values of a numerical column
    
    Arguments:
        -X: np.array: The full array containing the 
        categorical columns
        -y: np.array: The target numerical array 
    Returns:
        -best_split_candidate: tuple: The best column 
        with lowest variance
    """ 

    best_candidates=[(col, variance_reduction_categorical(X[:, col].reshape(-1, 1), y)) for col in range(X.shape[1])]
    best_candidates=sorted(best_candidates, key=lambda x: x[1][1])
    best_split_col=best_candidates[0]

    return best_split_col

def entropy(x):
    pass