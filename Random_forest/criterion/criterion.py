"""https://enjoymachinelearning.com/blog/gini-index-vs-entropy/"""
import numpy as np

def gini_impurity(X: np.array, y: np.array)->tuple:
    """
    The goal of this function is to take two numpy arrays, 
    and to compute the gini impurity of one related to the 
    other
    
    Arguments: 
        -X: np.array: The numpy array to be 
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

    full_gini_impurities=[(col, gini_impurity(X[:, col].reshape(-1, 1), y)) for col in range(X.shape[1])]
    split_pair = sorted(full_gini_impurities, key=lambda x: x[1])[0]
    return split_pair[0], split_pair[1]

def variance_reduction(X: np.array, y: np.array)->tuple:
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
        n_first_subset=first_subset.shape[0]
        n_second_subset=second_subset.shape[0]
        total_variance = (n_first_subset/n)*np.var(first_subset)+(n_second_subset/n)*np.var(second_subset)
        variance_candidates.append((candidate, total_variance))
    variance_candidates=sorted(variance_candidates,key=lambda x: x[1])
    best_candidate=variance_candidates[0]
    
def entropy(x):
    pass