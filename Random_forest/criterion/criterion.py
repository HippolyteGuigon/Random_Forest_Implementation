"""https://enjoymachinelearning.com/blog/gini-index-vs-entropy/"""
import numpy as np

def gini(X: np.array)->tuple:
    """
    The goal of this function is to take a numpy array, 
    column by column and to compute the gini coefficient
    of each column compared to the target column
    
    Arguments: 
        -X: np.array: The set of data to be compared with 
        the target column
    
    Returns:
        -split_pair: tuple: The column with the smallest 
        gini index compared to the target column"""
    gini_scores = []
    
    for col in range(X.shape[0]):
        if col==X.shape[1]-1:
            break
        
        gini = 0
        
        unique, counts = np.unique(X, return_counts=True)
        
        for key, val in zip(unique, counts):
            
            filtered_array = X[X[:,col]==key][:, -1]
            
            n = X.shape[0]
            
            ValueSum = filtered_array.shape[0]
            p = 0
            
            unique, counts = np.unique(filtered_array, return_counts=True)
            for i,j in zip(unique, counts):
                p += (j / ValueSum) ** 2
            
            gini += (val / n) * (1-p)
        
        gini_scores.append((col,gini-1))
    split_pair = sorted(gini_scores, key=lambda x: -x[1], reverse=True)[0]
    return split_pair 


def entropy(x):
    pass