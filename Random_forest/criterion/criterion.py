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
    # store all of our columns and gini scores
    gini_scores = []
    
    # iterate through each column in your dataframe
    for col in range(X.shape[0]):
        if col==X.shape[1]-1:
            break
        
        # skip our target column
        # no information gain on target columns!
        # we can't split here
        
        # resets for each column in your dataset
        gini = 0
        
        # get the value counts for that column
        unique, counts = np.unique(X, return_counts=True)
        # iterate through each unique value for that column
        
        for key, val in zip(unique, counts):
            
            # get the target variable seperated, based on
            # the independent variable
            filtered_array = X[X[:,col]==key][:, -1]
            
            # need n for the length
            n = X.shape[0]
            
            # sum of the value counts for that column
            ValueSum = filtered_array.shape[0]
            # need the probabilities of each class
            p = 0
            
            # we now have to send it to our gini impurity formula
            unique, counts = np.unique(filtered_array, return_counts=True)
            for i,j in zip(unique, counts):
                p += (j / ValueSum) ** 2
            
            # gini total for column 
            # is all uniques from each column
            gini += (val / n) * (1-p)
        
        # append our column name and gini score
        gini_scores.append((col,gini-1))
    
    # sort our gini scores lowest to highest
    split_pair = sorted(gini_scores, key=lambda x: -x[1], reverse=True)[0]
    print(f'''Split on {split_pair[0]} With Gini Index of {round(split_pair[1],3)}''')
    return split_pair 


def entropy(x):
    pass