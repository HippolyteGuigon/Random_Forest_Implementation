import numpy as np
import sys
from sklearn.utils import check_random_state
from Random_forest.decision_tree.decision_tree import Decision_Tree
from Random_forest.decision_tree.array_functions import is_float
from Random_forest.configs.confs import load_conf

main_params=load_conf("configs/main.yml", include=True)
max_depth=main_params["model_hyperparameters"]["max_depth"]
min_sample_split=main_params["model_hyperparameters"]["min_sample_split"]

class RandomForest:
    """
    The goal of this class is the 
    implementation of the final model
    
    Arguments:
        -random_state: int: The random_state
        used for data generation
        -max_depth: int: The maximum depth of
        each decision Tree composing the forest
        -min_sample_split: The minimum number of 
        samples required for having a split in the
        Node
        
    Returns:
        -None
    """
    def __init__(self, random_state=42, max_depth: int=max_depth, 
                 min_sample_split: int = min_sample_split, **kwargs) -> None:
        
        self.rng = check_random_state(random_state)
        self.max_depth=max_depth
        self.min_samples_split=min_sample_split

        for param, value in kwargs.items():
            if param not in main_params.keys():
                raise AttributeError(f"The Random Forest model has\
                                      no attribute {param}")
            else:
                setattr(self, param, value)

        for param, value in main_params["model_hyperparameters"].items():
            if not hasattr(self, param):
                setattr(self, param, value)

    def data_bootstrap(self, X: np.array, y: np.array)->np.array:
        """
        The goal of this function is to build a
        bootstraped dataset from the original one
        
        Arguments:
            -X: np.array: The original dataset
            -y: np.array: The target column
        Returns:
            -bootstraped_data: np.array: The
            bootstraped dataset
        """

        full_data=np.hstack((X,  y))
        n=full_data.shape[0]
        choosed_sample_index=sorted(np.random.choice(np.arange(0, n), size=(n, 1)).flatten())
        full_data_bootstraped=full_data[choosed_sample_index,:]
        X_bootstraped,  y_bootstraped = full_data_bootstraped[:, :-1], full_data_bootstraped[:, -1]
        return X_bootstraped, y_bootstraped.reshape(-1, 1)
    
    def fit(self, X: np.array, y: np.array)->None:
        """
        The goal  of this function is to fit the
        model, build a forest of Decision Trees
        that will be used afterwhise for the prediction 
        of new values
        
        Arguments: 
            -X: np.array: The array of values on which 
            the model will be fitted 
            -y: np.array: The target column
        Returns:
            -None
        """
        
        self.X=X
        self.y=y

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f"X and y must have the same number of rows, X\
                              contains {self.X.shape[0]} and y {self.y.shape[0]}")
        if self.y.shape[1]>1:
            raise ValueError(f"The target dataset must only contain\
                              one column, it contains {self.y.shape[1]}")
        if self.X.shape[0]<=self.min_samples_split:
            raise AssertionError(f"The number of rows\
            of X, {self.X.shape[0]} must be superior to min_sample_split\
                 hyperparameter {self.min_samples_split}")
        
        bootstraped_set=[self.data_bootstrap(self.X, self.y) for _ in range(0,self.n_estimators)]
        self.bootstraped_set=bootstraped_set
        model_set=[Decision_Tree(x[0], x[1]) for x in bootstraped_set]
        
        for x in model_set:
            x.grow_node(x.node)
        
        self.model_set=model_set

    def to_predict_data_allocation(self, data: np.array, node)->float:
        """
        The goal of this function is to 
        allocate the data to be predicted
        in a given decision tree in order
        to make a prediction after 
        
        Arguments: 
        -data: The data to be allocated 
        -node: The node in which the data
        will be splitted
        
        Returns: 
            -decision: The outcome of the tree
        """

        if node.left or node.right:
            if is_float(data[node.split_column]):
                if node.check_condition(data[node.split_column].astype(float)):
                    self.to_predict_data_allocation(data, node.left)
                else:
                    self.to_predict_data_allocation(data, node.right)    
            else:
                if node.check_condition(data[node.split_column]):
                    self.to_predict_data_allocation(data, node.left)
                else:
                    self.to_predict_data_allocation(data, node.right)  
            
        if is_float(node.y[0]):
            decision= np.mean(node.y.astype(float))
        else:
            values, counts = np.unique(node.y, return_counts=True)
            decision= values[counts.argmax()]
        return decision
    
    def individual_predict(self, X_to_predict: np.array)->float:
        """
        The goal of this function is to
        return the predictions made for
        a given array
        
        Arguments:
            -X_to_predict: np.array: The
            array of values to be predicted
        Returns:
            -predictions: np.array: The predictions
            made for a given array
        """
        
        predictions=[]

        for decision_tree in self.model_set:
            print("Iciiiii",X_to_predict,self.to_predict_data_allocation(X_to_predict,decision_tree.node))
            predictions.append(self.to_predict_data_allocation(X_to_predict,decision_tree.node))
        if isinstance(predictions[0], (float, int)):
            prediction= np.mean(predictions)
        else:
            values, counts = np.unique(predictions, return_counts=True)
            prediction= values[counts.argmax()] 
        return prediction
    
    def predict(self, X:np.array)->np.array:
        """
        The goal of this function is to
        predict a all array
        
        Arguments:
            -X: np.array: The array to be
            predicted
        
        Returns:
            -predicted: np.array: The prediction
            made by the model
        """

        if not hasattr(self, "model_set"):
            raise AssertionError("The model needs to be fitted\
                                 first before predicting")
        predicted=[]
        for x in X:
            predicted.append(self.individual_predict(x))

        return predicted