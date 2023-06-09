# Random_Forest_Implementation
The goal of this repository is to create an implementation in Python of the Random Forest algorithm

## Build Status

For the moment, the first version of the model is coded and ready to be used

However, it is still slow for the moment and the next steps is to reduce time complexity while making it more user friendly.

Throughout its construction, if you see any improvements that could be made in the code, do not hesitate to reach out at
Hippolyte.guigon@hec.edu

## Code style

The all project is coded under PEP-8 (https://peps.python.org/pep-0008/) and flake8 (https://pypi.org/project/flake8/) compliancy. Such compliance is verified during commits with pre-commits file ```.pre-commit-config.yaml```

## Installation

* This project uses a specific conda environment, to get it, run the following command: ```conda env create -f random_forest_env.yml```

* To install all necessary libraries, run the following code:```pip install -r requirements.txt```

* This project has its own package that is used. To get it, run the following command: ```python install setup.py```


## Screenshot

![alt text](https://github.com/HippolyteGuigon/Random_Forest_Implementation/blob/main/ressources/random_forest.png)

Image of a Random Forest decision process

## How to use ?

1. To import the model, run the following command: ```from Random_forest.model.model import RandomForest```

2. Then, use ```model=RandomForest()``` and adjust hyperparameters according to your goals. Not that if you want to use RandomForest for regression, you'll have to set ```objective="regression"``` and ```objective="classification"``` according to which suprvised method you wish to use

3. Run ```model.fit(X, y)```

4. Predict your data with ```model.predict(X_to_predict)```