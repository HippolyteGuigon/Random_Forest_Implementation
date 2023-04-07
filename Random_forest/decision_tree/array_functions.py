import numpy as np

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

def float_array_converter(arr: np.array)->np.array:

    convertArr = []
    for s in arr.ravel():    
        try:
            value = np.float_(s)
        except ValueError:
            value = s

        convertArr.append(value)

    return np.array(convertArr,dtype=object).reshape(arr.shape)