import numpy as np
import json
import omegaconf

def calculate_mode(arr, axis):
    """calculates mode in a given array"""
    vals, counts = np.unique(arr, return_counts=True, axis=axis)
    max_occurance = np.nanargmax(counts)
    agg = vals[max_occurance]
    return agg

def nested_set(dic, keys, value,enforce_old_type=True):
    """set value in nested dict
    
    >>> d = {}
    >>> nested_set(d, ['person', 'address', 'city'], 'New York')
    >>> d
    {'person': {'address': {'city': 'New York'}}}
    
    """
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    if enforce_old_type:
        type_ = type(dic[keys[-1]])
        
        if type_ in (list,tuple,dict,omegaconf.listconfig.ListConfig):
            value = json.loads(value)
        elif type_ in [int,float,bool,str,np.float64]:
            value = type_(value)    
    dic[keys[-1]] = value

