"""
helper_funcs.py

Author
------
Daniel Schonhaut
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com

Description
----------- 
Helper functions for common tasks with simple python classes.

Last Edited
----------- 
2/26/20
"""


def str_replace(obj_in, 
                replace_vals=None):
    """Multi-string replacement for strings and lists of strings.
    
    Parameters
    ----------
    obj_in : str or list[str]
        A single string or a list of strings
        with values that you want to replace.
    replace_vals : dict or OrderedDict
        {old_value: new value, ...} in the order given.
    
    Returns 
    -------
    obj_out : str or list[str]
        Same as obj_in but with values replaced.
    """
    obj_out = list(obj_in)
    n_items = len(obj_out)
    for _i in range(n_items):
        for old_str, new_str in replace_vals.items():
            obj_out[_i] = obj_out[_i].replace(old_str, new_str)
    
    if n_items == 1:
        obj_out = obj_out[0]
            
    return obj_out
