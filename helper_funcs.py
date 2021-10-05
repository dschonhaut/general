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
7/17/21
"""
import numpy as np
from collections import OrderedDict as od


def invert_dict(d):
    """Invert a dictionary of string keys and list values."""
    if type(d) == dict:
        newd = {}
    else:
        newd = od([])
    for k, v in d.items():
        for x in v:
            newd[x] = k
    return newd


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
    if isinstance(obj_in, str):
        obj_out = [obj_in]
    else:
        obj_out = obj_in.copy()

    for _i in range(len(obj_out)):
        for old_str, new_str in replace_vals.items():
            obj_out[_i] = obj_out[_i].replace(old_str, new_str)
    
    if isinstance(obj_in, str):
        obj_out = obj_out[0]
            
    return obj_out


def strip_space(in_str):
    """Strip 2+ adjoining spaces down to 1."""
    out_str = in_str
    for iSpace in range(len(in_str), 1, -1):
        search_str = ' ' * iSpace
        out_str = out_str.replace(search_str, ' ')
    return out_str

    
def circit(val,
           prop='r',
           scale=1):
    """Solve for the properties of, and/or transform, a circle.
    
    Parameters
    ----------
    val : number > 0
        Value of the input circle property.
    prop : str
        'r' = radius
        'd' = diameter
        'a' = area
        'c' = circumference
    scale : number > 0
        Applies val *= scale to the output circle.
    
    Returns
    -------
    circle : dict
        Contains r, d, a, and c versus the input circle.
    """
    # Transform the output circle.
    val *= scale

    # Solve the circle's properties.
    if prop == 'r':
        r = val
        d = r * 2
        a = np.pi * np.square(r)
        c = 2 * np.pi * r
    elif prop == 'd':
        d = val
        r = d / 2
        a = np.pi * np.square(r)
        c = 2 * np.pi * r
    elif prop == 'a':
        a = val
        r = np.sqrt(a / np.pi)
        d = r * 2
        c = 2 * np.pi * r
    elif prop == 'c':
        c = val
        r = c / (2 * np.pi)
        d = r * 2
        a = np.pi * np.square(r)
    
    # Store the outputs.
    circle = {'r': r,
              'd': d,
              'a': a,
              'c': c}
    
    return circle
