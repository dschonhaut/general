"""
String manipulation methods.
"""

import os.path as op
from glob import glob
import re


def add_presuf(infile, prefix=None, suffix=None):
    """Return full filepath with optional basename prefix/suffix.

    Parameters
    ----------
    infile : str
        The filepath to modify.
    prefix : str | None
        String to prepend to the basename. If None, no prefix is added.
    suffix : str | None
        String to append to the basename. If None, no suffix is added.

    Returns
    -------
    outfile : str
        The modified filepath.
    """
    infile = op.normpath(infile)
    if (prefix is None) and (suffix is None):
        return infile
    base = op.basename(infile)
    is_hidden = base[0] == "."
    if is_hidden:
        if isinstance(prefix, str):
            prefix = "." + prefix
        else:
            prefix = "."
        base = base[1:]
    else:
        if not isinstance(prefix, str):
            prefix = ""
    sbase = base.split(".")
    if isinstance(suffix, str):
        sbase[0] = sbase[0] + suffix
    outfile = op.join(op.dirname(infile), prefix + ".".join(sbase))
    return outfile


def split(string, delimiters):
    """Split a string into a list of substrings by 1+ delimiters."""
    if isinstance(delimiters, str):
        delimiters = [delimiters]

    pattern = "[" + re.escape("".join(delimiters)) + "]"
    return [substring for substring in re.split(pattern, string) if len(substring) > 0]


def str_replace(obj_in, replace_vals=None):
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

    for i, _ in enumerate(obj_out):
        for old_str, new_str in replace_vals.items():
            obj_out[i] = obj_out[i].replace(old_str, new_str)

    if isinstance(obj_in, str):
        obj_out = obj_out[0]

    return obj_out


def strip_space(in_str):
    """Strip 2+ adjoining spaces down to 1."""
    out_str = in_str.strip()
    for i in range(len(in_str), 1, -1):
        search_str = " " * i
        out_str = out_str.replace(search_str, " ")

    return out_str


def glob_sort_mtime(pattern):
    """Return files matching pattern in most recent modified order.

    Returns
    -------
    files : list of str
        List of files matching pattern, sorted by most recent modified
        (files[0] is the most recently modified).
    """
    files = sorted(glob(pattern), key=op.getmtime, reverse=True)
    return files
