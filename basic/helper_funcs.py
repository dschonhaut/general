"""
Helper functions for common tasks with simple python classes.
"""
from time import time
from itertools import chain, zip_longest
from collections import OrderedDict as od
import numpy as np
import pandas as pd
from scipy.stats import sem


class Timer(object):
    """I say how long things take to run."""

    def __init__(self, msg=None):
        """Start the global timer."""
        self.reset()
        if msg is None:
            self.msg = "Ran in "
        else:
            self.msg = msg

    def __str__(self):
        """Print how long the global timer has been running."""
        msg = self.msg + "{:.1f}s".format(self.check())
        return msg

    def check(self, reset=False):
        """Report the global runtime."""
        runtime = time() - self.start
        if reset:
            self.reset()
        return runtime

    def loop(self, key=None, verbose=True):
        """Report the loop runtime and reset the loop timer."""
        if not hasattr(self, "loops"):
            self.loops = od([])
        if not hasattr(self, "last_loop_start"):
            self.last_loop_start = self.start
        if key is None:
            key = "loop {}".format(len(self.loops) + 1)

        loop_runtime = time() - self.last_loop_start
        self.loops[key] = loop_runtime
        self.last_loop_start = time()
        if verbose:
            print("{}: {:.1f}s".format(key, self.loops[key]))

    def reset(self):
        """Reset the global timer."""
        self.start = time()


def weave(l1, l2):
    """Interleave two lists of same or different lengths."""
    return [x for x in chain(*zip_longest(l1, l2)) if x is not None]


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


def nunique(vals):
    return len(set(vals))


def count_unique(arr):
    """Return unique arr elements and their counts as a string."""
    counts = pd.Series(arr).value_counts(dropna=False).sort_index()
    return ", ".join(["{}: {}".format(k, v) for (k, v) in counts.items()])


def quartiles(arr, decimals=1):
    """Return each quartile of arr as a list."""
    if decimals == 0:
        output = ", ".join(
            [
                str(x)
                for x in np.round(
                    np.nanpercentile(arr, [0, 25, 50, 75, 100]), decimals
                ).astype(int)
            ]
        )
    else:
        output = ", ".join(
            [
                str(x)
                for x in np.round(np.nanpercentile(arr, [0, 25, 50, 75, 100]), decimals)
            ]
        )
    output += " (nan: {})".format(pd.isna(arr).sum())
    return output


def count_pct(vals, decimals=1):
    """Return count_nonzero/n (percent).

    Drops NaN values from the count.
    """
    vals = np.array(vals)
    nan_mask = pd.isna(vals)
    nan_count = np.count_nonzero(nan_mask)
    vals = vals[~nan_mask]
    numer = np.count_nonzero(vals > 0)
    denom = len(vals)
    if denom > 0:
        pct = numer / denom
    else:
        pct = 0
    string = "{}/{} ({:.{_}%})".format(
        numer,
        denom,
        pct,
        _=decimals,
    )
    if nan_count > 0:
        string += " (nan: {})".format(nan_count)
    return string


def mean_sem(vals, decimals=2):
    """Return mean ± standard error."""
    string = "{:.{_}f} ± {:.{_}f}".format(
        np.nanmean(vals), sem(vals, nan_policy="omit"), _=decimals
    )
    return string


def mean_sd(vals, decimals=2):
    """Return mean ± standard error."""
    string = "{:.{_}f} ± {:.{_}f}".format(np.nanmean(vals), np.nanstd(vals), _=decimals)
    return string


def gmean_sem(vals, decimals=2):
    """Return geometric mean ± standard error."""
    log_vals = np.log10(vals)
    _mean = 10 ** np.nanmean(log_vals)
    _sem = 10 ** sem(log_vals, nan_policy="omit")
    string = "{:.{_}f} ± {:.{_}f}".format(_mean, _sem, _=decimals)
    return string


def median_q(vals, decimals=2):
    """Return median (lower quartile, upper quartile)."""
    string = "{:.{_}f} ({:.{_}f}, {:.{_}f})".format(
        np.nanmedian(vals), *np.nanpercentile(vals, [25, 75]), _=decimals
    )
    return string


def circit(val, prop="r", scale=1):
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
    if prop == "r":
        r = val
        d = r * 2
        a = np.pi * np.square(r)
        c = 2 * np.pi * r
    elif prop == "d":
        d = val
        r = d / 2
        a = np.pi * np.square(r)
        c = 2 * np.pi * r
    elif prop == "a":
        a = val
        r = np.sqrt(a / np.pi)
        d = r * 2
        c = 2 * np.pi * r
    elif prop == "c":
        c = val
        r = c / (2 * np.pi)
        d = r * 2
        a = np.pi * np.square(r)

    # Store the outputs.
    circle = {"r": r, "d": d, "a": a, "c": c}

    return circle
