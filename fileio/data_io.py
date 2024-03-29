"""
Functions for data input and output.
"""
import pickle


def save_pickle(obj, fpath, verbose=True):
    """Save object as a pickle file."""
    with open(fpath, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    if verbose:
        print("Saved {}".format(fpath))


def open_pickle(fpath):
    """Return object."""
    with open(fpath, "rb") as f:
        obj = pickle.load(f)
    return obj


def read_excel(file: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(file, sheet_name=sheet)
