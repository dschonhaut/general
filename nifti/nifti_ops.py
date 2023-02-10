import os.path as op
import warnings
from inspect import isfunction
from collections import OrderedDict as od
import numpy as np
import pandas as pd
import nibabel as nib


def load_nii(
    infile,
    dtype=np.float32,
    flatten=False,
    conv_nan=0,
    binarize=False,
    int_rounding="nearest",
):
    """Load a NIfTI file and return the NIfTI image and data array.

    Returns (img, dat), with dat being an instance of img.dataobj loaded
    from disk. You can modify or delete dat and get a new version from
    disk: ```dat = np.asanyarray(img.dataobj)```

    Parameters
    ----------
    infile : str
        The nifti file to load.
    dtype : data-type
        Determines the data type of the data array returned.
    flatten : bool
        If true, `dat` is returned as a flattened copy of the
        `img`.dataobj array. Otherwise `dat`.shape == `img`.shape.
    conv_nan : bool, number, or NoneType object
        Convert NaNs to `conv_nan`. No conversion is applied if
        `conv_nan` is np.nan, None, or False.
    binarize : bool
        If true, `dat` values > 0 become 1 and all other values are 0.
        `dat` type is recast to np.uint8.
    int_rounding : str
        Determines how the data array is recast if `binarize` is false
        and `dtype` is an integer.
        `nearest` : round to the nearest integer
        `floor` : round down
        `ceil` : round up

    Returns
    -------
    img : Nifti1Image
    dat : ndarray or ndarray subclass
    """
    # Get the right file extension.
    if not op.exists(infile):
        infile = toggle_gzip(infile)

    # Load the NIfTI image and data array.
    img = nib.load(infile)
    dat = np.asanyarray(img.dataobj)

    # Format the data array.
    dat = _format_array(
        dat,
        dtype=dtype,
        flatten=flatten,
        conv_nan=conv_nan,
        binarize=binarize,
        int_rounding=int_rounding,
    )

    return img, dat


def load_nii_flex(obj, dat_only=False, **kws):
    """Load Nifti using flexible input formatting and variable outputs.

    Parameters
    ----------
    obj
        The Nifti object. Acceptable inputs include a filepath string,
        Nifti image, ndarray, or object that can be cast as an ndarray.
    dat_only : bool
        If true only the data array is returned; otherwise function
        returns the (img, dat) nifti pair.
    **kws are passed to _format_array()

    Returns
    -------
    [img] : Nifti1Image
        Returned only if `dat_only` is false.
    dat : ndarray or ndarray subclass
    """
    if isinstance(obj, str):
        img, dat = load_nii(obj, **kws)
        if dat_only:
            return dat
        else:
            return img, dat
    elif isinstance(obj, nib.Nifti1Pair):
        img = obj
        dat = np.asanyarray(img.dataobj)
        dat = _format_array(dat, **kws)
        if dat_only:
            return dat
        else:
            return img, dat
    elif isinstance(obj, np.ndarray):
        dat = _format_array(obj, **kws)
        if dat_only:
            return dat
        else:
            msg = "\nCannot return the (img, dat) pair due to missing header info."
            raise RuntimeError(msg)
    else:
        dat = _format_array(np.asanyarray(obj), **kws)
        if dat_only:
            return dat
        else:
            msg = "\nCannot return the (img, dat) pair due to missing header info."
            raise RuntimeError(msg)


def _format_array(
    dat,
    dtype=np.float32,
    flatten=False,
    conv_nan=0,
    binarize=False,
    int_rounding="nearest",
):
    """Format an array.

    Formatting options:
    - Flattening
    - NaN handling
    - Data type conversion

    Parameters
    ----------
    dtype : data-type
        Determines the data type returned.
    flatten : bool
        Return `dat` as a flattened copy of the input array.
    conv_nan : bool, number, or NoneType object
        Convert NaNs to `conv_nan`. No conversion is applied if
        `conv_nan` is np.nan, None, or False.
    binarize : bool
        If true, `dat` values > 0 become 1 and all other values are 0.
        `dat` type is recast to np.uint8.
    int_rounding : str
        Determines how the data array is recast if `binarize` is false
        and `dtype` is an integer.
        `nearest` : round to the nearest integer
        `floor` : round down
        `ceil` : round up

    Returns
    -------
    dat : ndarray or ndarray subclass
    """
    # Flatten the array.
    if flatten:
        dat = dat.ravel()

    # Convert NaNs.
    if not np.any((conv_nan is None, conv_nan is False, conv_nan is np.nan)):
        dat[np.isnan(dat)] = conv_nan

    # Recast the data type.
    if binarize or (dtype is bool):
        idx = dat > 0
        dat[idx] = 1
        dat[~idx] = 0
        if dtype is bool:
            dat = dat.astype(bool)
        else:
            dat = dat.astype(np.uint8)
    elif "int" in str(dtype):
        if int_rounding == "nearest":
            dat = np.rint(dat)
        elif int_rounding == "floor":
            dat = np.floor(dat)
        elif int_rounding == "ceil":
            dat = np.ceil(dat)
        else:
            raise ValueError("int_rounding='{}' not valid".format(int_rounding))
        dat = dat.astype(dtype)
    else:
        dat = dat.astype(dtype)

    return dat


def save_nii(img, dat, outfile, overwrite=False, verbose=True):
    """Save a new NIfTI image to disc and return the saved filepath."""
    if overwrite or not op.exists(outfile):
        newimg = nib.Nifti1Image(dat, affine=img.affine, header=img.header)
        newimg.to_filename(outfile)
        if verbose:
            print("Saved {}".format(outfile))
        return outfile
    else:
        return None


def toggle_gzip(infile):
    """Toggle file string gzipping."""
    if infile.endswith(".gz"):
        return infile[:-3]
    else:
        return infile + ".gz"


def create_suvr(
    pet, ref_region, dat_only=False, outfile=None, overwrite=False, verbose=False
):
    """Return the voxelwise SUVR data array and optionally save to disc.

    pet and ref_region can each be passed as a filepath string to a
    NIfTI image, a Nifti1Image or Nifti2Image, or an ndarray. If both
    are passed as ndarrays, the output SUVR cannot be saved as header
    info is unknown, and a warning will be raised to this effect.

    Parameters
    ----------
    pet : string, nifti image, or array-like
        The voxelwise PET image.
    ref_region : string, nifti image, or array-like
        The reference region. Must have the same dimensions as pet.
        Values > 0 will be used as the reference region mask.
    dat_only : bool
        If true only the data array is returned; otherwise function
        returns the (img, dat) nifti pair.
    outfile : string or None
        Filepath to the SUVR image that will be saved. If None, the SUVR
        array is returned but nothing is saved to disk.
    overwrite : bool
        If True and outfile exists, it will be overwritten. If False,
        outfile will not be overwritten.
    verbose : bool
        Whether to print the mean ref region value and saved file to
        standard output.

    Returns
    -------
    [suvr_img] : Nifti1Image
        Returned only if `dat_only` is false.
    suvr_dat : ndarray or ndarray subclass
    """
    # Load the PET image.
    if isinstance(pet, (str, nib.Nifti1Pair)):
        pet_img, pet_dat = load_nii_flex(pet)
    else:
        pet_dat = load_nii_flex(pet, dat_only=True)

    # Load the ref region.
    if isinstance(ref_region, (str, nib.Nifti1Pair)):
        rr_img, rr_dat = load_nii_flex(ref_region, binarize=True)
    else:
        rr_dat = load_nii_flex(ref_region, dat_only=True, binarize=True)

    assert pet_dat.shape == rr_dat.shape

    # Get ref region voxel coords.
    rr_idx = np.where(rr_dat)

    # Make the SUVR.
    rr_mean = np.mean(pet_dat[rr_idx])
    if verbose:
        print("Ref. region mean = {:.2f}".format(rr_mean))
    suvr_dat = pet_dat / rr_mean

    # Save the SUVR.
    if outfile and np.any((overwrite, not op.exists(outfile))):
        if "pet_img" not in locals():
            if "rr_img" not in locals():
                msg = (
                    "\nCannot save SUVR due to missing header info."
                    "\nMust import `pet` or `ref_region` as a filepath or NIfTI image."
                )
                warnings.warn(msg)
            else:
                suvr_img = rr_img
        else:
            suvr_img = pet_img
        outfile = save_nii(suvr_img, suvr_dat, outfile, overwrite, verbose)

    if dat_only:
        return suvr_dat
    else:
        if "pet_img" not in locals():
            if "rr_img" not in locals():
                msg = "\nCannot return the (img, dat) pair due to missing header info."
                raise RuntimeError(msg)
            else:
                suvr_img = rr_img
        else:
            suvr_img = pet_img
        return suvr_img, suvr_dat


def roi_desc(dat, rois, aggf=np.mean, conv_nan=0):
    """Apply `aggf` over `dat` values within each ROI mask.

    Parameters
    ----------
    dat :
        Filepath string, nifti image, or array-like object.
    rois : dict, {str: obj}
        Map each ROI name to its filepath string, nifti image, or array.
    aggf : function, list of functions, or dict of functions
        Function or functions to apply over `dat` values within each
        ROI.
    conv_nan : bool, number, or NoneType object
        Convert NaNs in `dat` to `conv_nan`. No conversion is applied if
        `conv_nan` is np.nan, None, or False.

    Returns
    -------
    grouped : Series or DataFrame
        `aggf` output for each agg function, for each ROI. Index is the
        ROI names, columns are the function names.
    """
    dat = load_nii_flex(dat, dat_only=True, conv_nan=conv_nan)

    if isfunction(aggf):
        grouped = pd.Series(dtype=np.float32, name=aggf.__name__)
        for roi, roi_mask in rois.items():
            mask = load_nii_flex(roi_mask, dat_only=True, binarize=True)
            assert dat.shape == mask.shape
            mask_idx = np.where(mask)
            grouped[roi] = aggf(dat[mask_idx])
    else:
        if not isinstance(aggf, dict):
            aggf = od({func.__name__: func for func in aggf})
        grouped = pd.DataFrame(index=list(rois.keys()), columns=list(aggf.keys()))
        for roi, roi_mask in rois.items():
            mask = load_nii_flex(roi_mask, dat_only=True, binarize=True)
            assert dat.shape == mask.shape
            mask_idx = np.where(mask)
            for func_name, func in aggf.items():
                grouped.at[roi, func_name] = func(dat[mask_idx])

    return grouped
