import os
import os.path as op
import gzip
import shutil
import warnings
from glob import glob
from inspect import isroutine
from collections import OrderedDict as od
import numpy as np
import pandas as pd
import nibabel as nib
import general.osops.os_utils as osu
import general.basic.str_methods as strm


def load_nii(
    infile,
    dtype=np.float32,
    squeeze=True,
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
    infile = find_gzip(infile)

    # Load the NIfTI image and data array.
    img = nib.load(infile)
    dat = np.asanyarray(img.dataobj)

    # Format the data array.
    dat = _format_array(
        dat,
        dtype=dtype,
        squeeze=squeeze,
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
        infile = find_gzip(obj)
        img, dat = load_nii(infile, **kws)
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
    squeeze=True,
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

    # Squeeze the array.
    elif squeeze:
        dat = np.squeeze(dat)

    # Convert NaNs.
    if not np.any((conv_nan is None, conv_nan is False, conv_nan is np.nan)):
        dat[np.invert(np.isfinite(dat))] = conv_nan

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


def save_nii(img, outfile, dat=None, overwrite=False, verbose=True):
    """Save a new NIfTI image to disc and return the saved filepath."""
    if op.exists(outfile) and not overwrite:
        raise FileExistsError(f"File {outfile} already exists")
    if dat is None:
        nib.save(img, outfile)
    else:
        newimg = nib.Nifti1Image(dat, affine=img.affine, header=img.header)
        newimg.to_filename(outfile)
    if verbose:
        print("Saved {}".format(outfile))
    return outfile


def dcm2niix(dcm_dir, remove_dicoms=False):
    """Run dcm2niix on dcm_dir and return the recon'd nifti path(s)."""
    cmd = "dcm2niix {}".format(dcm_dir)
    osu.run_cmd(cmd)
    niftis = glob(op.join(dcm_dir, "*.nii*"))
    if remove_dicoms:
        dicoms = glob(op.join(dcm_dir, "*.dcm"))
        for dcm in dicoms:
            os.remove(dcm)
    return niftis


def find_gzip(infile, raise_error=False, return_infile=False):
    """Find the existing file, gzipped or gunzipped.

    Return the infile if it exists, otherwise return the gzip-toggled
    version of the infile if it exists, otherwise return None or raise
    a FileNotFoundError.

    Parameters
    ----------
    infile : str
        The input file string.
    raise_error : bool
        If true, a FileNotFoundError is raised if the outfile does not
        exist.
    return_infile : bool
        If true, the infile is returned if the outfile does not exist.
        Otherwise None is returned if the outfile does not exist. This
        argument is ignored if raise_error is true.
    """
    if op.isfile(infile):
        outfile = infile
        return outfile
    elif op.isfile(toggle_gzip(infile)):
        outfile = toggle_gzip(infile)
        return outfile
    else:
        if raise_error:
            raise FileNotFoundError(
                "File not found: {}[.gz]".format(infile.replace(".gz", ""))
            )
        elif return_infile:
            return infile
        else:
            return None


def toggle_gzip(infile):
    """Return the gzip-toggled filepath.

    Parameters
    ----------
    infile : str
        The input file string.

    Returns
    -------
    outfile : str
        The output file string, which is the input file string minus
        the ".gz" extension if it exists in infile, or the input file
        string plus the ".gz" extension if it does not exist in infile.
    """
    if infile.endswith(".gz"):
        outfile = infile[:-3]
    else:
        outfile = infile + ".gz"
    return outfile


def gzip_nii(infile, rm_orig=True):
    """Gzip a *.nii file and return the output filename.

    Nothing happens if the infile does not exist or does not end with
    ".nii", in which case infile is returned.
    """
    if op.isfile(infile) and infile.endswith(".nii"):
        outfile = infile + ".gz"
        with open(infile, "rb") as f_in:
            with gzip.open(outfile, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        if rm_orig:
            os.remove(infile)
        return outfile
    else:
        return infile


def gunzip_nii(infile, rm_orig=True):
    """Gunzip a *.nii.gz file and return the output filename.

    Nothing happens if the infile does not exist or does not end with
    ".nii.gz", in which case infile is returned.
    """
    if op.isfile(infile) and infile.endswith(".nii.gz"):
        outfile = infile[:-3]
        with gzip.open(infile, "rb") as f_in:
            with open(outfile, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        if rm_orig:
            os.remove(infile)
        return outfile
    else:
        return infile


def gzip_or_gunzip_nii(infile, rm_orig=True):
    """Gzip *.nii or gunzip *.nii.gz file and return output filename.

    Nothing happens if the infile does not exist or does not end with
    ".nii" or ".nii.gz", in which case infile is returned.
    """
    if op.isfile(infile):
        if infile.endswith(".nii"):
            outfile = gzip_nii(infile, rm_orig)
            return outfile
        elif infile.endswith(".nii.gz"):
            outfile = gunzip_nii(infile, rm_orig)
            return outfile
    return infile


def recenter_niis(images, prefix=None, suffix=None, verbose=True):
    """Recenter 1+ nifti images in the center of the voxel grid.

    This process involves rewriting the image header and does not affect
    the underlying image data. Default behavior is to overwrite the
    infile unless a prefix or suffix is specified.

    A note of caution: If the infile plus a given prefix or suffix
    already exists, that file will be overwritten.

    Parameters
    ----------
    images : str or list
        Path to the image file or list of image files to recenter.
    prefix : str
        Prefix to use for the output file names.
    suffix : str
        Suffix to use for the output file names.

    Returns
    -------
    outfiles : list
        List of output file names.
    """
    if isinstance(images, str):
        images = [images]
    outfiles = []
    for image in images:
        *_, outfile = recenter_nii(
            image,
            prefix=prefix,
            suffix=suffix,
            save_output=True,
            overwrite=True,
            verbose=verbose,
        )
        outfiles.append(outfile)
    return outfiles


def recenter_nii(
    obj,
    outfile=None,
    prefix=None,
    suffix=None,
    save_output=True,
    overwrite=True,
    verbose=True,
    **kws,
):
    """Recenter nifti image to the center of the voxel grid as SPM."""
    # Load the input image.
    if isinstance(obj, str):
        infile = find_gzip(obj)
        img, dat = load_nii(infile)
    elif isinstance(obj, nib.Nifti1Pair):
        img, dat = load_nii_flex(obj, **kws)
        if np.all((outfile is None, prefix is None, suffix is None)):
            save_output = False
    else:
        raise ValueError("obj must be a NIfTI filepath or Nifti1Pair object")

    affine_new = img.affine.copy()
    for ii in range(3):
        affine_new[ii, 3] = ((img.shape[ii] - 1) * 0.5) * -affine_new[ii, ii]

    # Update the image with the new affine
    img.set_sform(affine_new)

    # Save the output image.
    if save_output:
        # Raise an error if outfile is specified along with prefix or suffix.
        if outfile is None:
            outfile = strm.add_presuf(infile, prefix, suffix)
        elif prefix is not None or suffix is not None:
            raise ValueError(
                "Either outfile or prefix/suffix can be defined, but not both"
            )
        if overwrite or not op.isfile(outfile):
            if verbose:
                print("Recentering {}".format(op.basename(outfile)))
            outfile = save_nii(
                img=img,
                outfile=outfile,
                dat=dat,
                overwrite=overwrite,
                verbose=verbose,
            )
    else:
        outfile = None

    return img, dat, outfile


def convert_values(
    infile,
    value_map,
    outfile=None,
    outfile_map=None,
    overwrite=False,
    verbose=True,
    **kws,
):
    """Convert values in a NIfTI image.

    Parameters
    ----------
    infile : str
        Path to the input image.
    value_map : dict
        Dictionary mapping old values to new values.
    outfile : str
        Path to the output image.
    outfile_map : dict
        Dictionary mapping new values to output filenames. Each new
        value is saved as a separate mask image.
    overwrite : bool
        Overwrite the output image file if it already exists.
    verbose : bool
        Print the output filepath if the output file is saved.
    **kws
        Keyword arguments passed to load_nii.

    Returns
    -------
    If outfile is provided, returns the output filepath.
    If outfile_map is provided, returns a list of output filepaths.
    """
    # Check that either outfile or outfile_map is provided but not both.
    if outfile is None and outfile_map is None:
        raise ValueError("Either outfile or outfile_map must be provided.")
    elif outfile is not None and outfile_map is not None:
        raise ValueError("Only one of outfile or outfile_map can be provided.")

    # Load the input nifti and create a zero-array with the same shape.
    img, indat = load_nii(infile, **kws)
    outdat = np.zeros_like(indat)

    # Loop over value_map items and convert old values in indat to new
    # values in outdat.
    for oldval, newval in value_map.items():
        outdat[indat == oldval] = newval

    # Save the output nifti.
    if outfile:
        outfile = save_nii(
            img=img, outfile=outfile, dat=outdat, overwrite=overwrite, verbose=verbose
        )
        return outfile
    # Save an output nifti file for each value in outfile_map.
    else:
        newfiles = []
        for newval, newfile in outfile_map.items():
            newdat = np.zeros_like(indat)
            newdat[outdat == newval] = 1
            newfile = save_nii(
                img=img,
                outfile=newfile,
                dat=newdat,
                overwrite=overwrite,
                verbose=verbose,
            )
            newfiles.append(newfile)
        return newfiles


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
        outfile = save_nii(
            img=suvr_img,
            outfile=outfile,
            dat=suvr_dat,
            overwrite=overwrite,
            verbose=verbose,
        )

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


def roi_desc(dat, rois, subrois=None, aggf=np.mean, conv_nan=0):
    """Apply `aggf` over `dat` values within each ROI mask.

    Parameters
    ----------
    dat :
        Filepath string, nifti image, or array-like object.
    rois : str, list[str], or dict-like {str: obj}
        Map each ROI name to its filepath string(s), nifti image, or
        array.
    subrois : dict of {str: int or list}
        Map each sub-ROI within the main ROI mask to a value or list of
        mask values that comprise it. The classic example is of an
        aparc+aseg file containing multiple regions with different
        labels. Note: subrois cannot be passed if len(rois) > 1.
    aggf : function, list of functions, or dict of functions
        Function or functions to apply over `dat` values within each
        ROI.
    conv_nan : bool, number, or NoneType object
        Convert NaNs in `dat` to `conv_nan`. No conversion is applied if
        `conv_nan` is np.nan, None, or False.

    Returns
    -------
    output : DataFrame
        `aggf` output for each agg function, for each ROI. Index is the
        ROI names, columns are the function names. The last column is
        ROI volume (number of voxels in the mask).
    """
    if (not isinstance(rois, str)) and (len(rois) > 1) and (subrois is not None):
        raise ValueError("Cannot define multiple rois and subrois")

    # Load the data array.
    dat = load_nii_flex(dat, dat_only=True, flatten=True, conv_nan=conv_nan)

    # Format the ROIs to be dict-like.
    if isinstance(rois, str):
        rois = [rois]

    if isinstance(rois, (list, tuple)):
        rois_dict = od([])
        for roi in rois:
            splits = strm.split(roi, ["_", "."])
            for ii, string in enumerate(splits):
                if string.startswith("mask-"):
                    roi_name = "-".join(string.split("-")[1:])
                    rois_dict[roi_name] = roi
                    break
                elif ii == len(splits) - 1:
                    rois_dict[".".join(op.basename(roi).split(".")[:-1])] = roi
        rois = rois_dict
    elif hasattr(rois, "keys"):
        pass
    else:
        raise ValueError("rois must be str, list, tuple, or dict-like")

    # Format the aggregation functions to be dict-like.
    if isroutine(aggf):
        aggf = od({aggf.__name__: aggf})
    elif not isinstance(aggf, dict):
        aggf = od({func.__name__: func for func in aggf})

    # Prepare the output DataFrame.
    if subrois is not None:
        output_idx = list(subrois.keys())
    else:
        output_idx = list(rois.keys())
    output_cols = list(aggf.keys()) + ["voxels"]
    output = pd.DataFrame(index=output_idx, columns=output_cols)
    output = output.rename_axis("roi")

    # Loop over the ROIs and sub-ROIs.
    for roi, roi_mask in rois.items():
        if subrois is not None:
            mask = load_nii_flex(roi_mask, dat_only=True, flatten=True, binarize=False)
            assert dat.shape == mask.shape
            for subroi, subroi_vals in subrois.items():
                mask_idx = np.where(np.isin(mask, subroi_vals))
                for func_name, func in aggf.items():
                    output.at[subroi, func_name] = func(dat[mask_idx])
                output.at[subroi, "voxels"] = mask_idx[0].size
        else:
            mask = load_nii_flex(roi_mask, dat_only=True, flatten=True, binarize=True)
            assert dat.shape == mask.shape
            mask_idx = np.where(mask)
            for func_name, func in aggf.items():
                output.at[roi, func_name] = func(dat[mask_idx])
            output.at[roi, "voxels"] = mask_idx[0].size

    return output


def calc_3d_smooth(res_in, res_target, squeeze=False, verbose=False):
    """Return FWHM of the Gaussian that smooths initial to target resolution.

    Parameters
    ----------
    res_in : float or array-like
        Starting resolution in mm.
    res_target : float or array-like
        Target resolution in mm.
    squeeze : bool
        If True, squeeze the output to single float if possible.
    verbose : bool
        Whether to print the FWHM to standard output.

    Returns
    -------
    fwhm : 3-length list or float
        Amount to smooth by in mm in each dimension.
    """
    if isinstance(res_in, (int, float)):
        res_in = [res_in, res_in, res_in]
    elif len(res_in) == 1:
        res_in = [res_in[0], res_in[0], res_in[0]]
    if isinstance(res_target, (int, float)):
        res_target = [res_target, res_target, res_target]
    elif len(res_target) == 1:
        res_target = [res_target[0], res_target[0], res_target[0]]
    res_in = np.asanyarray(res_in)
    res_target = np.asanyarray(res_target)
    assert res_in.size == res_target.size == 3
    assert res_in.min() > 0 and res_target.min() > 0
    fwhm = np.sqrt((res_target**2) - (res_in**2)).tolist()
    if squeeze:
        if fwhm[0] == fwhm[1] == fwhm[2]:
            fwhm = fwhm[0]
    if verbose:
        print("FWHM = {}".format(fwhm))
    return fwhm
