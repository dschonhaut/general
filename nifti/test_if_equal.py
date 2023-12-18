#!/usr/bin/env python

"""
$ test_if_equal.py img1.nii img2.nii [mask.nii]
"""

import sys
import os
import os.path as op
import numpy as np
import pandas as pd
import scipy.stats as stats
import transforms3d.affines as affines
import general.nifti.nifti_ops as nops


def compare_imgs(img1_path, img2_path, mask=None):
    """Test two niftis for affine transform and voxelwise equality.

    Parameters
    ----------
    img1_path : str
        Path to a nifti image.
    img2_path : str
        Path to a nifti image with equal dimensions as img1.
    mask : str
        Path to a nifti image with equal dimensions as img1 and img2.
    """
    # Load the input niftis
    img1, dat1 = nops.load_nii_flex(img1_path, flatten=True)
    img2, dat2 = nops.load_nii_flex(img2_path, flatten=True)
    assert dat1.shape == dat2.shape

    output = {}

    # Test equality of the affine transforms
    if np.allclose(img1.affine, img2.affine, equal_nan=True):
        output["affines_equal"] = True
    else:
        output["affines_equal"] = False
        output["affine1"] = img1.affine
        output["affine2"] = img2.affine
        output["affine1_sub_affine2"] = img1.affine - img2.affine

    # Identify nonzero voxels across both niftis and mask, if applicable
    if mask:
        maskdat = nops.load_nii_flex(mask, dat_only=True, flatten=True, binarize=True)
        output["n_voxels"] = np.flatnonzero(maskdat).size
        nonzero_voxels = np.intersect1d(
            np.flatnonzero(maskdat),
            np.intersect1d(np.flatnonzero(dat1), np.flatnonzero(dat2)),
        )
    else:
        output["n_voxels"] = dat1.size
        nonzero_voxels = np.intersect1d(np.flatnonzero(dat1), np.flatnonzero(dat2))

    # Compute voxel stats only on nonzero voxels
    dat1 = dat1[nonzero_voxels]
    dat2 = dat2[nonzero_voxels]

    # Test equality of the data arrays
    if np.allclose(dat1, dat2, equal_nan=True):
        output["arrays_equal"] = True
    else:
        output["arrays_equal"] = False

    # If affine and arrays are equal, return that the images are equal
    if output["affines_equal"] and output["arrays_equal"]:
        return output

    # Compute data array difference stats
    output["dat1_mean"] = dat1.mean()
    output["dat1_std"] = dat1.std()
    output["dat2_mean"] = dat2.mean()
    output["dat2_std"] = dat2.std()
    output["dat1_pcts"] = []
    output["dat2_pcts"] = []
    output["pcts"] = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    for x in output["pcts"]:
        output["dat1_pcts"].append(np.percentile(dat1, x))
        output["dat2_pcts"].append(np.percentile(dat2, x))
    dat1_sub_dat2 = dat1 - dat2
    dat1_sub_dat2_abs = np.abs(dat1_sub_dat2)
    dat1_sub_dat2_rmse = np.sqrt(np.mean(dat1_sub_dat2**2))
    output["dat1_sub_dat2_mean"] = dat1_sub_dat2.mean()
    output["dat1_sub_dat2_std"] = dat1_sub_dat2.std()
    output["dat1_sub_dat2_abs_mean"] = dat1_sub_dat2_abs.mean()
    output["dat1_sub_dat2_abs_std"] = dat1_sub_dat2_abs.std()
    output["dat1_sub_dat2_rmse"] = dat1_sub_dat2_rmse
    output["dat1_sub_dat2_pcts"] = np.percentile(dat1_sub_dat2, output["pcts"])
    output["dat1_sub_dat2_abs_pcts"] = np.percentile(dat1_sub_dat2_abs, output["pcts"])
    output["num_nonzero"] = dat1.size
    output["pct_nonzero"] = output["num_nonzero"] / output["n_voxels"]
    output["_r"] = stats.pearsonr(dat1, dat2)[0]

    return output


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(
            __doc__,
            compare_imgs.__doc__,
            sep="\n",
        )
        sys.exit(0)

    cwd = os.getcwd()

    # Get the images to test from command-line arguments
    img1_path = op.join(cwd, sys.argv[1])
    if not op.exists(img1_path):
        img1_path = sys.argv[1]

    img2_path = op.join(cwd, sys.argv[2])
    if not op.exists(img2_path):
        img2_path = sys.argv[2]

    if not op.exists(img1_path):
        print("\nCould not find {}\n".format(img1_path))
        exit()
    if not op.exists(img2_path):
        print("\nCould not find {}\n".format(img2_path))
        exit()

    if len(sys.argv) == 4:
        mask = op.join(cwd, sys.argv[3])
        if not op.exists(mask):
            mask = sys.argv[3]
        if not op.exists(mask):
            print("\nCould not find {}\n".format(mask))
            exit()
    else:
        mask = None

    # Figure out if the images are equal and print the result
    output = compare_imgs(img1_path, img2_path, mask)
    sep_len = 47

    # Print output
    if output["affines_equal"] and output["arrays_equal"]:
        print("\nImages are equal\n")
        sys.exit(0)
    print(
        "",
        "img1 : {}".format(op.basename(img1_path)),
        "img2 : {}".format(op.basename(img2_path)),
        "",
        "=" * sep_len,
        "",
        sep="\n",
    )
    if output["affines_equal"]:
        msg = "Image affines are equal"
        print(msg)
    else:
        msg = "Image affines are NOT equal"
        print(msg + "\n")
        aff = {
            "translation": [],
            "rotation": [],
            "zoom": [],
            "shear": [],
        }
        T, R, Z, S = affines.decompose(output["affine1"])
        aff["translation"].append(T)
        aff["rotation"].append(R)
        aff["zoom"].append(Z)
        aff["shear"].append(S)
        T, R, Z, S = affines.decompose(output["affine2"])
        aff["translation"].append(T)
        aff["rotation"].append(R)
        aff["zoom"].append(Z)
        aff["shear"].append(S)
        pad = " " * 10
        for k, v in aff.items():
            if np.allclose(v[0], v[1], atol=1e-7, rtol=0):
                if (k == "shear") and np.allclose(v, 0, atol=1e-7, rtol=0):
                    print("* {}: Equal (no shearing)".format(k.capitalize()))
                else:
                    print("* {}: Equal".format(k.capitalize()))
            else:
                print(
                    "* {}: NOT equal".format(k.capitalize()),
                    "    img1: {}".format(
                        np.array2string(v[0], precision=2, suppress_small=True).replace(
                            "\n", "\n" + pad
                        )
                    ),
                    "    img2: {}".format(
                        np.array2string(v[1], precision=2, suppress_small=True).replace(
                            "\n", "\n" + pad
                        )
                    ),
                    sep="\n",
                )
    print("", "=" * sep_len, "", sep="\n")

    if output["arrays_equal"]:
        if mask:
            msg = "Data arrays are equal within the mask"
        else:
            msg = "Data arrays are equal"
        print(msg, "", sep="\n")
    else:
        if mask:
            msg = "Data arrays are NOT equal within the mask"
        else:
            msg = "Data arrays are NOT equal"
        print(
            msg,
            "",
            "Image Stats (img1*img2 nonzero voxels)",
            "* nonzero voxels   : {:,}/{:,} ({:.2%})".format(
                output["num_nonzero"], output["n_voxels"], output["pct_nonzero"]
            ),
            "* img1             : {:,.8f} ± {:,.8f} (M ± SD)".format(
                output["dat1_mean"], output["dat1_std"]
            ),
            "* img2             : {:,.8f} ± {:,.8f}".format(
                output["dat2_mean"], output["dat2_std"]
            ),
            "* img1 - img2      : {:.8f} ± {:,.8f}".format(
                output["dat1_sub_dat2_mean"], output["dat1_sub_dat2_std"]
            ),
            "* |img1 - img2|    : {:.8f} ± {:,.8f}".format(
                output["dat1_sub_dat2_abs_mean"], output["dat1_sub_dat2_abs_std"]
            ),
            "* img1 - img2 RMSE : {:.8f}".format(output["dat1_sub_dat2_rmse"]),
            "* img1 ~ img2      : r = {:.8f}".format(output["_r"]),
            "",
            "Percentiles",
            "-" * 30,
            pd.DataFrame(
                [
                    output["dat1_pcts"],
                    output["dat2_pcts"],
                    output["dat1_sub_dat2_pcts"],
                    output["dat1_sub_dat2_abs_pcts"],
                ],
                columns=output["pcts"],
                index=["img1", "img2", "img1-img2", "|img1-img2|"],
            )
            .round(8)
            .T,
            "",
            sep="\n",
        )
    sys.exit(0)
