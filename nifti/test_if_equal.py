#!/Users/dschonhaut/mambaforge/envs/nipy310/bin/python

"""
$ test_if_equal.py img1.nii img2.nii [mask.nii]
"""

import sys
import os
import os.path as op
import numpy as np
import pandas as pd
import scipy.stats as stats

sys.path.append(op.join(op.expanduser("~"), "code"))
import general.nifti.nifti_ops as nops


def compare_imgs(img1_path, img2_path, mask=None):
    """Test two niftis for voxelwise equality.

    Note: Tests the data arrays for equality, not the image headers!

    Also prints voxelwise difference stats for nonzero voxels in both
    images. If a mask is included, only nonzero voxels within the mask
    are considered.

    Parameters
    ----------
    img1_path : str
        Path to a nifti image.
    img2_path : str
        Path to a nifti image with equal dimensions as img1.
    mask : str
        Path to a nifti image with equal dimensions as img1 and img2.
    """

    # Load the input files into numpy matrices
    dat1 = nops.load_nii_flex(img1_path, dat_only=True, flatten=True)
    dat2 = nops.load_nii_flex(img2_path, dat_only=True, flatten=True)
    assert dat1.shape == dat2.shape

    if mask:
        maskdat = nops.load_nii_flex(mask, dat_only=True, flatten=True, binarize=True)
        n_voxels = np.flatnonzero(maskdat).size
        nonzero_voxels = np.intersect1d(
            np.flatnonzero(maskdat),
            np.intersect1d(np.flatnonzero(dat1), np.flatnonzero(dat2)),
        )
    else:
        n_voxels = dat1.size
        nonzero_voxels = np.intersect1d(np.flatnonzero(dat1), np.flatnonzero(dat2))

    dat1 = dat1[nonzero_voxels]
    dat2 = dat2[nonzero_voxels]

    imgs_equal = False
    if np.allclose(dat1, dat2):
        imgs_equal = True
        return imgs_equal

    dat1_mean = dat1.mean()
    dat1_std = dat1.std()
    dat2_mean = dat2.mean()
    dat2_std = dat2.std()
    dat1_pcts = []
    dat2_pcts = []
    pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    for x in pcts:
        dat1_pcts.append(np.percentile(dat1, x))
        dat2_pcts.append(np.percentile(dat2, x))
    dat1_sub_dat2 = dat1 - dat2
    dat1_div_dat2 = dat1 / dat2
    dat1_sub_dat2_mean = dat1_sub_dat2.mean()
    dat1_sub_dat2_std = dat1_sub_dat2.std()
    dat1_div_dat2_mean = dat1_div_dat2.mean()
    dat1_div_dat2_std = dat1_div_dat2.std()
    dat1_sub_dat2_pcts = np.percentile(dat1_sub_dat2, pcts)
    dat1_div_dat2_pcts = np.percentile(dat1_div_dat2, pcts)
    num_nonzero = dat1.size
    pct_nonzero = num_nonzero / n_voxels
    _r = stats.pearsonr(dat1, dat2)[0]
    _rho = stats.spearmanr(dat1, dat2)[0]
    return [
        imgs_equal,
        dat1_mean,
        dat1_std,
        dat2_mean,
        dat2_std,
        pcts,
        dat1_pcts,
        dat2_pcts,
        dat1_sub_dat2_mean,
        dat1_sub_dat2_std,
        dat1_div_dat2_mean,
        dat1_div_dat2_std,
        dat1_sub_dat2_pcts,
        dat1_div_dat2_pcts,
        n_voxels,
        num_nonzero,
        pct_nonzero,
        _r,
        _rho,
    ]


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(
            __doc__,
            compare_imgs.__doc__,
            sep="\n",
        )
        exit()

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
    if isinstance(output, bool):
        imgs_equal = output
    else:
        (
            imgs_equal,
            dat1_mean,
            dat1_std,
            dat2_mean,
            dat2_std,
            pcts,
            dat1_pcts,
            dat2_pcts,
            dat1_sub_dat2_mean,
            dat1_sub_dat2_std,
            dat1_div_dat2_mean,
            dat1_div_dat2_std,
            dat1_sub_dat2_pcts,
            dat1_div_dat2_pcts,
            n_voxels,
            num_nonzero,
            pct_nonzero,
            _r,
            _rho,
        ) = output

    if imgs_equal:
        if mask:
            msg = "~ Images are voxelwise identical within the mask ~"
        else:
            msg = "~ Images are voxelwise identical ~"
        print(
            "",
            msg,
            "",
            sep="\n",
            end="\n" * 1,
        )
    else:
        if mask:
            msg = "~ Images are NOT equal within the mask ~"
        else:
            msg = "~ Images are NOT equal ~"
        print(
            "",
            msg,
            "",
            "img1 : {}".format(op.basename(img1_path)),
            "img2 : {}".format(op.basename(img2_path)),
            "",
            "Image Stats (img1*img2 nonzero voxels)",
            "-" * 52,
            "nonzero voxels       : {:,}/{:,} ({:.2%})".format(
                num_nonzero, n_voxels, pct_nonzero
            ),
            "img1 mean            : {:,.4f}".format(dat1_mean),
            "img1 std             : {:,.4f}".format(dat1_std),
            "img2 mean            : {:,.4f}".format(dat2_mean),
            "img2 std             : {:,.4f}".format(dat2_std),
            "img1 - img2 mean     : {:.4f}".format(dat1_sub_dat2_mean),
            "img1 - img2 std      : {:.4f}".format(dat1_sub_dat2_std),
            "img1 / img2 mean     : {:.4f}".format(dat1_div_dat2_mean),
            "img1 / img2 std      : {:.4f}".format(dat1_div_dat2_std),
            "img1 ~ img2 pearson  : {:.4f}".format(_r),
            "img1 ~ img2 spearman : {:.4f}".format(_rho),
            "",
            "Percentiles",
            "-" * 52,
            pd.DataFrame(
                [dat1_pcts, dat2_pcts, dat1_sub_dat2_pcts, dat1_div_dat2_pcts],
                columns=pcts,
                index=["img1", "img2", "img1-img2", "img1/img2"],
            )
            .round(4)
            .T,
            "",
            sep="\n",
            end="\n" * 1,
        )
