#!/Users/dschonhaut/mambaforge/bin/python

"""
common_funcs.py

Last Edited: 6/9/15

HWNI Location: /home/jagust/UCSF/Daniel/scripts

Dependencies: Python 2.7, numpy, nibabel

Description: A group of simple, common functions
that are frequently used by other scripts

Author: Daniel Schonhaut
Department of Neuorology
UCSF
"""
import os
import sys
import glob
import numpy as np
import nibabel as nb
import time


def get_lbl_id(string):
    """Find the LBL-ID in a string and return it as a substring.

    If no LBL-ID is found then return None.
    """
    lbl_id = ""
    nums = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    try:
        for n in range(len(string)):
            if string[n] == "B":
                if (
                    string[n + 1] in nums
                    and string[n + 2] in nums
                    and string[n + 3] in ("-", "_")
                    and string[n + 4] in nums
                    and string[n + 5] in nums
                    and string[n + 6] in nums
                ):
                    lbl_id = string[n : n + 7]
                    return lbl_id
        return None
    except:
        return None


def curtime():
    return time.strftime("%m-%d-%Y-%H-%M-%S")


def get_newest_mri(lbl_id):
    """For a given lbl_id, return the path to the most recent UCSF freesurfer-processed MRI

        Also return whether or not the most recent scan was processed (i.e. if MRI dir exists and
        freesurfer equivalent does not, the scan has probably been downloaded but not processed.

    If no MRI is found then return None.
    """
    mri_dir = "/home/jagust/UCSF/MRIs"
    fs_dir = "/home/jagust/UCSF/freesurfer_5_1"
    newest_dir = ""
    n = 1
    while True:
        if n == 1:
            if os.path.exists("{}/{}".format(mri_dir, lbl_id)):
                newest_dir = "{}/{}".format(fs_dir, lbl_id)
                if os.path.exists("{}/{}".format(fs_dir, lbl_id)):
                    processed = True
                else:
                    processed = False
                n += 1
            else:
                return None
        else:
            if os.path.exists("{}/{}".format(mri_dir + "-v" + str(n), lbl_id)):
                newest_dir = "{}/{}".format(fs_dir + "-v" + str(n), lbl_id)
                if os.path.exists("{}/{}".format(fs_dir, lbl_id)):
                    processed = True
                else:
                    processed = False
                n += 1
            else:
                return newest_dir, processed


def toggle_gzip(_path):
    """If path ends in .nii, return path ending in .nii.gz. And vice versa."""
    if _path.endswith(".nii"):
        return _path + ".gz"
    elif _path.endswith(".nii.gz"):
        return _path[: len(_path) - 3]
    else:
        return ""


def gzip_all(_path):
    """Recursively search path and gzip all .nii files. Return the number gzipped."""
    gunzipped = []
    for root, dirs, files in os.walk(_path):
        for f in files:
            if f[-4:] == ".nii":
                gunzipped.append(os.path.join(root, f))
    for f in gunzipped:
        cmd = "gzip {}".format(f)
        os.system(cmd)
    return len(gunzipped)


def crop_mri(img_path, dims):
    """Zero out all data points outside of the start/stop dims and save a new nifti image.

    dims is a variable like [x_start, x_stop, y_start, y_stop, z_start, z_stop]
    """
    orig_filename = os.path.split(img_path)[1]
    new_filename = (
        orig_filename[: orig_filename.find(".")]
        + "_cropped_{}-{}-{}-{}-{}-{}".format(
            dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]
        )
        + orig_filename[orig_filename.find(".") :]
    )
    img = nb.load(img_path)
    dat = img.get_data()
    newdat = np.zeros(dat.shape)
    for x in range(dat.shape[0]):
        for y in range(dat.shape[1]):
            for z in range(dat.shape[2]):
                if (
                    x < dims[0]
                    or x > dims[1]
                    or y < dims[2]
                    or y > dims[3]
                    or z < dims[4]
                    or z > dims[5]
                ):
                    newdat[x, y, z] = 0
                else:
                    newdat[x, y, z] = dat[x, y, z]
    newimg = nb.Nifti1Image(newdat, img.get_affine())
    newfile = os.path.join(os.path.split(img_path)[0], new_filename)
    newimg.to_filename(newfile)
    return newfile


def make_halo(sroi_path, roi_path, thresh, output_path):
    """Binarize sroi > 0.01, zero out sroi values where roi=1, and save as output_file.

    Arguments:
    sroi_path -- a smoothed or dilated roi (full file path string)
    roi_path -- the pre-smoothed version of sroi (full file path string)
    thresh -- threshold to binarize sroi above
    output_path -- name of the output file (full file path string)
    """
    sroi = nb.load(sroi_path)
    roi = nb.load(roi_path)
    sroi_dat = np.nan_to_num(sroi.get_data().squeeze())
    roi_dat = np.nan_to_num(roi.get_data().squeeze())
    roi_dat[roi_dat > 0] = 1
    output_dat = np.zeros(sroi.shape)
    output_dat[(sroi_dat > thresh) & (roi_dat != 1)] = 1
    output = nb.Nifti1Image(output_dat, sroi.get_affine())
    output.to_filename(output_path)
    return output_path


def get_roi_values(img_path, roi_path):
    """Get and return all values of img inside roi."""
    img = nb.load(img_path)
    roi = nb.load(roi_path)
    img_dat = np.nan_to_num(img.get_data().squeeze())
    roi_dat = np.nan_to_num(roi.get_data().squeeze())
    roi_dat[roi_dat > 0] = 1
    return img_dat[roi_dat == 1].tolist()


def combine_rois(output_path, *rois):
    """Combine one or more rois into a single file, with roi1=1, roi2=2, etc.

    Precedence is given in order of rois (so initial ROIs are not overwritten).
    """
    output_dat = np.zeros(nb.load(rois[0]).shape)
    for n in range(len(rois)):
        roi_dat = np.nan_to_num(nb.load(rois[n]).get_data().squeeze())
        roi_dat[roi_dat > 0] = 1
        output_dat[(roi_dat == 1) & (output_dat == 0)] = n + 1
    output = nb.Nifti1Image(output_dat, nb.load(rois[0]).get_affine())
    output.to_filename(output_path)


def combine_rois_keepvals(output_path, *rois):
    """Combine one or more rois into a single file, preserving intial ROI index values.

    Works with roi files that have multiple index values (like aparc+aseg).

    Precedence is given in order of rois (so initial ROIs are not overwritten).
    """
    if len(rois) == 1:
        rois = rois[0]
    shp = nb.load(rois[0]).shape
    output_dat = np.zeros(shp)
    for n in range(len(rois)):
        roi_dat = np.nan_to_num(nb.load(rois[n]).get_data().squeeze())
        for x in range(shp[0]):
            for y in range(shp[1]):
                for z in range(shp[2]):
                    if (roi_dat[x, y, z] > 0) and (output_dat[x, y, z] == 0):
                        output_dat[x, y, z] = roi_dat[x, y, z]
    output = nb.Nifti1Image(output_dat, nb.load(rois[0]).get_affine())
    output.to_filename(output_path)


def segment_halo_rois(output_path, roi, seg_files):
    """Segment roi, adding 0.1 if the voxel is GM, 0.2 if WM, and 0.3 if CSF.

    Arguments:
    output_path -- path to the image that gets created (string)
    roi -- path to the ROI image that will get segmented
    seg_files -- list of file paths like [c1, c2, c3]
    """
    shp = nb.load(roi).shape
    roi_dat = np.nan_to_num(nb.load(roi).get_data().squeeze())
    c1_dat = np.nan_to_num(nb.load(seg_files[0]).get_data().squeeze())
    c2_dat = np.nan_to_num(nb.load(seg_files[1]).get_data().squeeze())
    c3_dat = np.nan_to_num(nb.load(seg_files[2]).get_data().squeeze())
    for x in range(shp[0]):
        for y in range(shp[1]):
            for z in range(shp[2]):
                if roi_dat[x, y, z] > 1:
                    if (c1_dat[x, y, z] > c2_dat[x, y, z]) and (
                        c1_dat[x, y, z] > c3_dat[x, y, z]
                    ):
                        roi_dat[x, y, z] += 0.1
                    elif (c2_dat[x, y, z] > c1_dat[x, y, z]) and (
                        c2_dat[x, y, z] > c3_dat[x, y, z]
                    ):
                        roi_dat[x, y, z] += 0.2
                    else:
                        roi_dat[x, y, z] += 0.3
    output = nb.Nifti1Image(roi_dat, nb.load(roi).get_affine())
    output.to_filename(output_path)


def add_seg_masking(output_path, seg_files, seg_indices, aparc=None):
    """Add segmentation likelihood to an existing parcellation.

    Arguments:
    output_path -- path to the image that gets created (string)
    seg_files -- list of file paths like [c1, c2, c3]
    seg_indices -- list of the indices that should get written (e.g. [1, 2, 3])
    aparc -- path to the input roi or aparc/seg file. if aparc is None, makes
             a new image from a zero matrix
    """
    shp = nb.load(seg_files[0]).shape
    if aparc:
        roi_dat = np.nan_to_num(nb.load(aparc).get_data().squeeze())
    else:
        roi_dat = np.zeros(shp)
    c1_dat = np.nan_to_num(nb.load(seg_files[0]).get_data().squeeze())
    c2_dat = np.nan_to_num(nb.load(seg_files[1]).get_data().squeeze())
    c3_dat = np.nan_to_num(nb.load(seg_files[2]).get_data().squeeze())
    for x in range(shp[0]):
        for y in range(shp[1]):
            for z in range(shp[2]):
                if (roi_dat[x, y, z] == 0) and (
                    (c1_dat[x, y, z] > 0)
                    or (c2_dat[x, y, z] > 0)
                    or (c3_dat[x, y, z] > 0)
                ):
                    if (c1_dat[x, y, z] > c2_dat[x, y, z]) and (
                        c1_dat[x, y, z] > c3_dat[x, y, z]
                    ):
                        roi_dat[x, y, z] = seg_indices[0]
                    elif (c2_dat[x, y, z] > c1_dat[x, y, z]) and (
                        c2_dat[x, y, z] > c3_dat[x, y, z]
                    ):
                        roi_dat[x, y, z] = seg_indices[1]
                    else:
                        roi_dat[x, y, z] = seg_indices[2]
    output = nb.Nifti1Image(roi_dat, nb.load(seg_files[0]).get_affine())
    output.to_filename(output_path)


def swap_seg_values(seg_file, output_path, in_values, out_values):
    """Swap a list of values in an image for another set of values.

    Arguments:
    seg_file -- path to the input aparc/seg image
    output_path -- path to the image that gets created
    in_values -- a list of values presumably in the input seg_file
    out_values -- a list of values that get swapped in place of in_values, in order

    Example:
    If in_values = [1, 2, 3] and out_values = [10, 9, 7], then in the new image,
    1 becomes 10, 2 becomes 9, and 3 becomes 7.
    """
    output_dat = np.nan_to_num(nb.load(seg_file).get_data().squeeze())
    for n in range(len(in_values)):
        in_val = in_values[n]
        out_val = out_values[n]
        output_dat[output_dat == in_val] = out_val
    output = nb.Nifti1Image(output_dat, nb.load(seg_file).get_affine())
    output.to_filename(output_path)


def findnth(haystack, needle, n):
    parts = haystack.split(needle, n + 1)
    if len(parts) <= n + 1:
        return None
    return len(haystack) - len(parts[-1]) - len(needle)


# def recenter(img_path, newfile=''):
#    """Recenter img and add an r prefix to the filename."""
#    img = CMTransform(img_path).fix(newfile)


def get_motion_stats(motionf, flag_bad=False):
    """Given an SPM realignment file with 6 motion parameters, return a list of summary stats on motion.

    Arguments:
    motionf -- a .txt file written by SPM with the 6 parameters of movement calculated by realignment
               (note FSL realignment also gives these params, but in a different order. Haven't edited
                the code to work with both SPM and FSL, but you could manually format an FSL file like SPM
                and this would work fine.)
    flag_bad -- if True, the last variable returned will indicate whether the scan has too much motion or not.
                designed to test fmri scan quality, using logic adopted from the Seeley lab and explained below.

    Returns:
    max_trans_1d -- max mm moved between any 2 volumes along any 1 axis of translation
    max_rot_1d -- max degrees rotated between any 2 volumes along any 1 axis of rotation
    max_trans -- max mm moved between any 2 volumes along all 3 axes of translation
    mean_trans -- mean mm moved along all 3 axes of translation
    max_disp -- max displacement between any 2 volumes along all 6 axes of motion
    mean_disp -- mean displacement along all 6 axes of motion
    spikes -- number of spikes across volumes (1 spike = displacement > 1mm between any 2 volumes)
    ratio_spikes -- ratio of spikes to total volumes
    bad -- if select_bad=True, bad=True if max_trans_1d > 3 OR max_rot_1d > 3 OR ratio_spikes > 0.1

    """
    # parse the motion text file
    motion_params = np.loadtxt(motionf)

    # determine change in motion from one volume to the next
    motion_diff = np.diff(motion_params, axis=0)

    # calculate max movement in any one direction of movement
    # and along any one axis of rotation
    max_trans_1d = abs(motion_diff[:, :3]).max()
    max_rot_1d = abs(motion_diff[:, 3:6]).max() * (180 / np.pi)

    # calculate combined translation along x, y, z axes as
    # sqrt(x**2 + y**2 + z**2)
    trans = np.sqrt(np.sum(np.power(motion_diff[:, :3], 2), 1))
    max_trans_3d = trans.max()
    mean_trans_3d = trans.mean()

    # calculate total displacement (combination of translation and rotation)
    # using euler's rotation theorem to convert degrees into distance
    head_radius = 80
    trans_x = motion_diff[:, 0]
    trans_y = motion_diff[:, 1]
    trans_z = motion_diff[:, 2]
    phi = motion_diff[:, 3]
    theta = motion_diff[:, 4]
    psi = motion_diff[:, 5]
    disp = np.sqrt(
        0.2
        * head_radius**2
        * (
            (np.cos(phi) - 1) ** 2
            + (np.sin(phi)) ** 2
            + (np.cos(theta) - 1) ** 2
            + (np.sin(theta)) ** 2
            + (np.cos(psi) - 1) ** 2
            + (np.sin(psi)) ** 2
        )
        + (trans_x**2 + trans_y**2 + trans_z**2)
    )
    max_disp = disp.max()
    mean_disp = disp.mean()

    # calculate number of spikes
    spikes = (disp > 1).sum()
    ratio_spikes = spikes / float(motion_params.shape[0])

    output = [
        max_trans_1d,
        max_rot_1d,
        max_trans_3d,
        mean_trans_3d,
        max_disp,
        mean_disp,
        spikes,
        ratio_spikes,
    ]

    # OPTIONAL
    # indicate whether the scan has too much motion (more than 3mm translation
    # or 3A rotation along a single axis between any two volumes, or
    # greater than 10% of volumes have a displacement > 1mm)
    if flag_bad:
        criteria = [max_trans_1d > 3, max_rot_1d > 3, ratio_spikes > 0.1]
        bad = np.any(criteria)
        output.append(bad)

    return output
