"""
extract_indexed.py

Last Edited: 3/10/15

HWNI Location: /home/jagust/UCSF/Daniel/scripts

Dependencies: Python 2.7

Description: Functions to do things with indexed ROI images

Author: Daniel Schonhaut
Department of Neuorology
UCSF
"""

import sys, os
import glob
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import nibabel as nb
sys.path.insert(0,'/home/jagust/UCSF/Daniel/scripts')
import common_funcs as cf
import get_mean_intensity5 as gmi

def get_indexed_mean(pet, aparc, roi, thresh=None, thresh_dir=None):
    """return the mean of pet and voxel-volume within the list of roi indices given.
    if a thresh is given then return mean, voxel-volume, and number of voxels surpassing threshold.
    
    Arguments:
    pet -- file path to a nifti pet image (although, this would work with another image type too)
    aparc -- file path to an indexed image (like aparc+aseg) in the same space as the pet
    roi -- a list of indices to extract from
    thresh -- a threshold to test every voxel in the ROI against
    thresh_dir -- a direction to test the threshold for ('>' will count
                  all voxels above threshold, '<' will count all below)
    """
    # import the data
    pet_img = nb.load(pet)
    pet_dat = pet_img.get_data()
    aparc_img = nb.load(aparc)
    aparc_dat = aparc_img.get_data()
    
    # preprocess the data
    pet_dat = pet_dat.squeeze()
    aparc_dat = aparc_dat.squeeze()
    pet_dat[np.isnan(pet_dat)] = 0
    aparc_dat[np.isnan(aparc_dat)] = 0
    
    # create a mask from the 1 or more indices
    mask = np.zeros(pet_dat.shape)
    for index in roi:
        mask[aparc_dat==index] = 1
    
    # get the mean and return it
    mean_intensity = float(pet_dat[(mask==1) & (pet_dat>0)].mean())
    num_vox = int(len(pet_dat[(mask==1) & (pet_dat>0)]))
    if thresh:
        if thresh_dir == '>':
            vox_surpassing_thresh = len(pet_dat[(mask==1) & (pet_dat>thresh)])
        elif thresh_dir == '<':
            vox_surpassing_thresh = len(pet_dat[(mask==1) & (pet_dat>0) & (pet_dat<thresh)])
        return mean_intensity, num_vox, vox_surpassing_thresh
    else:
        return mean_intensity, num_vox

def get_group_mean_and_variance(pets, aparcs, roi=None):
    """return the mean and std dev for all voxels in pets that are in aparcs and are =! nan or 0 in pets.
    
    Arguments:
    pets -- a list of paths to nifti pet images
    aparcs -- a list of paths to aparc images (one to go with each pet; matches pet to aparc by lbl-id)
    roi -- a list of aparc indices to constrain the analysis to
    """
    for pet in pets[:1]:
        shape = nb.load(pet).shape
        shape += (len(pets),) 
        alldat = np.zeros(shape)
        
    for n, pet in enumerate(pets):
        pet_img = nb.load(pet)
        pet_dat = pet_img.get_data()
        aparc_img = nb.load([aparc for aparc in aparcs 
                             if cf.get_lbl_id(aparc) == cf.get_lbl_id(pet)][0])
        aparc_dat = aparc_img.get_data()
        pet_dat[np.isnan(pet_dat)] = 0
        aparc_dat[np.isnan(aparc_dat)] = 0
        if roi:
            aparc_dat[np.invert(np.in1d(aparc_dat, roi).reshape(aparc_dat.shape))] = 0
        pet_dat[aparc_dat<=0] = 0
        alldat[:,:,:,n] = pet_dat	
        
    return alldat[alldat>0].mean(), alldat[alldat>0].std()

def write_freesurfer_roi_to_file(aparc, roi_dir, roi_name, roi_vals):
    """create a new nifti roi from one or more indices in aparc.
    
    Arguments:
    aparc -- file path to a freesurfer aparc+aseg
    roi_dir -- directory that the new roi will be created in
    roi_name -- file name of the new roi
    roi_vals -- a list of freesurfer indices to use in making the roi
    """
    # import the data
    aparc_img = nb.load(aparc)
    aparc_shape = aparc_img.get_shape()
    aparc_affine = aparc_img.get_affine()
    aparc_dat = np.nan_to_num(aparc_img.get_data())
    
    # create a mask from the 1 or more indices
    mask = np.zeros(aparc_shape)
    mask[np.in1d(aparc_dat, roi_vals).reshape(aparc_shape)] = 1
        
    # write mask to new file
    new_img = nb.Nifti1Image(mask, aparc_affine)
    new_file = os.path.join(roi_dir, roi_name)
    new_img.to_filename(new_file)
    return new_file

def mask_pets_against_thresh(pets, thresh, thresh_dir, output_dir, aparcs=None):
    """for each pet in pets, create a mask of all voxels surpassing a threshold
    and for which the correspond aparc value is > 0 (so a mask would work in place of aparc if desired).
    
    Arguments:
    pets -- a list of paths to nifti pet images
    thresh -- a threshold to test each pet voxel against
    thresh_dir -- direction for the threshold test ('>' or '<' to keep voxels above or below thresh).
                  note that if '<' is used, pet voxels must be > 0 and < thresh.
    output_dir -- directory to create the new files in
    aparcs -- a list of paths to aparc images (one to go with each pet; matches pet to aparc by lbl-id)
    """
    for pet in pets:
        sub = cf.get_lbl_id(pet)
        pet_img = nb.load(pet)
        affine = pet_img.get_affine()
        pet_dat = pet_img.get_data()
        pet_dat[np.isnan(pet_dat)] = 0
        if aparcs:
            aparc_img = nb.load([aparc for aparc in aparcs 
                                 if cf.get_lbl_id(aparc) == cf.get_lbl_id(pet)][0])
            aparc_dat = aparc_img.get_data()
            aparc_dat[np.isnan(aparc_dat)] = 0
        else:
            aparc_dat = np.ones(pet_img.shape)
        mask = np.zeros(pet_img.shape)
        if thresh_dir == '>':
            mask[(aparc_dat>0) & (pet_dat>thresh)] = 1
            output_fname = '{}_vox-above-{}_{}'.format(sub, round(thresh, 2), os.path.split(pet)[1])
        elif thresh_dir == '<':
            mask[(aparc_dat>0) & (pet_dat>0) & (pet_dat<thresh)] = 1
            output_fname = '{}_vox-below-{}_{}'.format(sub, round(thresh, 2), os.path.split(pet)[1])
        output_file = os.path.join(output_dir, output_fname)
        output_img = nb.Nifti1Image(mask, affine)
        output_img.to_filename(output_file)
        return output_file
    return True
