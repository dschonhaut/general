#!/Users/danielschonhaut/anaconda2/bin/python

"""
compare_imgs.py

Last Edited: 8/15/16

HWNI Location: /home/jagust/UCSF/Daniel/scripts

Dependencies: Python 2.7, numpy, nibabel

Description: This script will test if two image arrays are roughly equal
(every element pair must be within alpha of each other).
Can be called as a function or from the command line.

Also returns a number of statistics comparing similarity between the images.

First two arguments from the command line are the images to compare.
If a third argument is given it's interpretted as a mask to contain
the statistic comparisons (but doesn't affect test of equality).
		   
Author: Daniel Schonhaut
Department of Neurology
UCSF
"""

import sys
import os
import numpy as np
import nibabel
import scipy.stats as stats

def compare_imgs(img1_path, img2_path, mask=None):
    """Compare similarity between 2 images and return if_equal and mean, stdev, and range of voxel differences.
	
    Voxel differences are taken as absolute value, and only considers voxels where at least 1 of the 2 images has a nonzero value.
	
    Keyword arguments:
    img1_path -- 	file path to a nifti image
    img2_path -- 	file path to a nifti image
    mask      --    file path to a nifti image (optional mask for comparing img1 and img2 voxels)
    """

    # Load the input files into numpy matrices
    dat1 = np.nan_to_num(nibabel.load(img1_path).get_data().squeeze()).flatten()
    dat2 = np.nan_to_num(nibabel.load(img2_path).get_data().squeeze()).flatten()
    #nonzero_voxels = np.invert((dat1==0) * (dat2==0))  
    nonzero_voxels = np.intersect1d(np.flatnonzero(dat1), np.flatnonzero(dat2))
    if mask:
        maskdat = np.nan_to_num(nibabel.load(mask).get_data().squeeze()).flatten()
        nonzero_voxels = np.intersect1d(np.flatnonzero(maskdat), nonzero_voxels)
    dat1 = dat1[nonzero_voxels]
    dat2 = dat2[nonzero_voxels]
    
    imgs_equal = False
    if np.allclose(dat1,dat2):
        imgs_equal = True

    dat1_mean = dat1.mean()
    dat2_mean = dat2.mean()
    dat1_pcts = []
    dat2_pcts = []
    pcts = [0, 10, 25, 50, 75, 90, 100]
    for x in pcts:
        dat1_pcts.append(np.percentile(dat1, x))
        dat2_pcts.append(np.percentile(dat2, x))
    dat1_sub_dat2 = (dat1 - dat2)
    dat1_div_dat2 = (dat1 / dat2)
    dat1_sub_dat2_mean = (dat1 - dat2).mean()
    dat1_div_dat2_mean = (dat1 / dat2).mean()
    dat1_sub_dat2_pcts = []
    dat1_div_dat2_pcts = []
    for x in pcts:
        dat1_sub_dat2_pcts.append(np.percentile(dat1_sub_dat2, x))
        dat1_div_dat2_pcts.append(np.percentile(dat1_div_dat2, x))
    num_nonzero = dat1.size
    pct_nonzero = float(dat1.size) / len(nonzero_voxels)
    pcor = stats.pearsonr(dat1, dat2)[0]
    scor = stats.spearmanr(dat1, dat2)[0]
    return [imgs_equal, dat1_mean, dat2_mean, dat1_pcts, dat2_pcts,
            dat1_sub_dat2_mean, dat1_div_dat2_mean, dat1_sub_dat2_pcts, dat1_div_dat2_pcts,
            num_nonzero, pct_nonzero, 
            pcor, scor]
	
if __name__ == '__main__':
    cwd = os.getcwd() + '/'
	
    # Get the images to test from command-line arguments
    if len(sys.argv) in (3,4):
        img1_path = cwd + sys.argv[1]
        if not os.path.exists(img1_path):
            img1_path = sys.argv[1]
		
        img2_path = cwd + sys.argv[2]
        if not os.path.exists(img2_path):
            img2_path = sys.argv[2]
		
        if not os.path.exists(img1_path):
            print '\nCould not find {}\n'.format(img1_path)
            exit()
        if not os.path.exists(img2_path):
            print '\nCould not find {}\n'.format(img2_path)
            exit()
        
        if len(sys.argv) == 4:
            mask = cwd + sys.argv[3]
            if not os.path.exists(mask):
                mask = sys.argv[3]
            if not os.path.exists(mask):
                print '\nCould not find {}\n'.format(mask)
                exit()
        else:
            mask = None
    else:
        print __doc__
        exit()
    	
    # Figure out if the images are equal and print the result
    imgs_equal, dat1_mean, dat2_mean, dat1_pcts, dat2_pcts, dat1_sub_dat2_mean, dat1_div_dat2_mean, dat1_sub_dat2_pcts, dat1_div_dat2_pcts, num_nonzero, pct_nonzero, pcor, scor = compare_imgs(img1_path, img2_path, mask)
    if imgs_equal:
        print '\nThe images are equal.\n'
    else:
        print '\nThe images are NOT equal.\n'
    if mask:
        print '{} ({}%) of masked voxels are nonzero.\n'.format(num_nonzero, round(pct_nonzero*100,2))
    else:
        print '{} ({}%) of voxels are nonzero.\n'.format(num_nonzero, round(pct_nonzero*100,2))
    print 'img1 mean {}, img2 mean {}\n'.format(round(dat1_mean, 4), round(dat2_mean, 4))
    print 'percentile values at 0, 10, 25, 50, 75, 90, 100:\n'
    print '\timg1: {}\n'.format([round(val, 2) for val in dat1_pcts])
    print '\timg2: {}\n'.format([round(val, 2) for val in dat2_pcts])
    print 'mean img1 - img2 = {}, mean img1 / img2 = {}\n'.format(round(dat1_sub_dat2_mean, 4), round(dat1_div_dat2_mean, 4))
    print 'percentile values at 0, 10, 25, 50, 75, 90, 100:\n'
    print '\timg1 - img2: {}\n'.format([round(val, 2) for val in dat1_sub_dat2_pcts])
    print '\timg2 / img2: {}\n'.format([round(val, 2) for val in dat1_div_dat2_pcts])
    print 'pearson {}, spearman {}\n'.format(round(pcor, 4), round(scor, 4))
