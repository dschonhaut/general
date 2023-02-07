#!/Users/danielschonhaut/anaconda2/bin/python

"""
get_mean_intensity5.py

Last Edited: 10/22/14

HWNI Location: /home/jagust/UCSF/Daniel/scripts

Dependencies: Python 2.7, numpy, nibabel

Description: This script will mask an roi image over
a brain image, returning the mean and standard deviation for voxels
in the masked region of the brain image. No files are altered in
running this script!
-- Statistics will only be calculated on voxels of the brain image for
   which the brain voxel value > 0 and the roi voxel value in that
   spatial location > 0.5
-- It is assumed that the brain image and roi are the same size
   and are in the same space
-- Called from the command line:
       The script will take up to 3 arguments:
           1. file name of the brain image
    	   2. file name of the roi
	   3 (OPTIONAL). file name of the c1 gray matter mask
    	   e.g. $ get_mean_intensity.py brain.nii roi.nii
-- Imported as a module:
	   The script contains a function, get_mean_intensity, which
	   takes up to 4 arguments:
	       1. file path to the brain image
		   2. file path to the roi (default is None)
		   3. file path to the gray matter mask (default is None)
		   4. gray matter threshold (default is 0.3)
		   
Author: Daniel Schonhaut
Department of Neurology
UCSF
"""

import sys
import os
import numpy as np
import nibabel

def get_mean_intensity(brain_path, roi_path=None, gm_mask_path=None, gm_thresh=0.3):
	"""Mask an roi and/or gray matter mask over a brain image and return mean and stdev.
	
	Keyword arguments:
	brain_path -- 	file path to a nifti brain image. only voxels with values > 0
				  	will be used in the calculations of mean and stdev.
	roi_path   -- 	file path to a nifti roi of the same size as the brain
				  	image. only voxels with values > 0.5 will be used
				  	in the calculations of mean and stdev.
	gm_mask_path -- file path to a nifti gray matter mask of the same size as the
					brain image. only voxels with values >= the gm_thresh will be
					used in the calculations of mean and stdev.
	gm_thresh --    threshold for determining which voxels from the gm_mask will be
					used in masking the brain image (typically gm_mask voxels should
					be probabilities of being gray matter from 0-1)
					
	Notes:
	If an roi_path and gm_mask_path are both given, calculations will be done on voxels
	that are > 0.5 in the roi AND that are >= the gray matter threshold. 
	"""

	# Load the input files into numpy matrices
	brain = nibabel.load(brain_path).get_data()
	brain = brain.astype(float)
	brain[np.isnan(brain)] = 0
	
	if roi_path:
		roi = nibabel.load(roi_path).get_data()
	if gm_mask_path:
		gm_mask = nibabel.load(gm_mask_path).get_data()
	
	# Reshape arrays to eliminate empty dimensions
	brain = brain.squeeze()
	if roi_path:
		roi = roi.squeeze()
		roi[np.isnan(roi)] = 0
	if gm_mask_path:
		gm_mask = gm_mask.squeeze()
		gm_mask[np.isnan(gm_mask)] = 0
	
	# Make the complete array that will be used for masking
	mask = np.zeros(brain.shape)
	if roi_path and not gm_mask_path:
		mask[(roi>0.5) & (brain>0)] = 1
	if gm_mask_path and not roi_path:
		mask[(gm_mask>=gm_thresh) & (brain>0)] = 1
	if roi_path and gm_mask_path:
		mask[(roi>0.5) & (gm_mask>=gm_thresh) & (brain>0)] = 1
	
	# Return mean intensity for the masked region of the brain image
	mean_intensity = brain[mask==1].mean()
	stdev = brain[mask==1].std()
	return float(mean_intensity), float(stdev)

if __name__ == '__main__':
	cwd = os.getcwd() + '/'
	
	# Get the brain image path and roi path from command-line arguments
	if len(sys.argv) in (3,4):
		brain_path = cwd + sys.argv[1]
		if not os.path.exists(brain_path):
			brain_path = sys.argv[1]
		
		roi_path = cwd + sys.argv[2]
		if not os.path.exists(roi_path):
			roi_path = sys.argv[2]
		
		if len(sys.argv) > 3:
			gm_path = cwd + sys.argv[3]		
			if not os.path.exists(gm_path):
				gm_path = sys.argv[3]
		else:
			gm_path = None
		
		if not os.path.exists(brain_path):
			print '\nCould not find {}\n'.format(brain_path)
			exit()
		if not os.path.exists(roi_path):
			print '\nCould not find {}\n'.format(roi_path)
			exit()
	else:
		print __doc__
		exit()
		
	# Calculate mean and stdev over the masked region, 
	# then print the output to the command line
	if gm_path:	
		mean_intensity, stdev = get_mean_intensity(brain_path,roi_path,gm_path)
		msg = '\nMasked {} and {} onto {}\n\nMean intensity in the masked region ' \
		  'is {}\nwith a standard deviation of {}\n' \
		  .format(os.path.split(roi_path)[1], os.path.split(gm_path)[1],
		  		  os.path.split(brain_path)[1], mean_intensity, stdev)
	else:
		mean_intensity, stdev = get_mean_intensity(brain_path,roi_path)
		msg = '\nMasked {} onto {}\n\nMean intensity in the masked region ' \
		  'is {}\nwith a standard deviation of {}\n' \
		  .format(os.path.split(roi_path)[1], os.path.split(brain_path)[1], 
		  		  mean_intensity, stdev)
	print msg
