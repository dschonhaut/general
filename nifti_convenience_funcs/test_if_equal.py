#!/Users/danielschonhaut/anaconda2/bin/python

"""
test_if_equal.py

Last Edited: 9/24/14

HWNI Location: /home/jagust/UCSF/Daniel/scripts

Dependencies: Python 2.7, numpy, nibabel

Description: This script will test if two arrays are roughly equal
(every element pair must be within alpha of each other).
Can be called as a function or from the command line.
		   
Author: Daniel Schonhaut
Department of Neurology
UCSF
"""

import sys
import os
import numpy as np
import nibabel

def test_if_equal(img1_path, img2_path, alpha):
	"""Test if two image arrays are within alpha of each other at every point
	
	Keyword arguments:
	img1_path -- 	file path to a nifti image
	img2_path -- 	file path to a nifti image
	alpha     --    threshold for comparison
	"""

	# Load the input files into numpy matrices
	dat1 = nibabel.load(img1_path).get_data().squeeze()
	dat2 = nibabel.load(img2_path).get_data().squeeze()
	return np.allclose(dat1,dat2)
	#for x in range(dat1.shape[0]):
	#	for y in range(dat1.shape[1]):
	#		for z in range(dat1.shape[2]):
	#			if abs(dat1[x,y,z] - dat2[x,y,z]) > alpha:
	#				return False
	#return True
	
if __name__ == '__main__':
	cwd = os.getcwd() + '/'
	
	# Get the images to test from command-line arguments
	if len(sys.argv) == 3:
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
	else:
		print __doc__
		exit()
		
	# Figure out if the images are equal and print the result
	ans = test_if_equal(img1_path, img2_path, 0.01)
	if ans:
		print '\nThe images are equal.\n'
	else:
		print '\nThe images are NOT equal.\n'
