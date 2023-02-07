#!/Users/danielschonhaut/anaconda2/bin/python

"""
plot_frames.py

Last Edited: 9/8/15

HWNI Location: /home/jagust/UCSF/Daniel/scripts

Dependencies: Python 2.7, numpy, nibabel, matplotlib

Description: This script will plot the mean intensities of all voxels
for each frame given as an argument. Intended to be used from the
command line (see examples) though works as a standalone function, too.

Examples:

$ plot_frames.py *.nii

$ plot_frames.py *.nii ../scan2/*.nii
		   
Author: Daniel Schonhaut
Department of Neurology
UCSF
"""

import sys
if (__name__ == '__main__') and (len(sys.argv) == 1):
    print __doc__
    exit()
import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
sys.path.append('/home/jagust/UCSF/Daniel/scripts')
import common_funcs as cf

def plot_frames(frames, save_values=False):
    """Plot the mean value across all voxels for each frame in frames in the order given.
    
    If save_values is True, save a file with the mean intensity for each frame.
    
    Keyword Arguments:
    frames -- a list of file paths
    """
    print 'Plotting mean values for {} frames'.format(len(frames))
    means = {}
    for index, frame in enumerate(frames):
        means[index+1] = [os.path.split(frame)[1], np.nan_to_num(nb.load(frame).get_data()).mean()]
    
    output = 'Mean intensities of frames in order:\n'
    for i in range(1,len(means)+1):
        output += '{}: {} ({})\n'.format(i, round(means[i][1],2),means[i][0])  
    fname = os.path.join(os.path.split(frames[0])[0],'intensities_by_frame.txt')
    f = open(fname, 'w')
    f.write(output)
    f.close()
    os.system('cat {}'.format(fname))
    if not save_values:
        os.system('rm {}'.format(fname))
        
    plt.plot(sorted(means.keys()), [means[i][1] for i in sorted(means.keys())])
    plt.title('{}: Mean intensities of frames in order globbed'.format(cf.get_lbl_id(frames[0])))
    plt.show()
        
if __name__ == '__main__':
	plot_frames(sys.argv[1:])
