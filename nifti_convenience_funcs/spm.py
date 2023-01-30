#!/Users/danielschonhaut/anaconda2/bin/python

"""
spm.py

Last Edited: 8/24/16

HWNI Location: /home/jagust/dschon/UCSF/Daniel/scripts

Dependencies: Python 2.7, nipype, matlab-spm8, matlab-2014a-spm12

Description: Functions for spm12 coregistration, normalization, realignment,
smoothing, and segmentation, with the ability to pass input arguments 
from the command line.

Examples:
1. Reslice image1 to image2:
        $ spm.py coreg reslice image1 image2

2. Coregister image1 to image2:
        $ spm.py coreg estimate image1 image2

3. Coregister and reslice image1 to image2:
        $ spm.py coreg est_reslice image1 image2

4. Coregister and reslice image1 to image2 and apply to images 3 and 4
        $ spm.py coreg est_reslice image1 image2 image3 image4 

5. Normalize image1 to template and apply to image1
        $ spm.py norm image1 template image1

6. Normalize image1 to template and apply to images 1, 2 and 3
        $ spm.py norm image1 template image1 image2 image3

7. Realign frames 2-6 to frame 1
        $ spm.py realign frame1 frame2 frame3 frame4 frame5 frame6
        OR
        $ spm.py realign frame*
		
8. Realign frames 1, 2, 4 to frame 3
        $ spm.py realign frame3 frame1 frame2 frame4

9. Smooth image1 to 4.5 x 4.5 x 3.5
        $ spm.py smooth image1 4.5 4.5 3.5

10. Segment image1
        $ spm.py segment image1
        
11. Segment image1 and save DARTEL-space c1 and c2
        $ spm.py segment image1 dartel
        
12. Segment image1 and warp c1 and c2 to MNI space (with and without modulation)
        $ spm.py segment image1 warp
        
13. Segment image1 and with DARTEL-space and warped c1 and c2
        $ spm.py segment image1 dartel warp
        
Author: Daniel Schonhaut
Department of Neurology
UCSF
"""

import os
import sys
if __name__ == '__main__':	
    if len(sys.argv) == 1:
        print __doc__
        exit()
import nipype.interfaces.spm as spm

def spm_coreg(jobtype, source, target, other_images=None, n_neighbor=False):
    """Coregister in matlab-spm8.
    
    Keyword arguments:
    source -- the image that is resliced to match the target
    target -- the image that is assumed to remain stationary
    other_images -- any other images that the coreg parameters are applied to
    jobtype -- estwrite, estimate, or write
    """
    coreg = spm.Coregister(matlab_cmd='matlab-spm8')
    coreg.inputs.target = target
    coreg.inputs.source = source
    if other_images is not None:
        coreg.inputs.apply_to_files = other_images
    coreg.inputs.jobtype = jobtype
    if n_neighbor:
        coreg.inputs.write_interp = 0
    coreg.run()

def spm12_coreg(jobtype, source, target, other_images=None, n_neighbor=False):
    """Coregister in matlab-spm12.
    
    Keyword arguments:
    source -- the image that is resliced to match the target
    target -- the image that is assumed to remain stationary
    other_images -- any other images that the coreg parameters are applied to
    jobtype -- estwrite, estimate, or write
    """
    coreg = spm.Coregister(matlab_cmd='matlab-2014a-spm12')
    coreg.inputs.target = target
    coreg.inputs.source = source
    if other_images is not None:
        coreg.inputs.apply_to_files = other_images
    coreg.inputs.jobtype = jobtype
    if n_neighbor:
        coreg.inputs.write_interp = 0
    coreg.run()


def spm_normalize(source, template, images_to_write, bounding_box=[[-90, -126, -73], [90, 90, 108]], n_neighbor=False):
    """Normalize (Estimate & Write) in matlab-spm8.
	
    Keyword arguments:
    template -- a template image to match the source image with
    source -- the image that is warped to match the template
    images_to_write -- these are the images for warping according to the estimated parameters
    bounding_box -- (optional) the bounding box, in mm, of the volume that is to be written
    """
    norm = spm.Normalize(matlab_cmd='matlab-spm8')
    norm.inputs.jobtype = 'estwrite'
    norm.inputs.template = template
    norm.inputs.source = source
    norm.inputs.apply_to_files = images_to_write
    if bounding_box is not None:
        norm.inputs.write_bounding_box = bounding_box
    if n_neighbor:
        norm.inputs.write_interp = 0
    norm.run()
	
def spm12_normalize(source, template, images_to_write, bounding_box=[[-90, -126, -73], [90, 90, 108]], n_neighbor=False):
    """Normalize (Estimate & Write) in matlab-spm12.
	
    Keyword arguments:
    template -- a template image to match the source image with
    source -- the image that is warped to match the template
    images_to_write -- these are the images for warping according to the estimated parameters
    bounding_box -- (optional) the bounding box, in mm, of the volume that is to be written
    """
    norm = spm.Normalize(matlab_cmd='matlab-2014a-spm12')
    norm.inputs.jobtype = 'estwrite'
    norm.inputs.template = template
    norm.inputs.source = source
    norm.inputs.apply_to_files = images_to_write
    if bounding_box is not None:
        norm.inputs.write_bounding_box = bounding_box
    if n_neighbor:
        norm.inputs.write_interp = 0
    norm.run()

def spm_realign(frames):
    """Realign in matlab-spm8.
    
    Keyword arguments:
    frames -- list of paths to the frames you want to realign
              (frames are realigned to the first path in the list)
    """
    realign = spm.Realign(matlab_cmd='matlab-spm8')
    realign.inputs.in_files = frames
    realign.inputs.register_to_mean = True
    results = realign.run()
           
def spm12_realign(frames):
    """Realign in matlab-spm12.
    
    Keyword arguments:
    frames -- list of paths to the frames you want to realign
              (frames are realigned to the first path in the list)
    """
    realign = spm.Realign(matlab_cmd='matlab-2014a-spm12')
    realign.inputs.in_files = frames
    realign.inputs.register_to_mean = True
    results = realign.run()

def spm_smooth(infile, smooth_params, prefix='s'):
    """3D Gaussian smoothing of image volumes in matlab-spm8.
    
    Keyword arguments:
    infile -- path to the image file that you want to smooth
    smooth_params -- a 3-element list with the smoothing parameters
    """
    smooth = spm.Smooth(matlab_cmd='matlab-spm8')
    smooth.inputs.in_files = infile
    smooth.inputs.fwhm = smooth_params
    smooth.inputs.out_prefix = prefix
    smooth.run()
    
def spm12_smooth(infile, smooth_params, prefix='s'):
    """3D Gaussian smoothing of image volumes in matlab-spm12.
    
    Keyword arguments:
    infile -- path to the image file that you want to smooth
    smooth_params -- a 3-element list with the smoothing parameters
    """
    smooth = spm.Smooth(matlab_cmd='matlab-2014a-spm12')
    smooth.inputs.in_files = infile
    smooth.inputs.fwhm = smooth_params
    smooth.inputs.out_prefix = prefix
    smooth.run()
    
def spm_segment(infile):
    """Segment in matlab-spm8.
    
    Keyword arguments:
    infile -- path to the MRI image that you want to segment.
    """
    seg = spm.Segment(matlab_cmd='matlab-spm8')
    seg.inputs.data = infile
    seg.run()

def spm12_segment(infile, dartel=False, warp=False):
    """Segment in matlab-spm12.
    
    Keyword arguments:
    infile -- path to the MRI image that you want to segment.
    """
    seg = spm.NewSegment(matlab_cmd='matlab-2014a-spm12')
    seg.inputs.channel_files = infile
    tpm_path = os.path.join(spm.Info.version('matlab-2014a-spm12')['path'], 'tpm', 'TPM.nii')
    if dartel:
        dartel_params = (True, True)
    else:
        dartel_params = (True, False)
    if warp:
        warp_params = (True, True)
    else:
        warp_params = (False, False)
    tissue1 = ((tpm_path, 1), 1, dartel_params, warp_params)
    tissue2 = ((tpm_path, 2), 1, dartel_params, warp_params)
    tissue3 = ((tpm_path, 3), 2, (True, False), (False, False))
    tissue4 = ((tpm_path, 4), 3, (False, False), (False, False))
    tissue5 = ((tpm_path, 5), 4, (False, False), (False, False))
    tissue6 = ((tpm_path, 6), 2, (False, False), (False, False))
    seg.inputs.tissues = [tissue1, tissue2, tissue3, tissue4, tissue5, tissue6]
    seg.run()

if __name__ == '__main__':	
    cwd = os.getcwd() + '/'
    if len(sys.argv) == 1:
        print __doc__
        exit()
    # Coregistration
    elif sys.argv[1] == 'coreg':
        # Get the needed information from command line arguments
        jobtype = sys.argv[2]
        if jobtype == 'est_reslice':
            jobtype = 'estwrite'
        elif jobtype == 'reslice':
            jobtype = 'write'
        source = cwd + sys.argv[3]
        target = cwd + sys.argv[4]
        if not os.path.exists(source):
            source = sys.argv[3]
        if not os.path.exists(target):
            target = sys.argv[4]
        if len(sys.argv) > 5:
            other_images = [path for path in sys.argv[5:]]
        else:
            other_images = None
        # Run coreg
        translate = {'estimate':'Coregistering', 'estwrite':'Coregistering and reslicing',
					 'write':'Reslicing'}
        msg = '\n{} {} to {}'.format(translate[jobtype], os.path.split(source)[1],
                                     os.path.split(target)[1])
        if other_images is not None:
            msg += ' and also applying parameters to '
            for path in other_images:
                msg += os.path.split(path)[1] + ', '
            msg = msg[:len(msg)-2] + '...'
        else:
            msg += '...'
        print msg
        spm12_coreg(jobtype, source, target, other_images)
        print '\nDone!\n'
    # Normalization
    elif sys.argv[1] == 'norm':
        # Get the needed information from command line arguments
        source = cwd + sys.argv[2]
        template = cwd + sys.argv[3]
        if not os.path.exists(source):
            source = sys.argv[2]
        if not os.path.exists(template):
            template = sys.argv[3]
        images_to_write = [path for path in sys.argv[4:]]
        # Run normalization
        msg = 'Estimating parameters to normalize {} to {},\n' \
			  'and applying to '.format(os.path.split(source)[1],
									 	os.path.split(template)[1])
        for path in images_to_write:
            msg += os.path.split(path)[1] + ', '
        msg = msg[:len(msg)-2] + '...'
        print msg
        spm12_normalize(source, template, images_to_write)
        print '\nDone!\n'	
    # Realign
    elif sys.argv[1] == 'realign':
        # Get the needed information from command line arguments 	
        if len(sys.argv) > 2:
            frames = [path for path in sys.argv[2:]]
        msg = '\nRealigning\n'
        for frame in frames:
            msg += os.path.split(frame)[1] + ', '
        msg = msg[:len(msg)-2] + '...'
        print msg
        spm12_realign(frames)
        print '\nDone!\n'
    # Smooth
    elif sys.argv[1] == 'smooth':
        # Get the needed information from command-line arguments
        infile = sys.argv[2]
        smooth_params = [float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])]
        # Run smooth
        msg = '\nSmoothing {} to {}'.format(infile, smooth_params)
        print msg
        spm12_smooth(infile, smooth_params)
        print '\nDone!\n'    	
    # Segment
    elif sys.argv[1] == 'segment':
        # Get the needed information from command-line arguments
        infile = os.path.join(cwd, sys.argv[2])
        if not os.path.exists(infile):
            infile = sys.argv[2]
        if 'dartel' in sys.argv:
            dartel = True
        else:
            dartel = False
        if 'warp' in sys.argv:
            warp = True
        else:
            warp = False
        # Run segment
        msg = '\nSegmenting {}'.format(infile)
        print msg
        spm12_segment(infile, dartel, warp)
        print '\nDone!\n'
