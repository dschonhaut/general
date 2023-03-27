#!/Users/dschonhaut/mambaforge/envs/nipy310/bin/python

"""
spm.py

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

Daniel Schonhaut
Memory and Aging Center
Weill Institute for Neurosciences
University of California, San Francisco
"""


import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        exit()
import os
import os.path as op

sys.path.append(op.join(op.expanduser("~"), "code"))
from general.basic.helper_funcs import Timer
import nipype.interfaces.spm as spm


def spm_coreg(
    source,
    target,
    other_images=None,
    jobtype="estwrite",
    out_prefix="r",
    n_neighbor=False,
):
    """Coregister in matlab-spm12.

    Parameters
    ----------
    jobtype : str
        "estimate" -- Rigid body transform.
        "write" -- Reslice to target voxel dimensions.
        "estwrite" -- Rigid body transform and reslicing.
    source : str
        Path to the image that is coregistered and/or resliced to match
        the target.
    target : str
        Path to the image that serves as the stationary coregistration
        target.
    other_images : list
        List of additional image files to apply coregistration
        parameters to.
    out_prefix : str
        Prefix for the output files.
    n_neighbor : bool
        Use nearest neighbor instead of 4th degree B-spline
        interpolation for image reslicing.
        .
    """
    coreg = spm.Coregister()
    coreg.inputs.target = target
    coreg.inputs.source = source
    if other_images is not None:
        coreg.inputs.apply_to_files = other_images
    coreg.inputs.jobtype = jobtype
    if n_neighbor and jobtype in ["write", "estwrite"]:
        coreg.inputs.write_interp = 0
    coreg.run()


def spm_norm(
    source,
    template,
    images_to_write,
    bounding_box=[[-90, -126, -73], [90, 90, 108]],
    n_neighbor=False,
):
    """Normalize (Estimate & Write) in matlab-spm12.

    Keyword arguments:
    template -- a template image to match the source image with
    source -- the image that is warped to match the template
    images_to_write -- these are the images for warping according to the estimated parameters
    bounding_box -- (optional) the bounding box, in mm, of the volume that is to be written
    """
    norm = spm.Normalize()
    norm.inputs.jobtype = "estwrite"
    norm.inputs.template = template
    norm.inputs.source = source
    norm.inputs.apply_to_files = images_to_write
    if bounding_box is not None:
        norm.inputs.write_bounding_box = bounding_box
    if n_neighbor:
        norm.inputs.write_interp = 0
    norm.run()


def spm_realign(frames):
    """Realign in matlab-spm12.

    Keyword arguments:
    frames -- list of paths to the frames you want to realign
              (frames are realigned to the first path in the list)
    """
    realign = spm.Realign()
    realign.inputs.in_files = frames
    realign.inputs.register_to_mean = True
    results = realign.run()


def spm_segment(infiles):
    """Segment structural images into different tissue classes.

    Parameters
    ----------
    infiles: str
        Filepath(s) to the MRI(s) to segment.
    """
    seg = spm.NewSegment()
    seg.inputs.channel_files = infiles
    seg.inputs.channel_info = (0.0001, 60, (False, False))
    seg.run()


def spm_smooth(infiles, fwhm, prefix="s"):
    """Apply 3D Gaussian smoothing to one or more nifti images.

    Parameters
    ----------
    infiles : list or str
        Filepath(s) for the image(s) to smooth.
    fwhm : list(float, float, float)
        Smoothing kernel size (mm) in each dimension.
    prefix : str
        Output filename prefix.

    Returns
    -------
    outfiles : list
        Filepath(s) to the smoothed outputs.
    """
    smooth = spm.Smooth()
    smooth.inputs.in_files = infiles
    smooth.inputs.fwhm = fwhm
    smooth.inputs.out_prefix = prefix
    smooth.run()
    return smooth._list_outputs()["smoothed_files"]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        exit()

    jobname = sys.argv[1]
    cwd = os.getcwd()
    timer = Timer()

    # Coregister
    # ----------
    if jobname[:5].lower() == "coreg":
        # Get the needed information from command line arguments
        jobtype = sys.argv[2]
        if jobtype == "est_reslice":
            jobtype = "estwrite"
        elif jobtype == "reslice":
            jobtype = "write"
        source = op.join(cwd, sys.argv[3])
        target = op.join(cwd, sys.argv[4])
        if not op.exists(source):
            source = sys.argv[3]
        if not op.exists(target):
            target = sys.argv[4]
        if len(sys.argv) > 5:
            other_images = [path for path in sys.argv[5:]]
        else:
            other_images = None
        # Run coreg
        translate = {
            "estimate": "Coregistering",
            "estwrite": "Coregistering and reslicing",
            "write": "Reslicing",
        }
        msg = "\n{} {} to {}".format(
            translate[jobtype], op.basename(source), op.basename(target)
        )
        if other_images is not None:
            msg += " and also applying parameters to "
            for path in other_images:
                msg += op.basename(path) + ", "
            msg = msg[: len(msg) - 2] + "..."
        else:
            msg += "..."
        print(msg)
        spm_coreg(jobtype, source, target, other_images)
        print("", timer, sep="\n", end="\n" * 2)

    # Normalize
    # ---------
    elif jobname[:4].lower() == "norm":
        source = op.join(cwd, sys.argv[2])
        template = op.join(cwd, sys.argv[3])
        if not op.exists(source):
            source = sys.argv[2]
        if not op.exists(template):
            template = sys.argv[3]
        images_to_write = [path for path in sys.argv[4:]]

        msg = (
            "Estimating parameters to normalize {} to {},\n"
            "and applying to ".format(op.basename(source), op.basename(template))
        )
        for path in images_to_write:
            msg += op.basename(path) + ", "
        msg = msg[: len(msg) - 2] + "..."
        print(msg)
        spm_norm(source, template, images_to_write)
        print("", timer, sep="\n", end="\n" * 2)

    # Realign
    # -------
    elif jobname.lower() == "realign":
        # Get the needed information from command line arguments
        if len(sys.argv) > 2:
            frames = [path for path in sys.argv[2:]]
        msg = "\nRealigning\n"
        for frame in frames:
            msg += op.basename(frame) + ", "
        msg = msg[: len(msg) - 2] + "..."
        print(msg)
        spm_realign(frames)
        print("", timer, sep="\n", end="\n" * 2)

    # Segment
    # -------
    elif jobname[:3].lower() == "seg":
        # Get the needed information from command-line arguments
        infile = op.join(cwd, sys.argv[2])
        if not op.exists(infile):
            infile = sys.argv[2]
        if "dartel" in sys.argv:
            dartel = True
        else:
            dartel = False
        if "warp" in sys.argv:
            warp = True
        else:
            warp = False
        # Run segment
        msg = "\nSegmenting {}".format(infile)
        print(msg)
        spm_segment(infile)
        print("", timer, sep="\n", end="\n" * 2)

    # Smooth
    # ------
    elif jobname.lower() == "smooth":
        fwhm = [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])]
        infiles = sys.argv[5:]
        print(
            "\nSmoothing by ({:.2f}, {:.2f}, {:.2f})".format(*fwhm),
            *infiles,
            sep="\n  "
        )
        spm_smooth(infiles, fwhm)
        print("", timer, sep="\n", end="\n" * 2)
