#!/Users/dschonhaut/mambaforge/envs/nipy310/bin/python

"""
spm.py

Description: Functions for automatic nifti recentering and SPM12
coregistration, smoothing, and segmentation, with a full command-line
interface for each.
"""

import sys
import os
import os.path as op
import argparse
import shutil

sys.path.append(op.join(op.expanduser("~"), "code"))
from general.basic.helper_funcs import Timer
import general.nifti.nifti_ops as nops
import numpy as np
import nipype.interfaces.spm as spm


def recenter(images, prefix=None, suffix=None):
    """Recenter images in the center of the voxel grid."""
    if isinstance(images, str):
        images = [images]
    outfiles = []
    for image in images:
        *_, outfile = nops.recenter_nii(image, prefix=prefix, suffix=suffix)
        outfiles.append(outfile)
    return outfiles


def spm_coregister(
    source,
    target,
    other_images=[],
    jobtype="estwrite",
    out_prefix="r",
    n_neighbor=False,
    **kws
):
    """Coregister in matlab-spm12.

    Parameters
    ----------
    source : str
        Path to the image that is coregistered and/or resliced to match
        the target.
    target : str
        Path to the image that serves as the stationary coregistration
        target.
    other_images : list
        List of additional image files to apply coregistration
        parameters to.
    jobtype : str
        "estimate" -- Rigid body transform
        "write" -- Reslice to target voxel dimensions
        "estwrite" -- Rigid body transform and reslicing
    out_prefix : str
        Prefix for the output files.
    n_neighbor : bool
        Reslice using nearest neighbor interpolation instead of B-spline.
    kws : dict
        Additional keyword arguments are passed to spm.Coregister

    Returns
    -------
    outfiles : list
        List of paths to the output files.
    """
    # Check that all files exist.
    if other_images in (None, [], ""):
        other_images = []
    all_files = [source, target] + other_images
    assert all([op.isfile(f) for f in all_files])

    # Initialize the coregistration interface.
    coreg = spm.Coregister(**kws)

    # If the job is to coregister without reslicing, make a copy of each
    # input file using the specified output prefix so input files remain
    # unchanged (default SPM behavior is to overwrite the input header).
    if jobtype == "estimate":
        # Copy the source image.
        _source = op.join(op.dirname(source), out_prefix + op.basename(source))
        shutil.copy(source, _source)
        source = _source
        if len(other_images) > 0:
            # Copy the other images.
            _other_images = []
            for f in other_images:
                _f = op.join(op.dirname(f), out_prefix + op.basename(f))
                shutil.copy(f, _f)
                _other_images.append(_f)
            other_images = _other_images
        outfiles = [source] + other_images
    else:
        outfiles = [op.join(op.dirname(source), out_prefix + op.basename(source))] + [
            op.join(op.dirname(f), out_prefix + op.basename(f)) for f in other_images
        ]

    # Set the coregistration parameters.
    coreg.inputs.source = source
    coreg.inputs.target = target
    if other_images is not None:
        coreg.inputs.apply_to_files = other_images
    coreg.inputs.jobtype = jobtype
    if n_neighbor and jobtype in ["write", "estwrite"]:
        coreg.inputs.write_interp = 0

    # Run coregistration.
    result = coreg.run()
    outfiles = np.concatenate(
        (
            np.atleast_1d(result.outputs.get_traitsfree()["coregistered_source"]),
            np.atleast_1d(
                result.outputs.get_traitsfree().get("coregistered_files", [])
            ),
        )
    ).tolist()
    return outfiles


def spm8_normalize(
    source,
    template,
    other_images=[],
    bounding_box=None,
    n_neighbor=False,
    prefix="w",
    **kws
):
    """Normalize (Estimate & Write) using the old matlab-spm8 algorithm.

    Default template is the 91 x 109 x 91 voxel, 2mm^3 MNI152 T1
    template provided in FSL.

    Parameters
    ----------
    source : str
        Path to the image that is warped to match the template.
    other_images : list
        List of additional image files to apply normalization parameters
        to.
    template : str | None
        Path to the template that the source image is warped to.
    bounding_box : list | None
        List of two lists, each containing three elements, specifying
        the bounding box for the normalized images, in mm.
    n_neighbor : bool
        Warp using nearest neighbor interpolation instead of B-spline.
    prefix : str
        Output filename prefix.
    kws : dict
        Additional keyword arguments are passed to spm.Normalize
    """
    # Check that all files exist.
    if other_images in (None, [], ""):
        other_images = []
    all_files = [source] + other_images
    assert all([op.isfile(f) for f in all_files])
    if template in (None, [], ""):
        FSL_DIR = os.environ("FSL_DIR")
        template = nops.gunzip_nii(
            op.join(FSL_DIR, "data", "standard", "MNI152_T1_2mm.nii.gz"),
            rm_orig=False,
        )
    assert np.isfile(template)

    norm8 = spm.Normalize(**kws)
    norm8.inputs.image_to_align = source
    if len(other_images) > 0:
        norm8.inputs.apply_to_files = other_images
    if bounding_box is not None:
        norm8.inputs.write_bounding_box = bounding_box
    if n_neighbor:
        norm8.inputs.write_interp = 0
    norm8.out_prefix = prefix
    norm8.inputs.jobtype = "estwrite"
    result = norm8.run()
    return result


def spm_normalize(
    source,
    other_images=[],
    tpm=None,
    bounding_box=None,
    n_neighbor=False,
    prefix="w",
    **kws
):
    """Normalize (Estimate & Write) in matlab-spm12.

    Parameters
    ----------
    source : str
        Path to the image that is warped to match the template.
    other_images : list
        List of additional image files to apply normalization parameters
        to.
    tpm : str | None
        Path to the template in the form of 4D tissue probability maps
        to use for normalization. By default, the TPMs distributed with
        SPM12 in MNI space are used.
    bounding_box : list | None
        List of two lists, each containing three elements, specifying
        the bounding box for the normalized images, in mm.
    n_neighbor : bool
        Warp using nearest neighbor interpolation instead of B-spline.
    prefix : str
        Output filename prefix.
    kws : dict
        Additional keyword arguments are passed to spm.Normalize
    """
    # Check that all files exist.
    if other_images in (None, [], ""):
        other_images = []
    all_files = [source] + other_images
    assert all([op.isfile(f) for f in all_files])
    if tpm in (None, [], ""):
        tpm = None
    else:
        assert np.isfile(tpm)

    norm = spm.Normalize12(**kws)
    norm.inputs.image_to_align = source
    if len(other_images) > 0:
        norm.inputs.apply_to_files = other_images
    if tpm is not None:
        norm.inputs.tpm = tpm
    if bounding_box is None:
        bounding_box = [[-78, -112, -50], [78, 76, 85]]  # SPM12 default
    norm.inputs.write_bounding_box = bounding_box
    if n_neighbor:
        norm.inputs.write_interp = 0
    norm.out_prefix = prefix
    norm.inputs.jobtype = "estwrite"
    result = norm.run()
    return result


def spm_realign(infiles, prefix="r", **kws):
    """Realign (Estimate & Reslice) 3D nifti frames in matlab-spm12.

    Uses rigid body transformation and a two pass procedure to realign
    frames to the mean of the images after the first realignment.

    Parameters
    ----------
    infiles : list or str
        Filepath(s) to the 3D nifti frame(s) to realign. Frames are
        realigned to the first path in the list.
    prefix : str
        Output filename prefix.
    kws : dict
        Additional keyword arguments are passed to spm.Realign
    """
    realign = spm.Realign(**kws)
    realign.inputs.in_files = infiles
    realign.inputs.out_prefix = prefix
    realign.inputs.register_to_mean = True
    realign.inputs.jobtype = "estwrite"
    result = realign.run()
    return result


def spm_segment(infiles, keep_matf=False, **kws):
    """Segment structural images into different tissue classes in SPM12.

    Parameters
    ----------
    infiles: list or str
        Filepath(s) to the MRI(s) to segment.
    keep_matf: bool
        If True, keep the transformation parameter file.
    kws : dict
        Additional keyword arguments are passed to spm.NewSegment
    """
    # Check that all files exist.
    if isinstance(infiles, str):
        infiles = [infiles]
    assert all([op.isfile(f) for f in infiles])

    # Initialize the segmentation interface.
    seg = spm.NewSegment(**kws)

    # Set the segmentation parameters.
    seg.inputs.channel_files = infiles
    seg.inputs.channel_info = (0.0001, 60, (False, False))

    # Run segmentation.
    result = seg.run()

    # Remove the transformation parameter files if requested.
    if not keep_matf:
        for f in result.outputs.get_traitsfree()["transformation_mat"]:
            os.remove(f)

    # Return the output files.
    outfiles = np.ravel(
        result.outputs.get_traitsfree()["native_class_images"], order="F"
    ).tolist()
    return outfiles


def spm_smooth(infiles, fwhm=None, res_in=None, res_target=None, prefix="s", **kws):
    """Apply 3D Gaussian smoothing to one or more nifti images in SPM12.

    Parameters
    ----------
    infiles : list or str
        Filepath(s) for the image(s) to smooth.
    fwhm : num > 0 or 3-element list of nums > 0 or None
        Smoothing kernel size (mm) in each dimension. Either res_in and
        res_target must both be specified or fwhm must be specified.
    res_in : num > 0 or 3-element list of nums > 0 or None
        Starting resolution in mm. Either res_in and res_target must
        both be specified or fwhm must be specified.
    res_target : num > 0 or 3-element list of nums > 0 or None
        Target resolution in mm. Either res_in and res_target must
        both be specified or fwhm must be specified.
    prefix : str
        Output filename prefix.
    kws : dict
        Additional keyword arguments are passed to spm.Smooth

    Returns
    -------
    outfiles : list
        Filepath(s) to the smoothed outputs.
    """
    # Check that all files exist.
    if isinstance(infiles, str):
        infiles = [infiles]
    assert all([op.isfile(f) for f in infiles])

    # Check that the smoothing kernel is valid.
    if fwhm is None:
        assert res_in is not None and res_target is not None
        fwhm = calc_3d_smooth(res_in, res_target)
    else:
        assert res_in is None and res_target is None
        if isinstance(fwhm, (int, float, str)):
            fwhm = [float(fwhm)] * 3
        elif len(fwhm) == 1:
            fwhm = [float(fwhm[0])] * 3
        else:
            fwhm = [float(val) for val in fwhm]
        assert len(fwhm) == 3
        assert all([val > 0 for val in fwhm])

    # Initialize the smoothing interface.
    smooth = spm.Smooth(**kws)

    # Set the smoothing parameters.
    smooth.inputs.in_files = infiles
    smooth.inputs.fwhm = fwhm
    smooth.inputs.out_prefix = prefix

    # Run smoothing.
    result = smooth.run()
    outfiles = smooth._list_outputs()["smoothed_files"]
    return outfiles


def calc_3d_smooth(res_in, res_target, verbose=False):
    """Return FWHM of the Gaussian that smooths initial to target resolution.

    Parameters
    ----------
    res_in : float or array-like
        Starting resolution in mm.
    res_target : float or array-like
        Target resolution in mm.

    Returns
    -------
    fwhm : 3-length list
        Amount to smooth by in mm in each dimension.
    """
    if isinstance(res_in, (int, float)):
        res_in = [res_in, res_in, res_in]
    elif len(res_in) == 1:
        res_in = [res_in[0], res_in[0], res_in[0]]
    if isinstance(res_target, (int, float)):
        res_target = [res_target, res_target, res_target]
    elif len(res_target) == 1:
        res_target = [res_target[0], res_target[0], res_target[0]]
    res_in = np.asanyarray(res_in)
    res_target = np.asanyarray(res_target)
    assert res_in.size == res_target.size == 3
    assert res_in.min() > 0 and res_target.min() > 0
    fwhm = np.sqrt((res_target**2) - (res_in**2)).tolist()
    if verbose:
        print("FWHM = {}".format(fwhm))
    return fwhm


def _parse_args():
    """Parse and return command line arguments."""
    examples = _get_examples()
    msg = (
        ("-" * 51)
        + "\n"
        + "See `$ spm JOB -h` for help with specific functions\n"
        + ("-" * 51)
        + "\n\n"
    )
    parser = argparse.ArgumentParser(
        prog="spm",
        description="""Run SPM12 from the command line
-------------------------------
-------------------------
-------------------""",
        exit_on_error=False,
        formatter_class=TextFormatter,
        epilog=msg + "Examples:\n" + examples["all"],
    )
    subparsers = parser.add_subparsers(
        dest="job", metavar="JOB", help="The SPM job to perform"
    )

    # Recenter
    parser_recenter = subparsers.add_parser(
        "recenter",
        help="Recenter",
        formatter_class=TextFormatter,
        epilog="Examples:\n" + examples["recenter"],
    )
    parser_recenter.add_argument(
        "-i",
        "--images",
        required=True,
        type=str,
        nargs="+",
        help="Paths to 1+ images to smooth",
    )
    parser_recenter.add_argument(
        "-p",
        "--prefix",
        default=None,
        type=str,
        help="Output file prefix. Default: %(default)s",
    )
    parser_recenter.add_argument(
        "--suffix",
        default=None,
        type=str,
        help="Output file suffix. Default: %(default)s",
    )

    # Coregister
    parser_coreg = subparsers.add_parser(
        "coreg",
        help="Coregister",
        formatter_class=TextFormatter,
        epilog="Examples:\n" + examples["coreg"],
    )
    parser_coreg.add_argument(
        "-s",
        "--source",
        required=True,
        type=str,
        help="Path to the image that is coregistered/resliced to match the target",
    )
    parser_coreg.add_argument(
        "-t",
        "--target",
        required=True,
        type=str,
        help="Path to the stationary coregistration target",
    )
    parser_coreg.add_argument(
        "-o",
        "--other",
        default=[],
        nargs="+",
        type=str,
        help=(
            "List of other images to which coreg parameters are applied. "
            + "Default: %(default)s"
        ),
    )
    parser_coreg.add_argument(
        "-j",
        "--jobtype",
        type=str,
        default="estwrite",
        choices=["estimate", "write", "estwrite"],
        metavar="JOBTYPE",
        help="{}\n{}\n{}\n\n{}".format(
            "estimate : Rigid body transform",
            "write    : Reslice to target voxel dimensions",
            "estwrite : Rigid body transform and reslicing",
            "Default: %(default)s",
        ),
    )
    parser_coreg.add_argument(
        "-p",
        "--prefix",
        default="r",
        type=str,
        help="Output file prefix. Default: %(default)s",
    )
    parser_coreg.add_argument(
        "-nn",
        "--nearest_neighbor",
        action="store_true",
        help="Reslice using nearest neighbor interpolation instead of B-spline",
    )
    parser_coreg.add_argument(
        "--recenter",
        action="store_true",
        help="Recenter input images before running SPM.",
    )

    # Normalize8
    parser_norm8 = subparsers.add_parser(
        "norm8",
        help="Old Normalize in SPM8",
        formatter_class=TextFormatter,
        epilog="Examples:\n" + examples["norm8"],
    )
    parser_norm8.add_argument(
        "-s",
        "--source",
        required=True,
        type=str,
        help="Path to the image that is warped to match the target",
    )
    parser_norm8.add_argument(
        "-o",
        "--other",
        default=[],
        nargs="+",
        type=str,
        help=(
            "List of other images to which normalization parameters are applied. "
            + "Default: %(default)s"
        ),
    )
    parser_norm8.add_argument(
        "-t",
        "--template",
        type=str,
        help="Path to the target template",
    )
    parser_norm8.add_argument(
        "-bb",
        "--bounding_box",
        help="Bounding box for the normalized images. Do not include any spaces!",
    )
    parser_norm8.add_argument(
        "-nn",
        "--nearest_neighbor",
        action="store_true",
        help="Reslice using nearest neighbor interpolation instead of B-spline",
    )
    parser_norm8.add_argument(
        "-p",
        "--prefix",
        default="w",
        type=str,
        help="Output file prefix. Default: %(default)s",
    )
    parser_norm8.add_argument(
        "--recenter",
        action="store_true",
        help="Recenter input images before running SPM.",
    )

    # Normalize12
    parser_norm = subparsers.add_parser(
        "norm",
        help="Normalize",
        formatter_class=TextFormatter,
        epilog="Examples:\n" + examples["norm"],
    )
    parser_norm.add_argument(
        "-s",
        "--source",
        required=True,
        type=str,
        help="Path to the image that is warped to match the target",
    )
    parser_norm.add_argument(
        "-o",
        "--other",
        default=[],
        nargs="+",
        type=str,
        help=(
            "List of other images to which normalization parameters are applied. "
            + "Default: %(default)s"
        ),
    )
    parser_norm.add_argument(
        "-t",
        "--tpm",
        type=str,
        help="Path to the target TPM template",
    )
    parser_norm.add_argument(
        "-bb",
        "--bounding_box",
        help="Bounding box for the normalized images. Do not include any spaces!",
    )
    parser_norm.add_argument(
        "-nn",
        "--nearest_neighbor",
        action="store_true",
        help="Reslice using nearest neighbor interpolation instead of B-spline",
    )
    parser_norm.add_argument(
        "-p",
        "--prefix",
        default="w",
        type=str,
        help="Output file prefix. Default: %(default)s",
    )
    parser_norm.add_argument(
        "--recenter",
        action="store_true",
        help="Recenter input images before running SPM.",
    )

    # Realign
    parser_realign = subparsers.add_parser(
        "realign",
        help="Realign",
        formatter_class=TextFormatter,
        epilog="Examples:\n" + examples["realign"],
    )
    parser_realign.add_argument(
        "-i",
        "--images",
        required=True,
        type=str,
        nargs="+",
        help="Paths to 1+ images to realign",
    )
    parser_realign.add_argument(
        "-p",
        "--prefix",
        default="r",
        type=str,
        help="Output file prefix. Default: %(default)s",
    )
    parser_realign.add_argument(
        "--recenter",
        action="store_true",
        help="Recenter input images before running SPM.",
    )

    # Segment
    parser_seg = subparsers.add_parser(
        "seg",
        help="Segment",
        formatter_class=TextFormatter,
        epilog="Examples:\n" + examples["seg"],
    )
    parser_seg.add_argument(
        "-i",
        "--images",
        required=True,
        type=str,
        nargs="+",
        help="Paths to 1+ images to segment",
    )
    parser_seg.add_argument(
        "--keep_matf",
        action="store_true",
        help="Keep the .mat transformation parameter file that is generated by SPM",
    )
    parser_seg.add_argument(
        "--recenter",
        action="store_true",
        help="Recenter input images before running SPM.",
    )

    # Smooth
    parser_smooth = subparsers.add_parser(
        "smooth",
        help="Smooth",
        formatter_class=TextFormatter,
        epilog="Examples:\n" + examples["smooth"],
    )
    parser_smooth.add_argument(
        "-i",
        "--images",
        required=True,
        type=str,
        nargs="+",
        help="Paths to 1+ images to smooth",
    )
    parser_smooth.add_argument(
        "-k",
        "--fwhm",
        type=float,
        nargs="+",
        help=(
            "Full width at half maximumum of the Gaussian smoothing kernel, in mm.\n"
            + "If 3 numbers are passed these are applied along the x, y, z dimensions.\n"
            + "If 1 number is passed, it is used as the FWHM for all three dimensions"
        ),
    )
    parser_smooth.add_argument(
        "--res_in",
        type=float,
        nargs="+",
        help=(
            "Resolution of the input images, in mm.\n"
            + "If 3 numbers are passed these are applied along the x, y, z dimensions.\n"
            + "If 1 number is passed, it is used as the FWHM for all three dimensions"
        ),
    )
    parser_smooth.add_argument(
        "--res_target",
        type=float,
        nargs="+",
        help=(
            "Target resolution of the smoothed images, in mm.\n"
            + "If 3 numbers are passed these are applied along the x, y, z dimensions.\n"
            + "If 1 number is passed, it is used as the target FWHM for all three dimensions"
        ),
    )
    parser_smooth.add_argument(
        "-p",
        "--prefix",
        default="s",
        type=str,
        help="Output file prefix. Default: %(default)s",
    )
    parser_smooth.add_argument(
        "--recenter",
        action="store_true",
        help="Recenter input images before running SPM.",
    )

    # # Set general MATLAB/SPM12 execution options and paths.
    # parser.add_argument(
    #     "--matlab_cmd",
    #     type=str,
    #     nargs="+",
    #     action=JoinStrings,
    #     help="Matlab command to use",
    # )
    # parser.add_argument(
    #     "--mfile",
    #     default=True,
    #     action=argparse.BooleanOptionalAction,
    #     help="Run m-code using m-file",
    # )
    # parser.add_argument(
    #     "--paths", type=str, nargs="+", help="Paths to add to matlabpath"
    # )
    # parser.add_argument(
    #     "--use_mcr",
    #     default=True,
    #     action=argparse.BooleanOptionalAction,
    #     help="Run m-code using standalone SPM MCR",
    # )

    # Parse the command line arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    elif (len(sys.argv) == 2) and (sys.argv[1] not in ["-h", "--help"]):
        parser.parse_args(sys.argv[1:] + ["-h"])
        sys.exit()
    args = parser.parse_known_args()
    args = parser.parse_args(args[1], args[0])
    return args


def _get_examples():
    examples = {
        "recenter": """
  Recenter image1.nii and overwrite the original file header:
    $ spm recenter -i image1.nii

  Recenter *.nii images in the current directory and save new files with the prefix 'r':
    $ spm recenter -i *.nii -p r
    """,
        "coreg": """
  Coregister (estimate & reslice) source.nii to target.nii:
    $ spm coreg -s source.nii -t target.nii

  Coregister (estimate & reslice) source.nii to target.nii and apply transformation
  parameters to all frame*.nii images in the current directory:
    $ spm coreg -s source.nii -t target.nii -o frame*.nii

  Coregister (estimate & reslice) source.nii to target.nii using nearest neighbor
  interpolation for reslicing:
    $ spm coreg -s source.nii -t target.nii -nn

  Reslice source.nii to match the voxel dimensions of target.nii:
    $ spm coreg -s source.nii -t target.nii -j write
  """,
        "norm8": """
  [Add SPM8 normalization examples here.]
  """,
        "norm": """
  [Add SPM12 normalization examples here.]
  """,
        "realign": """
  [Add realign examples here.]
  """,
        "seg": """
  Segment image1.nii:
    $ spm seg -i image1.nii

  Segment image1.nii, image2.nii, and image3.nii:
    $ spm seg -i image{1..3}.nii
  """,
        "smooth": """
  Smooth image1.nii by 4.5 mm^3:
    $ spm smooth -i image1.nii -k 4.5

  Smooth image1.nii from 6.5 x 6.5 x 7.5 to 8 mm^3 and prefix output files with 's8':
    $ spm smooth -i image1.nii --res_in 6.5 6.5 7.5 --res_target 8 -p s8
  """,
    }
    examples["all"] = (
        examples["recenter"]
        + examples["coreg"]
        + examples["norm8"]
        + examples["norm"]
        + examples["realign"]
        + examples["seg"]
        + examples["smooth"]
    )
    return examples


class JoinStrings(argparse.Action):
    """Join multiple arguments into a single string."""

    def __init__(self, sep=" ", *args, **kws):
        self.sep = sep
        super().__init__(*args, **kws)

    def __call__(self, parser, namespace, values, option_string):
        setattr(namespace, self.dest, self.sep.join(values))


class TextFormatter(argparse.RawTextHelpFormatter):
    """Custom formatter for argparse help text."""

    # use defined argument order to display usage
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = "usage: "

        # if usage is specified, use that
        if usage is not None:
            usage = usage % dict(prog=self._prog)

        # if no optionals or positionals are available, usage is just prog
        elif usage is None and not actions:
            usage = "%(prog)s" % dict(prog=self._prog)
        elif usage is None:
            prog = "%(prog)s" % dict(prog=self._prog)
            # build full usage string
            action_usage = self._format_actions_usage(actions, groups)  # NEW
            usage = " ".join([s for s in [prog, action_usage] if s])
            # omit the long line wrapping code
        # prefix with 'usage:'
        return "%s%s\n\n" % (prefix, usage)


if __name__ == "__main__":
    args = _parse_args()
    cwd = os.getcwd()
    timer = Timer()

    # Recenter
    # --------
    if args.job == "recenter":
        # Get command line parameters
        images = [nops.gunzip_nii(op.abspath(f)) for f in args.images]

        # Run recenter
        for image in images:
            _ = nops.recenter_nii(image, prefix=args.prefix, suffix=args.suffix)
        print("", timer, sep="\n", end="\n" * 2)

    # Coregister
    # ----------
    if args.job == "coreg":
        # Get command line parameters
        jobtype = args.jobtype
        source = nops.gunzip_nii(op.abspath(args.source))
        target = nops.gunzip_nii(op.abspath(args.target))
        if len(args.other) > 0:
            other_images = [nops.gunzip_nii(op.abspath(f)) for f in args.other]
        else:
            other_images = []

        # Optionally recenter
        if args.recenter:
            for image in [source] + other_images:
                _ = nops.recenter_nii(image)

        # Run coreg
        jobtype_fancy = {
            "estimate": "Coregistering",
            "estwrite": "Coregistering and reslicing",
            "write": "Reslicing",
        }
        msg = "\n{} {} to {}".format(
            jobtype_fancy[jobtype], op.basename(source), op.basename(target)
        )
        if other_images:
            msg += "\nand also applying parameters to:\n\t{}".format(
                "\n\t".join([op.basename(f) for f in other_images])
            )
        print(msg)

        outfiles = spm_coregister(
            source=source,
            target=target,
            other_images=other_images,
            jobtype=jobtype,
            out_prefix=args.prefix,
            n_neighbor=args.nearest_neighbor,
        )
        print(
            "",
            "Files created:\n\t{}".format(
                "\n\t".join([op.basename(f) for f in outfiles])
            ),
            sep="\n",
            end="\n" * 2,
        )
        print("", timer, sep="\n", end="\n" * 2)

    # Segment
    # -------
    elif args.job == "seg":
        images = [nops.gunzip_nii(op.abspath(f)) for f in args.images]

        # Optionally recenter
        if args.recenter:
            for image in images:
                _ = nops.recenter_nii(image)

        # Run segment
        print(
            "\nSegmenting {} images:\n\t{}".format(
                len(images), "\n\t".join([op.basename(f) for f in other_images])
            )
        )
        outfiles = spm_segment(images, keep_matf=args.keep_matf)
        print("", timer, sep="\n", end="\n" * 2)

    # Smooth
    # ------
    elif args.job == "smooth":
        images = [nops.gunzip_nii(op.abspath(f)) for f in args.images]

        # Optionally recenter
        if args.recenter:
            for image in images:
                _ = nops.recenter_nii(image)

        # Run smooth
        if args.fwhm:
            print("Smoothing {} images by {} mm:".format(len(images), args.fwhm))
        else:
            print(
                "Smoothing {} images from {} to {} mm:".format(
                    len(images), args.res_in, args.res_target
                )
            )
        outfiles = spm_smooth(
            images,
            fwhm=args.fwhm,
            res_in=args.res_in,
            res_target=args.res_target,
            prefix=args.prefix,
        )
        print(
            "",
            "Files created:\n\t{}".format(
                "\n\t".join([op.basename(f) for f in outfiles])
            ),
            sep="\n",
            end="\n" * 2,
        )
        print("", timer, sep="\n", end="\n" * 2)
