#!/usr/bin/env python

"""
$ extract_rois.py -i [pet1.nii pet2.nii ...] -m [mask1.nii mask2.nii ...]
OR
$ extract_rois.py -i [pet1.nii pet2.nii ...] -a [aparc+aseg1.nii aparc+aseg2.nii ...]
"""

import sys
import os.path as op
import importlib.resources
import argparse
from collections import OrderedDict as od
import numpy as np
import pandas as pd
from general.basic.helper_funcs import Timer
import general.nifti.nifti_ops as nops


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


def get_default_roi_file():
    # Assuming 'general.nifti' is a package and 'fsroi_list.csv' is the file within it
    resource_path = importlib.resources.files("general.nifti")
    file_path = resource_path / "fsroi_list.csv"
    if file_path.is_file():
        return file_path.as_posix()
    else:
        raise FileNotFoundError("Default ROI file not found.")


def _parse_args():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="""
$ extract_rois.py -i [pet1.nii pet2.nii ...] -m [mask1.nii mask2.nii ...]
OR
$ extract_rois.py -i [pet1.nii pet2.nii ...] -a [aparc+aseg1.nii aparc+aseg2.nii ...]

But see options for further customization.

Extract image mean and ROI volume info from 1+ nifti images (-i|--images) and 1+ regions
of interest, which can be provided in the form of binary masks (-m|--masks) or
aparc+aseg style parcellation files (-a|--aparcs).

For parcellations, a 2-column CSV file (-f|--roi_file) is used that contains ROI names
and corresponding integer labels (or, for aggregate ROIs, a list of semicolon-separated
integer labels). If no roi_file is specified, a default with FreeSurfer mappings is used
(https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT).
It is possible to extract values from a subset of ROIs by specifying them by name
(-r|--aparc_rois). Otherwise, all ROIs in the CSV file are extracted by default.
--list_rois can be used to print the ROI names and labels from the CSV file without
doing any actual extractions.

Output is printed to the console by default, but can also be saved to a CSV file
(-o|--outputf). It is also possible to suppress printing the output dataframe to the
console (-q|--quiet).
        """,
        formatter_class=TextFormatter,
        exit_on_error=False,
    )
    parser.add_argument(
        "-i",
        "--images",
        type=str,
        nargs="+",
        help="Paths to 1+ images to extract values from",
    )
    parser.add_argument(
        "-m",
        "--masks",
        type=str,
        nargs="*",
        help="Paths to 1+ masks to apply over each image",
    )
    parser.add_argument(
        "-a",
        "--aparcs",
        type=str,
        nargs="*",
        help=(
            "Paths to 1+ parcellation files with ROI labels. Should match\n"
            + "the number of images, as images[i] is paired with aparcs[i]\n"
            + "for i = 1...len(images)"
        ),
    )
    parser.add_argument(
        "-r",
        "--aparc_rois",
        type=str,
        nargs="*",
        help="Names of ROIs to extract values from",
    )
    parser.add_argument(
        "-f",
        "--roi_file",
        type=str,
        default=get_default_roi_file(),
        help=(
            "Path to the 2-column CSV file with ROI names and int or\n"
            "semicolon-separated labels\n"
            "(default: {})".format(get_default_roi_file())
        ),
    )
    parser.add_argument(
        "-l",
        "--list_rois",
        action="store_true",
        help="List ROI names and labels from the ROIs CSV file",
    )
    parser.add_argument(
        "-o",
        "--outputf",
        type=str,
        help="Output CSV filepath. If not specified, output is printed but not saved",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=str,
        default="long",
        choices=["long", "wide", "l", "w"],
        help=(
            "Shape of the output dataframe. '-o wide' pivots the 'roi' column into\n"
            "multiple columns (one for each region)\n"
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Don't print the output dataframe to the terminal",
    )
    # Print help if no arguments are given.
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    else:
        args = parser.parse_args()
        return args


def load_rois(roi_file):
    """Load dictionary of ROI names to lists of 1+ int labels."""
    rois = pd.read_csv(roi_file)
    rois = od(zip(rois.iloc[:, 0], rois.iloc[:, 1]))
    try:
        rois = {
            k: list(np.unique([int(x) for x in v.split(";")])) for k, v in rois.items()
        }
    except AttributeError:
        pass
    rois = pd.Series(rois)
    rois = pd.Series(rois)
    return rois


if __name__ == "__main__":
    timer = Timer()

    # Get command line arguments.
    args = _parse_args()

    # Print ROI names and labels in a nicely-formatted table.
    def fmt_long_str(x, maxlen=50):
        """Truncate long strings of semicolon-separated values."""
        if len(x) <= maxlen:
            return x
        elif len(x) > maxlen:
            stop = x[maxlen + 4 :].find(";")
            if stop == -1:
                return x
            else:
                # find the last ';'
                start_last = x.rfind(";") + 1
                return x[: stop + maxlen] + "..." + x[start_last:]

    if args.list_rois:
        all_rois = pd.read_csv(args.roi_file)
        all_rois["n_labels"] = all_rois.iloc[:, 1].apply(lambda x: len(x.split(";")))
        all_rois.iloc[:, 1] = all_rois.iloc[:, 1].apply(fmt_long_str)
        print(all_rois.to_markdown(index=False, tablefmt="rst"))
        print(args.roi_file, end="\n" * 2)
        sys.exit(0)

    # Load the ROI dictionary
    all_rois = load_rois(args.roi_file)

    # Check that at least one of masks or aparcs is specified
    if args.masks is None and args.aparcs is None:
        print(
            "ERROR: At least one of --masks (-m) or --aparcs (-a) must be specified\n"
        )
        sys.exit(1)

    # Check that the number of images and parcellations match
    if args.aparcs is not None:
        if (len(args.aparcs) > 1) and (len(args.images) != len(args.aparcs)):
            print(
                "ERROR: Number of images and parcellation files must match,\n"
                + "or there must be only one parcellation file specified.\n"
                + "Found {} images and {} parcellation files".format(
                    len(args.images), len(args.aparcs)
                )
            )
            sys.exit(1)

    # Extract ROI values from masks
    output = []
    if args.masks is not None:
        for img in args.images:
            _output = nops.roi_desc(dat=img, rois=args.masks)
            _output = _output.reset_index()
            _output.insert(0, "image_file", img)
            _output.insert(1, "roi_file", args.masks)
            output.append(_output)

    # Extract ROI values from parcellations
    if args.aparcs is not None:
        # Get ROIs
        keep_rois = {}
        if args.aparc_rois is None:
            keep_rois = all_rois
        else:
            for roi in args.aparc_rois:
                if roi not in all_rois.keys():
                    print(f"WARNING: {roi} missing from {args.roi_file}")
                else:
                    keep_rois[roi] = all_rois[roi]
        # Broadcast inputs if needed
        if (len(args.images) > 1) and (len(args.aparcs) == 1):
            args.aparcs = args.aparcs * len(args.images)
        # Extract ROI values
        for img, aparc in zip(args.images, args.aparcs):
            _output = nops.roi_desc(dat=img, rois=aparc, subrois=keep_rois)
            _output = _output.reset_index()
            _output.insert(0, "image_file", img)
            _output.insert(1, "roi_file", aparc)
            output.append(_output)

    output = pd.concat(output).reset_index(drop=True)
    output = output.rename(columns={"voxels": "voxel_count"})

    # Pivot the output dataframe
    if args.shape in ["wide", "w"]:
        output = output.pivot(
            index=["image_file", "roi_file"],
            columns="roi",
            values=["mean", "voxel_count"],
        )
        output.columns = ["_".join(col[::-1]).strip() for col in output.columns.values]
        output = output.reset_index()

    # Save output.
    if args.outputf is not None:
        output.to_csv(args.outputf, index=False)
        print(f"\nSaved output to {op.abspath(args.outputf)}")

    # Print output.
    if not args.quiet:
        output["image_file"] = output["image_file"].apply(op.basename)
        output["roi_file"] = output["roi_file"].apply(op.basename)
        for col in output.columns:
            if "mean" in col:
                output[col] = output[col].astype(float)
            elif "voxel_count" in col:
                output[col] = output[col].astype(float)
        output.columns = output.columns.str.replace("_", "\n")
        print(
            output.to_markdown(
                index=False,
                tablefmt="rst",
                floatfmt=".4f",
                intfmt=",",
            )
        )

    print(timer)
    sys.exit(0)
