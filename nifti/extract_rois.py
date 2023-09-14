#!/usr/bin/env python

"""
$ extract_rois.py [pet.nii] [aparc+aseg.nii] [roi1 roi2 ...]
"""

import sys
import os
import os.path as op
import argparse
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


def _parse_args():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="""Extract mean PET and ROI volume from 1+ ROIs.""",
        formatter_class=TextFormatter,
        exit_on_error=False,
    )
    parser.add_argument(
        "pet",
        type=str,
        help="Path to PET nifti that you want to extract mean ROI values from",
    )
    parser.add_argument(
        "aparc", type=str, help="Path to coregistered aparc+aseg nifti with ROI labels"
    )
    parser.add_argument(
        "rois",
        type=str,
        nargs="+",
        default=None,
        help="Names of ROIs to extract values from",
    )
    args = parser.parse_args()
    return args


def roi_map():
    """Map ROI names to FreeSurfer labels."""
    rois = {
        "meta_temporal": [
            18,
            54,
            1006,
            1007,
            1009,
            1015,
            1016,
            2006,
            2007,
            2009,
            2015,
            2016,
        ],
    }
    return rois


if __name__ == "__main__":
    # Get command line arguments.
    args = _parse_args()

    # Get ROIs.
    all_rois = roi_map()
    keep_rois = {}
    if args.rois is None:
        keep_rois = all_rois
    else:
        for roi in args.rois:
            if roi not in all_rois.keys():
                print(f"{roi} not in the ROI dictionary")
            else:
                keep_rois[roi] = all_rois[roi]

    # Extract ROI values.
    output = nops.roi_desc(dat=args.pet, rois=args.aparc, sub_rois=keep_rois)

    # Print output.
    print("PET: {}".format(args.pet))
    print("aparc+aseg: {}".format(args.aparc))
    print(output, end="\n" * 2)
    sys.exit(0)
