#!/usr/bin/env python

import sys
import os.path as op
import warnings
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nibabel as nib
from nilearn import plotting
from style import custom_colormaps
from general.basic.helper_funcs import *
import general.array.array_operations as aop
import general.basic.str_methods as strm
import general.nifti.nifti_ops as nops

mpl.rcParams["pdf.fonttype"] = 42


def create_multislice(
    imagef,
    subj=None,
    tracer=None,
    image_date=None,
    title=None,
    display_mode="z",
    cut_coords=[-50, -38, -26, -14, -2, 10, 22, 34],
    cmap=None,
    colorbar=True,
    cbar_tick_format=".1f",
    n_cbar_ticks=5,
    cbar_label=None,
    vmin=0,
    vmax=None,
    hide_cbar_values=False,
    autoscale=False,
    autoscale_values_gt0=True,
    autoscale_min_pct=0.5,
    autoscale_max_pct=99.5,
    crop=True,
    mask_thresh=0.05,
    crop_prop=0.05,
    annotate=False,
    draw_cross=False,
    facecolor=None,
    fontcolor=None,
    font={"tick": 12, "label": 14, "title": 16, "annot": 14},
    figsize=(13.33, 7.5),
    dpi=300,
    pad_figure=True,
    fig=None,
    ax=None,
    save_fig=True,
    outfile=None,
    overwrite=False,
    verbose=True,
    **kws,
):
    """Create a multislice plot of image and return the saved file.

    Parameters
    ----------
    imagef : str
        The path to the image file to plot.
    subj : str, default : None
        The subject ID. Used only for setting the figure title if an
        explicit title is not provided.
    tracer : str, default : None
        The PET tracer used. Used for:
        - Setting the figure title if an explicit title is not provided
        - Setting vmin and vmax if these are not provided and autoscale
          is False
        - Setting cmap, facecolor, and fontcolor if these are not
          provided
    image_date : str, default : None
        The image acquisition date. Used only for setting the figure
        title if an explicit title is not provided.
    title : str, optional
        The figure title.
    display_mode : str, default : 'z'
        The direction of slice cuts (see nilearn.plotting.plot_img):
        - 'x': sagittal
        - 'y': coronal
        - 'z': axial
        - 'ortho': 3 cuts are performed in orthogonal directions
        - 'tiled': 3 cuts are performed and arranged in a 2x2 grid
        - 'mosaic': 3 cuts are performed along multiple rows and columns
    cut_coords : list, default : [-50, -38, -26, -14, -2, 10, 22, 34]
        The MNI coordinates of the point where the cut is performed
        (see nilearn.plotting.plot_img):
        - If display_mode is 'ortho' or 'tiled', this should be a
          3-tuple: (x, y, z)
        - For display_mode == 'x', 'y', or 'z', then these are the
          coordinates of each cut in the corresponding direction
        - If None is given, the cuts are calculated automatically
        - If display_mode is 'mosaic', and the number of cuts is the
          same for all directions, cut_coords can be specified as an
          integer. It can also be a length 3 tuple specifying the number
          of cuts for every direction if these are different
        - If display_mode is 'x', 'y', or 'z', cut_coords can be an
          integer, in which case it specifies the number of cuts to
          perform
    cmap : str, default: None
        The colormap to use. Either a string that is a name of a
        matplotlib colormap, or a matplotlib colormap object. "nih" as
        defined by mricron and "turbo" are also recognized.
    colorbar : bool, default : False
        If True, a colorbar is displayed below the image slices
        showing color mappings from vmin to vmax.
    cbar_tick_format : str, default : '%.2f'
        Controls how to format the tick labels of the colorbar. Ex:
        use "%i" to display as integers.
    n_cbar_ticks : int, default : 3
        The number of ticks to display on the colorbar.
    cbar_label : str, default : None
        The colorbar label. If None, the code will try to infer this
        from the tracer name.
    vmin : float, default: 0
        The minimum value of the colormap range.
    vmax : float, default: None
        The maximum value of the colormap range.
    hide_cbar_values : bool, default : False
        If True, the colorbar values are not displayed but are merely
        labeled from "min" to "max." Overrides n_cbar_ticks.
    autoscale : bool, default: False
        If True, autoscale vmin and vmax according to min and max
        percentiles of image voxel values. Does not override vmin or
        vmax if these variables are already defined (and hence, the
        default behavior is to set vmin to 0 and autoscale only the
        vmax intensity).
    autoscale_values_gt0 : bool, default: True
        If True, colormap intensities are autoscaled using only voxel
        values greater than zero to determine percentile cutoffs.
    autoscale_min_pct: float, default: 0.5
        The percentile of included voxel values to use for autoscaling
        the minimum colormap intensity (vmin).
    autoscale_max_pct: float, default: 99.5
        The percentile of included voxel values to use for autoscaling
        the maximum colormap intensity (vmax).
    crop : bool, default : True
        If True, the code attempts to crop the image to remove empty
        space around the edges.
    mask_thresh : float, default : None
        Cropping threshold for the first image; used together with
        crop_prop to determine how aggresively to remove planes of
        mostly empty space around the image.
    crop_prop : float, default : 0.05
        The cropping threshold for removing empty space around the edges
        of the image.
    annotate : bool, default : False
        If True, positions and L/R annotations are added to the plot.
    draw_cross : bool, default : False
        If True, a cross is drawn on top of the image slices to indicate
        the cut position.
    facecolor : str, default : None
        The background color of the figure.
    fontcolor : str, default : None
        The font color used for all text in the figure.
    font : dict, default : {'tick':12,'label':14,'title':16,'annot':14}
        Font sizes for the different text elements.
    figsize : tuple, default : (13.33, 7.5)
        The figure size in inches (w, h).
    dpi : int, default : 300
        The figure resolution.
    pad_figure : bool, default : True
        If True, whitespace is added at top and bottom of the figure.
    fig : matplotlib.figure.Figure, default : None
        The preexisting figure to use.
    ax : matplotlib.axes.Axes, default : None
        The preexisting axes to use.
    save_fig : bool, default : True
        If True, the figure is saved if overwrite is False or outfile
        doesn't already exist.
    outfile : str, default : None
        The path to the output file. If None, the output file is created
        automatically by appending '_multislice' to the input image
        filename.
    overwrite : bool, default : False
        If True and save_fig, outfile is overwritten if it already
        exists.
    verbose : bool, default : True
        If True, print status messages.
    **kws are passed to nifti_ops.load_nii()

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    outfile : str
        The path to the saved file (or None if save_fig is False and
        outfile does not already exist).
    """
    # Return the outfile if it already exists.
    if outfile is None:
        outfile = (
            strm.add_presuf(imagef, suffix="_multislice")
            .replace(".nii.gz", ".pdf")
            .replace(".nii", ".pdf")
        )
    if op.isfile(outfile) and not overwrite:
        if verbose:
            print(
                "  See existing multislice PDF:"
                + "\n\t{}".format(op.dirname(outfile))
                + "\n\t{}".format(op.basename(outfile))
            )
        return None, outfile
    # Check that the image file exists.
    nops.find_gzip(imagef, raise_error=True)

    # Configure plot parameters.

    # Get min and max values for the colormap using autoscale
    # percentages if autoscale is True and if vmin and vmax are not
    # already defined.
    if autoscale:
        img, dat = nops.load_nii(imagef, **kws)
        if autoscale_values_gt0:
            dat = dat[dat > 0]
        if vmin is None:
            vmin = np.percentile(dat, autoscale_min_pct)
        if vmax is None:
            vmax = np.percentile(dat, autoscale_max_pct)

    # Get tracer-specific plotting parameters.
    tracer, tracer_fancy, vmin, vmax, cmap, facecolor, fontcolor = get_tracer_defaults(
        tracer,
        imagef,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        facecolor=facecolor,
        fontcolor=fontcolor,
    )
    if cmap == "nih":
        cmap = custom_colormaps.nih_cmap()
    elif cmap == "avid":
        cmap = custom_colormaps.avid_cmap()
    elif cmap == "turbo":
        cmap = custom_colormaps.turbo_cmap()

    # Crop the data array.
    img, dat = nops.load_nii(imagef, **kws)
    if crop:
        if mask_thresh is None:
            mask_thresh = vmin * 2
        mask = dat > mask_thresh
        dat = aop.crop_arr3d(dat, mask, crop_prop)
        img = nib.Nifti1Image(dat, img.affine)
        img, *_ = nops.recenter_nii(img)

    # Make the plot.
    plt.close("all")
    if fig is None or ax is None:
        if pad_figure:
            fig, ax = plt.subplots(
                3, 1, height_ratios=[0.75, 5, 1.75], figsize=figsize, dpi=dpi
            )
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax = np.ravel(ax)

    if pad_figure:
        iax = 1
    else:
        iax = 0
    _ax = ax[iax]

    # Format remaining plot parameters.
    if len(cut_coords) == 1:
        cut_coords = cut_coords[0]
    black_bg = True if facecolor == "k" else False
    if display_mode == "mosaic":
        cut_coords = None
        _colorbar = True
        colorbar = False
    elif display_mode in ("ortho", "tiled"):
        _colorbar = True
        colorbar = False
    else:
        _colorbar = False

    # Call the plotting function.
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    _ = plotting.plot_img(
        img,
        cut_coords=cut_coords,
        display_mode=display_mode,
        annotate=annotate,
        draw_cross=draw_cross,
        black_bg=black_bg,
        cmap=cmap,
        colorbar=_colorbar,
        vmin=vmin,
        vmax=vmax,
        title=None,
        figure=fig,
        axes=_ax,
    )
    warnings.resetwarnings()

    # Add the colorbar.
    if colorbar:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=_ax,
            location="bottom",
            pad=0.1,
            shrink=0.25,
            aspect=15,
            drawedges=False,
        )
        cbar.outline.set_color(fontcolor)
        cbar.outline.set_linewidth(1)
        cbar.ax.tick_params(
            labelsize=font["tick"], labelcolor=fontcolor, color=fontcolor, width=1
        )
        if hide_cbar_values:
            cbar.ax.set_xticks([vmin, vmax])
            cbar.ax.set_xticklabels(["Low", "High"])
        else:
            cbar_ticks = np.linspace(vmin, vmax, n_cbar_ticks)
            cbar.ax.set_xticks(cbar_ticks)
            cbar.ax.set_xticklabels(
                ["{:{_}}".format(tick, _=cbar_tick_format) for tick in cbar_ticks]
            )
        if cbar_label is None:
            if tracer is None:
                cbar_label = "Value"
            else:
                cbar_label = f"{tracer_fancy} SUVR"
        cbar.ax.set_xlabel(
            cbar_label,
            fontsize=font["label"],
            color=fontcolor,
            labelpad=8,
        )

    # Add the title.
    if title is None:
        title = op.basename(imagef) + "\n"
        if subj:
            title += f"Participant: {subj}\n"
        if image_date:
            title += f"Scan date: {image_date}\n"
        if tracer:
            title += f"Tracer: {tracer_fancy}\n"
        if hide_cbar_values:
            title += "SUVR range: {:.1f}-{:.1f}".format(vmin, vmax)
    ax[0].set_title(
        title,
        fontsize=font["title"],
        color=fontcolor,
        loc="left",
    )

    # Set the background color.
    for iax in range(len(ax)):
        _ax = ax[iax]
        _ax.set_facecolor(facecolor)
    fig.patch.set_facecolor(facecolor)

    # Get rid of lines at the top and bottom of the figure.
    if pad_figure:
        for iax in [0, 2]:
            _ax = ax[iax]
            _ax.axis("off")

    # Save the figure as a pdf.
    if save_fig:
        fig.savefig(
            outfile,
            facecolor=facecolor,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.2,
        )
        if verbose:
            print(
                "  Saved new multislice PDF:"
                + "\n\t{}".format(op.dirname(outfile))
                + "\n\t{}".format(op.basename(outfile))
            )
    elif not op.isfile(outfile):
        outfile = None

    return fig, outfile


def create_2multislice(
    image1f,
    image2f,
    img1_subj=None,
    img2_subj=None,
    img1_tracer=None,
    img2_tracer=None,
    img1_date=None,
    img2_date=None,
    img1_title=None,
    img2_title=None,
    display_mode="z",
    cut_coords=[-50, -38, -26, -14, -2, 10, 22, 34],
    img1_cmap=None,
    img2_cmap=None,
    img1_colorbar=True,
    img2_colorbar=True,
    img1_cbar_tick_format=".1f",
    img2_cbar_tick_format=".1f",
    img1_n_cbar_ticks=5,
    img2_n_cbar_ticks=5,
    img1_cbar_label=None,
    img2_cbar_label=None,
    img1_vmin=0,
    img2_vmin=0,
    img1_vmax=None,
    img2_vmax=None,
    autoscale=False,
    autoscale_values_gt0=True,
    autoscale_min_pct=0.5,
    autoscale_max_pct=99.5,
    img1_crop=True,
    img2_crop=True,
    img1_mask_thresh=0.05,
    img2_mask_thresh=0.05,
    img1_crop_prop=0.05,
    img2_crop_prop=0.05,
    img1_annotate=False,
    img2_annotate=False,
    img1_draw_cross=False,
    img2_draw_cross=False,
    img1_facecolor=None,
    img2_facecolor=None,
    img1_fontcolor=None,
    img2_fontcolor=None,
    fig_facecolor=None,
    font={"tick": 12, "label": 14, "title": 16, "annot": 14},
    figsize=(13.33, 7.5),
    dpi=300,
    pad_figure=True,
    fig=None,
    ax=None,
    save_fig=True,
    outfile=None,
    overwrite=False,
    verbose=True,
    **kws,
):
    """Create a multislice plot of two images and return the saved file.

    Parameters
    ----------
    image1f : str
        The path to the first image file to plot.
    image2f : str
        The path to the second image file to plot.
    img1_subj : str, default : None
        The subject ID for the first image. Used only for setting
        img1_title if an explicit title is not provided.
    img2_subj : str, default : None
        The subject ID for the second image. Used only for setting
        img2_title if an explicit title is not provided.
    img1_tracer : str, default : None
        The PET tracer used for the first image. Used for:
        - Setting img1_title if an explicit title is not provided
        - Setting img1_vmin and img1_vmax if these are not provided and
          autoscale is False
        - Setting img1_cmap, img1_facecolor, and img1_fontcolor if these
          are not provided
    img2_tracer : str, default : None
        The PET tracer used for the second image. Used for:
        - Setting img2_title if an explicit title is not provided
        - Setting img2_vmin and img2_vmax if these are not provided and
          autoscale is False
        - Setting img2_cmap, img2_facecolor, and img2_fontcolor if these
          are not provided
    img1_date : str, default : None
        The acquisition date for the first image. Used only for setting
        img1_title if an explicit title is not provided.
    img2_date : str, default : None
        The acquisition date for the second image. Used only for setting
        img2_title if an explicit title is not provided.
    img1_title : str, optional
        The axis title for the first image.
    img2_title : str, optional
        The axis title for the second image.
    display_mode : str, default : 'z'
        The direction of slice cuts (see nilearn.plotting.plot_img):
        - 'x': sagittal
        - 'y': coronal
        - 'z': axial
        - 'ortho': 3 cuts are performed in orthogonal directions
        - 'tiled': 3 cuts are performed and arranged in a 2x2 grid
        - 'mosaic': 3 cuts are performed along multiple rows and columns
    cut_coords : list, default : [-50, -38, -26, -14, -2, 10, 22, 34]
        The MNI coordinates of the point where the cut is performed
        (see nilearn.plotting.plot_img):
        - If display_mode is 'ortho' or 'tiled', this should be a
          3-tuple: (x, y, z)
        - For display_mode == 'x', 'y', or 'z', then these are the
          coordinates of each cut in the corresponding direction
        - If None is given, the cuts are calculated automatically
        - If display_mode is 'mosaic', and the number of cuts is the
          same for all directions, cut_coords can be specified as an
          integer. It can also be a length 3 tuple specifying the number
          of cuts for every direction if these are different
        - If display_mode is 'x', 'y', or 'z', cut_coords can be an
          integer, in which case it specifies the number of cuts to
          perform
    img1_cmap : str, default: None
        The colormap to use for the first image. Either a string that is
        a name of a matplotlib colormap, or a matplotlib colormap
        object. "nih" as defined by mricron is also recognized.
    img2_cmap : str, default: None
        The colormap to use for the second image. Either a string that
        is a name of a matplotlib colormap, or a matplotlib colormap
        object. "nih" as defined by mricron is also recognized.
    img1_colorbar : bool, default : False
        If True, a colorbar is displayed below the image slices
        showing color mappings from vmin to vmax.
    img2_colorbar : bool, default : False
        If True, a colorbar is displayed below the image slices
        showing color mappings from vmin to vmax.
    img1_cbar_tick_format : str, default : '%.2f'
        Controls how to format the tick labels of the colorbar. Ex:
        use "%i" to display as integers.
    img2_cbar_tick_format : str, default : '%.2f'
        Controls how to format the tick labels of the colorbar. Ex:
        use "%i" to display as integers.
    img1_n_cbar_ticks : int, default : 3
        The number of ticks to display on the colorbar.
    img2_n_cbar_ticks : int, default : 3
        The number of ticks to display on the colorbar.
    img1_cbar_label : str, default : None
        The colorbar label. If None, the code will try to infer this
        from the tracer name.
    img2_cbar_label : str, default : None
        The colorbar label. If None, the code will try to infer this
        from the tracer name.
    img1_vmin : float, default: 0
        The minimum value of the colormap range for the first image.
    img2_vmin : float, default: 0
        The minimum value of the colormap range for the second image.
    img1_vmax : float, default: None
        The maximum value of the colormap range for the first image.
    img2_vmax : float, default: None
        The maximum value of the colormap range for the second image.
    autoscale : bool, default: False
        If True, autoscale vmin and vmax according to min and max
        percentiles of image voxel values. Does not override vmin or
        vmax if these variables are already defined (and hence, the
        default behavior is to set vmin to 0 and autoscale only the
        vmax intensity).
    autoscale_values_gt0 : bool, default: True
        If True, colormap intensities are autoscaled using only voxel
        values greater than zero to determine percentile cutoffs.
    autoscale_min_pct: float, default: 0.5
        The percentile of included voxel values to use for autoscaling
        the minimum colormap intensity (vmin).
    autoscale_max_pct: float, default: 99.5
        The percentile of included voxel values to use for autoscaling
        the maximum colormap intensity (vmax).
    img1_crop : bool, default : True
        If True, the code attempts to crop the image to remove empty
        space around the edges.
    img2_crop : bool, default : True
        If True, the code attempts to crop the image to remove empty
        space around the edges.
    img1_mask_thresh : float, default : None
        Cropping threshold for the first image; used together with
        img1_crop_prop to determine how aggresively to remove planes
        of mostly empty space around the image.
    img2_mask_thresh : float, default : None
        Cropping threshold for the second image; used together with
        img1_crop_prop to determine how aggresively to remove planes
        of mostly empty space around the image.
    img1_crop_prop : float, default : 0.05
        The cropping threshold for removing empty space around the edges
        of the image.
    img2_crop_prop : float, default : 0.05
        The cropping threshold for removing empty space around the edges
        of the image.
    img1_annotate : bool, default : False
        If True, positions and L/R annotations are added to the plot.
    img2_annotate : bool, default : False
        If True, positions and L/R annotations are added to the plot.
    img1_draw_cross : bool, default : False
        If True, a cross is drawn on top of the image slices to indicate
        the cut position.
    img2_draw_cross : bool, default : False
        If True, a cross is drawn on top of the image slices to indicate
        the cut position.
    img1_facecolor : str, default : None
        The axis background color of the first image.
    img2_facecolor : str, default : None
        The axis background color of the second image.
    img1_fontcolor : str, default : None
        The font color used for all text for the first image.
    img2_fontcolor : str, default : None
        The font color used for all text for the second image.
    fig_facecolor : str, default : None
        The figure background color.
    font : dict, default : {'tick':12,'label':14,'title':16,'annot':14}
        Font sizes for the different text elements.
    figsize : tuple, default : (13.33, 7.5)
        The figure size in inches (w, h).
    dpi : int, default : 300
        The figure resolution.
    pad_figure : bool, default : True
        If True, whitespace is added at top and bottom of the figure.
    fig : matplotlib.figure.Figure, default : None
        The preexisting figure to use.
    ax : matplotlib.axes.Axes, default : None
        The preexisting axes to use.
    save_fig : bool, default : True
        If True, the figure is saved if overwrite is False or outfile
        doesn't already exist.
    outfile : str, default : None
        The path to the output file. If None, the output file is created
        automatically by appending '_multislice' to the input image
        filename.
    overwrite : bool, default : False
        If True and save_fig, outfile is overwritten if it already
        exists.
    verbose : bool, default : True
        If True, print status messages.
    **kws are passed to nifti_ops.load_nii()

    Returns
    -------
    outfile : str
        The path to the saved file (or None if save_fig is False and
        outfile does not already exist).
    """
    # Return the outfile if it already exists.
    if outfile is None:
        outfile = op.join(
            op.dirname(image1f),
            strm.add_presuf(
                "{}__VERSUS__{}".format(
                    op.basename(image1f).split(".")[0],
                    op.basename(image2f).split(".")[0],
                ),
                suffix="_multislice",
            )
            + ".pdf",
        )
    if op.isfile(outfile) and not overwrite:
        if verbose:
            print(
                "  See existing multislice PDF:"
                + "\n\t{}".format(op.dirname(outfile))
                + "\n\t{}".format(op.basename(outfile))
            )
        return outfile
    # Check that the image files exist.
    else:
        _ = nops.find_gzip(image1f, raise_error=True)
        _ = nops.find_gzip(image2f, raise_error=True)

    # Configure plot parameters.

    # Create the figure.
    plt.close("all")
    if fig is None or ax is None:
        if pad_figure:
            fig, ax = plt.subplots(
                5, 1, height_ratios=[0.75, 5, 1, 5, 1.75], figsize=figsize, dpi=dpi
            )
        else:
            fig, ax = plt.subplots(
                3, 1, height_ratios=[5, 0.5, 5], figsize=figsize, dpi=dpi
            )
        ax = np.ravel(ax)

    # Get min and max values for the colormap using autoscale
    # percentages if autoscale is True and if vmin and vmax are not
    # already defined.
    if autoscale:
        img1, dat1 = nops.load_nii(image1f, **kws)
        img2, dat2 = nops.load_nii(image2f, **kws)
        if autoscale_values_gt0:
            dat1 = dat1[dat1 > 0]
            dat2 = dat2[dat2 > 0]
        if np.all([img1_vmin is None, img2_vmin is None]):
            img1_vmin = np.mean(
                [
                    np.percentile(dat1, autoscale_min_pct),
                    np.percentile(dat2, autoscale_min_pct),
                ]
            )
            img2_vmin = img1_vmin
        if np.all([img1_vmax is None, img2_vmax is None]):
            img1_vmax = np.mean(
                [
                    np.percentile(dat1, autoscale_max_pct),
                    np.percentile(dat2, autoscale_max_pct),
                ]
            )
            if img1_tracer is not None:
                if (img1_tracer == "fdg") and (img1_vmax < 2.2):
                    img1_vmax = 2.2
                elif img1_vmax < 2.5:
                    img1_vmax = 2.5
            img2_vmax = img1_vmax

    # Set plot parameters based on tracer if the tracer is known to this
    # script and if the parameters have not already been set. If the
    # tracer is not defined, or not known and can't be inferred from
    # the image filename, then any undefined parameters are set to the
    # default values shown at the bottom of this section.
    tracer_labels = {
        "fbb": "[18F]Florbetaben",
        "fbp": "[18F]Florbetapir",
        "pib": "[11C]PIB",
        "ftp": "[18F]Flortaucipir",
        "fdg": "[18F]FDG",
    }
    for ii in range(2):
        # Get the parameters for the current image.
        if ii == 0:
            imagef = image1f
            subj = img1_subj
            tracer = img1_tracer
            image_date = img1_date
            title = img1_title
            cmap = img1_cmap
            vmin = img1_vmin
            vmax = img1_vmax
            facecolor = img1_facecolor
            fontcolor = img1_fontcolor
            # Get tracer-specific plotting parameters.
            (
                tracer,
                tracer_fancy,
                vmin,
                vmax,
                cmap,
                facecolor,
                fontcolor,
            ) = get_tracer_defaults(
                img1_tracer,
                image1f,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                facecolor=facecolor,
                fontcolor=fontcolor,
            )
            colorbar = img1_colorbar
            cbar_tick_format = img1_cbar_tick_format
            n_cbar_ticks = img1_n_cbar_ticks
            cbar_label = img1_cbar_label
            crop = img1_crop
            mask_thresh = img1_mask_thresh
            crop_prop = img1_crop_prop
            annotate = img1_annotate
            draw_cross = img1_draw_cross
        else:
            imagef = image2f
            subj = img2_subj
            tracer = img2_tracer
            image_date = img2_date
            title = img2_title
            cmap = img2_cmap
            vmin = img2_vmin
            vmax = img2_vmax
            facecolor = img2_facecolor
            fontcolor = img2_fontcolor
            # Get tracer-specific plotting parameters.
            (
                tracer,
                tracer_fancy,
                vmin,
                vmax,
                cmap,
                facecolor,
                fontcolor,
            ) = get_tracer_defaults(
                img1_tracer,
                image1f,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                facecolor=facecolor,
                fontcolor=fontcolor,
            )
            colorbar = img2_colorbar
            cbar_tick_format = img2_cbar_tick_format
            n_cbar_ticks = img2_n_cbar_ticks
            cbar_label = img2_cbar_label
            crop = img2_crop
            mask_thresh = img2_mask_thresh
            crop_prop = img2_crop_prop
            annotate = img2_annotate
            draw_cross = img2_draw_cross

        # Set missing parameters.
        if tracer is None:
            tracer = ""
        _tracer = tracer
        tracer = tracer.lower()
        if tracer not in tracer_labels:
            infile_basename = op.basename(imagef).lower()
            for key in tracer_labels:
                if key in infile_basename:
                    tracer = key
        tracer_fancy = tracer_labels.get(tracer, _tracer)
        if tracer == "fbb":
            if vmin is None:
                vmin = 0
            if vmax is None:
                vmax = 2.5
            if cmap is None:
                cmap = "binary_r"
            if facecolor is None:
                facecolor = "k"
            if fontcolor is None:
                fontcolor = "w"
        elif tracer == "fbp":
            if vmin is None:
                vmin = 0
            if vmax is None:
                vmax = 2.5
            if cmap is None:
                cmap = "binary"
            if facecolor is None:
                facecolor = "w"
            if fontcolor is None:
                fontcolor = "k"
        if tracer == "pib":
            if vmin is None:
                vmin = 0
            if vmax is None:
                vmax = 2.5
            if cmap is None:
                cmap = "nih"
            if facecolor is None:
                facecolor = "k"
            if fontcolor is None:
                fontcolor = "w"
        elif tracer == "ftp":
            if vmin is None:
                vmin = 0
            if vmax is None:
                vmax = 3.7
            if cmap is None:
                cmap = "avid"
            if facecolor is None:
                facecolor = "k"
            if fontcolor is None:
                fontcolor = "w"
        elif tracer == "fdg":
            if vmin is None:
                vmin = 0
            if vmax is None:
                vmax = 2.2
            if cmap is None:
                cmap = "nih"
            if facecolor is None:
                facecolor = "k"
            if fontcolor is None:
                fontcolor = "w"
        else:
            if vmin is None:
                vmin = 0
            if vmax is None:
                vmax = 2.5
            if cmap is None:
                cmap = "nih"
            if facecolor is None:
                facecolor = "k"
            if fontcolor is None:
                fontcolor = "w"

        if cmap == "nih":
            cmap = custom_colormaps.nih_cmap()
        elif cmap == "avid":
            cmap = custom_colormaps.avid_cmap()
        elif cmap == "turbo":
            cmap = custom_colormaps.turbo_cmap()

        # Crop the data array.
        img, dat = nops.load_nii(imagef, **kws)
        if crop:
            if mask_thresh is None:
                mask_thresh = vmin * 2
            mask = dat > mask_thresh
            dat = aop.crop_arr3d(dat, mask, crop_prop)
            img = nib.Nifti1Image(dat, img.affine)
            img, *_ = nops.recenter_nii(img)

        # Make the plot.
        if pad_figure:
            iax = 1 + (ii * 2)
        else:
            iax = 0 + (ii * 2)
        _ax = ax[iax]
        black_bg = True if facecolor == "k" else False
        warnings.filterwarnings("ignore", category=UserWarning)
        _ = plotting.plot_img(
            img,
            cut_coords=cut_coords,
            display_mode=display_mode,
            annotate=annotate,
            draw_cross=draw_cross,
            black_bg=black_bg,
            cmap=cmap,
            colorbar=False,
            vmin=vmin,
            vmax=vmax,
            title=None,
            figure=fig,
            axes=_ax,
        )
        warnings.resetwarnings()

        # Adjust the colorbar.
        if colorbar:  # colorbar:
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=_ax,
                location="bottom",
                pad=0,
                fraction=0.1,
                shrink=0.25,
                aspect=15,
                drawedges=False,
            )
            cbar.outline.set_color(fontcolor)
            cbar.outline.set_linewidth(1)
            cbar.ax.tick_params(
                labelsize=font["tick"], labelcolor=fontcolor, color=fontcolor, width=1
            )
            cbar_ticks = np.linspace(vmin, vmax, n_cbar_ticks)
            cbar.ax.set_xticks(cbar_ticks)
            cbar.ax.set_xticklabels(
                ["{:{_}}".format(tick, _=cbar_tick_format) for tick in cbar_ticks]
            )
            if cbar_label is None:
                if tracer is None:
                    cbar_label = "Value"
                else:
                    cbar_label = f"{tracer_fancy} SUVR"
            cbar.ax.set_xlabel(
                cbar_label,
                fontsize=font["label"],
                color=fontcolor,
                labelpad=4,
            )

        # Add the colorbar.
        if False:
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=_ax,
                location="right",
                orientation="vertical",
                pad=0.25,
                shrink=0.67,
                aspect=15,
                fraction=0.15,
                drawedges=False,
            )
            cbar.outline.set_color(fontcolor)
            cbar.outline.set_linewidth(1)
            cbar.ax.tick_params(
                labelsize=font["tick"], labelcolor=fontcolor, color=fontcolor, width=1
            )
            cbar_ticks = np.linspace(vmin, vmax, n_cbar_ticks)
            cbar.ax.set_yticks(cbar_ticks)
            cbar.ax.set_yticklabels(
                ["{:{_}}".format(tick, _=cbar_tick_format) for tick in cbar_ticks]
            )
            if cbar_label is None:
                if tracer is None:
                    cbar_label = "Value"
                else:
                    cbar_label = f"{tracer_fancy} SUVR"
            cbar.ax.set_ylabel(
                cbar_label,
                fontsize=font["label"],
                color=fontcolor,
                labelpad=6,
                rotation=90,
            )

        # Add the title.
        if title is None:
            _ax = ax[0]
            title = op.basename(imagef) + "\n"
            if subj:
                title += f"Participant: {subj}\n"
            if image_date:
                title += f"Scan date: {image_date}\n"
            if tracer:
                title += f"Tracer: {tracer_fancy}"
        _ax.set_title(
            title,
            fontsize=font["title"],
            color=fontcolor,
            loc="left",
            pad=3,
        )

        # Set the axis background color.
        _ax.set_facecolor(facecolor)

    # Get rid of lines at the top and bottom of the figure.
    if pad_figure:
        for iax in [0, 2, 4]:
            _ax = ax[iax]
            _ax.set_facecolor(facecolor)
            _ax.axis("off")
    else:
        iax = 1
        _ax = ax[iax]
        _ax.axis("off")

    # Set the figure background color.
    if fig_facecolor is None:
        if img1_facecolor == img2_facecolor:
            fig_facecolor = img1_facecolor
        else:
            fig_facecolor = "w"
    fig.patch.set_facecolor(facecolor)

    # Save the figure as a pdf.
    if save_fig:
        fig.savefig(
            outfile,
            facecolor=fig_facecolor,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.2,
        )
        if verbose:
            print(
                "  Saved new multislice PDF:"
                + "\n\t{}".format(op.dirname(outfile))
                + "\n\t{}".format(op.basename(outfile))
            )
    elif not op.isfile(outfile):
        outfile = None

    return outfile


def create_multislice_multi(
    image_files,
    display_mode="z",
    cut_coords=[-50, -38, -26, -14, -2, 10, 22, 34],
    cmap=None,
    n_cbar_ticks=2,
    vmin=0,
    vmax=None,
    annotate=False,
    draw_cross=False,
    autoscale=False,
    crop=True,
    mask_thresh=0.05,
    crop_prop=0.05,
    outfile=None,
    verbose=True,
):
    """Create a multislice plot of multiple images and return the saved file."""
    # Set the output file name.
    if outfile is None:
        outfile = (
            strm.add_presuf(image_files[0], suffix="_multislice")
            .replace(".nii.gz", ".pdf")
            .replace(".nii", ".pdf")
        )
    if verbose and op.isfile(outfile):
        print("Overwriting existing file: {}".format(outfile))

    # Set default parameters.
    if annotate is None:
        annotate = False
    if draw_cross is None:
        draw_cross = False

    # Plot the images.
    with PdfPages(outfile) as pdf:
        for imagef in image_files:
            if verbose and nops.find_gzip(imagef) is None:
                print("Skipping missing file: {}".format(imagef))
                continue
            fig, _ = create_multislice(
                imagef,
                display_mode=display_mode,
                cut_coords=cut_coords,
                cmap=cmap,
                n_cbar_ticks=n_cbar_ticks,
                vmin=vmin,
                vmax=vmax,
                annotate=annotate,
                draw_cross=draw_cross,
                autoscale=autoscale,
                crop=crop,
                mask_thresh=mask_thresh,
                crop_prop=crop_prop,
                pad_figure=False,
                save_fig=False,
            )
            pdf.savefig(fig)
            plt.close()

    if verbose:
        print("Saved {}".format(outfile))

    return outfile


def get_tracer_defaults(
    tracer, imagef=None, vmin=None, vmax=None, cmap=None, facecolor=None, fontcolor=None
):
    """Set undefined plot parameters based on tracer."""
    tracer_labels = {
        "fbb": "[18F]Florbetaben",
        "fbp": "[18F]Florbetapir",
        "nav": "[18F]NAV4694",
        "pib": "[11C]PIB",
        "ftp": "[18F]Flortaucipir",
        "fdg": "[18F]FDG",
    }
    if tracer is None:
        tracer = ""
    _tracer = tracer
    tracer = tracer.lower()
    if (tracer not in tracer_labels) and (imagef is not None):
        infile_basename = op.basename(imagef).lower()
        for key in tracer_labels:
            if key in infile_basename:
                tracer = key
    tracer_fancy = tracer_labels.get(tracer, _tracer)
    if tracer == "fbb":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 2.5
        if cmap is None:
            cmap = "binary_r"
        if facecolor is None:
            facecolor = "k"
        if fontcolor is None:
            fontcolor = "w"
    elif tracer == "fbp":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 2.5
        if cmap is None:
            cmap = "binary"
        if facecolor is None:
            facecolor = "w"
        if fontcolor is None:
            fontcolor = "k"
    elif tracer == "nav":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 2.5
        if cmap is None:
            cmap = "nih"
        if facecolor is None:
            facecolor = "k"
        if fontcolor is None:
            fontcolor = "w"
    elif tracer == "pib":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 2.5
        if cmap is None:
            cmap = "nih"
        if facecolor is None:
            facecolor = "k"
        if fontcolor is None:
            fontcolor = "w"
    elif tracer == "ftp":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 4.5
        if cmap is None:
            cmap = "nih"
        if facecolor is None:
            facecolor = "k"
        if fontcolor is None:
            fontcolor = "w"
    elif tracer == "fdg":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 2.2
        if cmap is None:
            cmap = "nih"
        if facecolor is None:
            facecolor = "k"
        if fontcolor is None:
            fontcolor = "w"
    else:
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 2.5
        if cmap is None:
            cmap = "nih"
        if facecolor is None:
            facecolor = "k"
        if fontcolor is None:
            fontcolor = "w"

    # Import custom colormaps.
    if cmap == "avid":
        cmap = custom_colormaps.avid_cmap()
    elif cmap == "nih":
        cmap = custom_colormaps.nih_cmap()
    elif cmap == "turbo":
        cmap = custom_colormaps.turbo_cmap()

    return tracer, tracer_fancy, vmin, vmax, cmap, facecolor, fontcolor


def _parse_args():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="""Save a PDF file showing multiple slices from 1+ NIfTI scans.""",
        formatter_class=TextFormatter,
        exit_on_error=False,
    )
    parser.add_argument(
        "-i",
        "--images",
        required=True,
        type=str,
        nargs="+",
        help="Paths to 1+ NIfTI images",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        required=True,
        type=str,
        help="Name of the output PDF file to be created",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="z",
        choices=["x", "y", "z", "ortho", "tiled", "mosaic"],
    )
    parser.add_argument(
        "-s",
        "--slices",
        type=int,
        nargs="+",
        default=[-50, -38, -26, -14, -2, 10, 22, 34],
        help=(
            "List of image slices to show along the z-axis, in MNI coordinates\n"
            + "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--crop",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Crop the multislice images to the brain",
    )
    parser.add_argument(
        "--mask_thresh",
        type=float,
        default=0.05,
        help=(
            "Cropping threshold for defining empty voxels outside the brain;\n"
            + "used together with crop_prop to determine how aggresively\n"
            + "to remove planes of mostly empty space around the image\n"
            + "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--crop_prop",
        type=float,
        default=0.05,
        help=(
            "Defines how tightly to crop the brain for multislice creation\n"
            + "(proportion of empty voxels in each plane that are allowed to be cropped)\n"
            + "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--cmap",
        type=str,
        help=(
            "Colormap to use for the multislice images (overrides the\n"
            + "tracer-specific defaults)"
        ),
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Add positions and L/R annotations to the multislice images",
    )
    parser.add_argument(
        "--draw_cross",
        action="store_true",
        help="Draw a cross on the multislice images to indicate the cut position",
    )
    parser.add_argument(
        "--autoscale",
        action="store_true",
        help=(
            "Set multislice vmin and vmax to to 0.01 and the 99.5th percentile\n"
            + "of nonzero values, respectively (overrides --vmin, --vmax,\n"
            + "and tracer-specific default scaling)"
        ),
    )
    parser.add_argument(
        "--vmin",
        type=float,
        help=(
            "Minimum intensity threshold for the multislice images\n"
            + "(overrides the tracer-specific defaults)"
        ),
    )
    parser.add_argument(
        "--vmax",
        type=float,
        help=(
            "Maximum intensity threshold for the multislice images\n"
            + "(overrides the tracer-specific defaults)"
        ),
    )
    parser.add_argument(
        "--n_cbar_ticks",
        type=float,
        default=2,
        help=(
            "Number of ticks to show for the  multislice PDF colorbar\n"
            + "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Run without printing output"
    )

    # Parse the command line arguments
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    return args


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
    # Start the timer.
    timer = Timer(msg="\nTotal runtime: ")

    # Get command line arguments.
    args = _parse_args()
    verbose = True
    if args.quiet:
        verbose = False

    multislicef = create_multislice_multi(
        image_files=args.images,
        outfile=args.outfile,
        display_mode=args.mode,
        cut_coords=args.slices,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        annotate=args.annotate,
        draw_cross=args.draw_cross,
        autoscale=args.autoscale,
        n_cbar_ticks=args.n_cbar_ticks,
        crop=args.crop,
        mask_thresh=args.mask_thresh,
        crop_prop=args.crop_prop,
        verbose=verbose,
    )

    # Print the total runtime.
    if verbose:
        print(timer)

    sys.exit(0)
