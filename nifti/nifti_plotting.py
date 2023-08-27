import os.path as op
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nilearn import plotting
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
    autoscale=False,
    autoscale_values_gt0=True,
    autoscale_min_pct=0.5,
    autoscale_max_pct=99.5,
    crop=True,
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
        The direction of slice cuts (see nilearn.plotting.plot_anat):
        - 'x': sagittal
        - 'y': coronal
        - 'z': axial
        - 'ortho': 3 cuts are performed in orthogonal directions
        - 'tiled': 3 cuts are performed and arranged in a 2x2 grid
        - 'mosaic': 3 cuts are performed along multiple rows and columns
    cut_coords : list, default : [-50, -38, -26, -14, -2, 10, 22, 34]
        The MNI coordinates of the point where the cut is performed
        (see nilearn.plotting.plot_anat):
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
        defined by mricron is also recognized.
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
        return outfile
    # Check that the image file exists.
    else:
        _ = nops.find_gzip(imagef, raise_error=True)

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
            vmax = 2.5
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

    if cmap == "nih":
        cmap = nih_cmap()

    # Crop the data array.
    img, dat = nops.load_nii(imagef, **kws)
    if crop:
        mask = dat > (vmin * 2)
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
    black_bg = True if facecolor == "k" else False
    warnings.filterwarnings("ignore", category=UserWarning)
    _ = plotting.plot_anat(
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

    return outfile


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
    img1_mask_thresh=None,
    img2_mask_thresh=None,
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
        The direction of slice cuts (see nilearn.plotting.plot_anat):
        - 'x': sagittal
        - 'y': coronal
        - 'z': axial
        - 'ortho': 3 cuts are performed in orthogonal directions
        - 'tiled': 3 cuts are performed and arranged in a 2x2 grid
        - 'mosaic': 3 cuts are performed along multiple rows and columns
    cut_coords : list, default : [-50, -38, -26, -14, -2, 10, 22, 34]
        The MNI coordinates of the point where the cut is performed
        (see nilearn.plotting.plot_anat):
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
            colorbar = img1_colorbar
            cbar_tick_format = img1_cbar_tick_format
            n_cbar_ticks = img1_n_cbar_ticks
            cbar_label = img1_cbar_label
            crop = img1_crop
            mask_thresh = img1_mask_thresh
            crop_prop = img1_crop_prop
            annotate = img1_annotate
            draw_cross = img1_draw_cross
            facecolor = img1_facecolor
            fontcolor = img1_fontcolor
        else:
            imagef = image2f
            subj = img2_subj
            tracer = img2_tracer
            image_date = img2_date
            title = img2_title
            cmap = img2_cmap
            vmin = img2_vmin
            vmax = img2_vmax
            colorbar = img2_colorbar
            cbar_tick_format = img2_cbar_tick_format
            n_cbar_ticks = img2_n_cbar_ticks
            cbar_label = img2_cbar_label
            crop = img2_crop
            mask_thresh = img2_mask_thresh
            crop_prop = img2_crop_prop
            annotate = img2_annotate
            draw_cross = img2_draw_cross
            facecolor = img2_facecolor
            fontcolor = img2_fontcolor

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
            cmap = nih_cmap()
        elif cmap == "avid":
            cmap = avid_cmap()

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
        _ = plotting.plot_anat(
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


def nih_cmap():
    """Return the NIH colormap as a matplotlib ListedColormap."""
    n_colors = 256
    color_list = np.array(
        [
            [0, 0, 0, 0],
            [0.059, 85, 0, 170],
            [0.122, 0, 0, 85],
            [0.247, 0, 0, 255],
            [0.309, 0, 85, 255],
            [0.372, 0, 170, 170],
            [0.434, 0, 255, 170],
            [0.497, 0, 255, 0],
            [0.559, 85, 255, 85],
            [0.625, 255, 255, 0],
            [0.75, 255, 85, 0],
            [0.85, 255, 0, 0],
            [1.0, 172, 0, 0],
        ]
    )
    color_list[:, 1:] = color_list[:, 1:] / 255
    ii = 0
    cmap = []
    for ii in range(color_list.shape[0] - 1):
        cmap += list(
            sns.blend_palette(
                colors=[color_list[ii, 1:], color_list[ii + 1, 1:]],
                n_colors=np.rint(np.diff(color_list[ii : ii + 2, 0])[0] * n_colors),
            )
        )
    cmap = mpl.colors.ListedColormap(cmap)
    return cmap


def avid_cmap():
    """Return the Avid colormap as a matplotlib ListedColormap."""
    n_colors = 255
    color_list = np.array(
        [
            [0 / n_colors, 0, 0, 0],
            [1 / n_colors, 0, 0, 0],
            [2 / n_colors, 0, 2, 3],
            [3 / n_colors, 0, 5, 7],
            [4 / n_colors, 1, 7, 10],
            [5 / n_colors, 1, 9, 14],
            [6 / n_colors, 1, 12, 17],
            [7 / n_colors, 1, 14, 20],
            [8 / n_colors, 2, 16, 24],
            [9 / n_colors, 2, 19, 27],
            [10 / n_colors, 2, 21, 30],
            [11 / n_colors, 2, 23, 34],
            [12 / n_colors, 2, 25, 37],
            [13 / n_colors, 3, 28, 41],
            [14 / n_colors, 3, 30, 44],
            [15 / n_colors, 3, 32, 47],
            [16 / n_colors, 3, 35, 51],
            [17 / n_colors, 4, 37, 54],
            [18 / n_colors, 4, 39, 57],
            [19 / n_colors, 4, 42, 61],
            [20 / n_colors, 4, 44, 64],
            [21 / n_colors, 4, 46, 68],
            [22 / n_colors, 5, 49, 71],
            [23 / n_colors, 5, 51, 74],
            [24 / n_colors, 5, 53, 78],
            [25 / n_colors, 5, 56, 81],
            [26 / n_colors, 5, 58, 84],
            [27 / n_colors, 6, 60, 88],
            [28 / n_colors, 6, 62, 91],
            [29 / n_colors, 6, 65, 95],
            [30 / n_colors, 6, 67, 98],
            [31 / n_colors, 7, 69, 101],
            [32 / n_colors, 7, 72, 105],
            [33 / n_colors, 7, 74, 108],
            [34 / n_colors, 7, 77, 108],
            [35 / n_colors, 7, 79, 107],
            [36 / n_colors, 7, 82, 107],
            [37 / n_colors, 7, 85, 107],
            [38 / n_colors, 7, 87, 107],
            [39 / n_colors, 8, 90, 106],
            [40 / n_colors, 8, 93, 106],
            [41 / n_colors, 8, 95, 106],
            [42 / n_colors, 8, 98, 106],
            [43 / n_colors, 8, 101, 105],
            [44 / n_colors, 8, 103, 105],
            [45 / n_colors, 8, 106, 105],
            [46 / n_colors, 8, 109, 105],
            [47 / n_colors, 8, 111, 104],
            [48 / n_colors, 8, 114, 104],
            [49 / n_colors, 9, 117, 104],
            [50 / n_colors, 9, 120, 104],
            [51 / n_colors, 9, 122, 103],
            [52 / n_colors, 9, 125, 103],
            [53 / n_colors, 9, 128, 103],
            [54 / n_colors, 9, 130, 103],
            [55 / n_colors, 9, 133, 102],
            [56 / n_colors, 9, 136, 102],
            [57 / n_colors, 9, 138, 102],
            [58 / n_colors, 9, 141, 102],
            [59 / n_colors, 10, 144, 101],
            [60 / n_colors, 10, 146, 101],
            [61 / n_colors, 10, 149, 101],
            [62 / n_colors, 10, 152, 101],
            [63 / n_colors, 10, 154, 100],
            [64 / n_colors, 10, 157, 100],
            [65 / n_colors, 10, 157, 99],
            [66 / n_colors, 11, 157, 98],
            [67 / n_colors, 11, 156, 97],
            [68 / n_colors, 11, 156, 96],
            [69 / n_colors, 12, 156, 95],
            [70 / n_colors, 12, 156, 95],
            [71 / n_colors, 12, 156, 94],
            [72 / n_colors, 13, 156, 93],
            [73 / n_colors, 13, 155, 92],
            [74 / n_colors, 13, 155, 91],
            [75 / n_colors, 14, 155, 90],
            [76 / n_colors, 14, 155, 89],
            [77 / n_colors, 14, 155, 88],
            [78 / n_colors, 15, 154, 87],
            [79 / n_colors, 15, 154, 86],
            [80 / n_colors, 15, 154, 85],
            [81 / n_colors, 16, 154, 85],
            [82 / n_colors, 16, 154, 84],
            [83 / n_colors, 16, 154, 83],
            [84 / n_colors, 17, 153, 82],
            [85 / n_colors, 17, 153, 81],
            [86 / n_colors, 17, 153, 80],
            [87 / n_colors, 18, 153, 79],
            [88 / n_colors, 18, 153, 78],
            [89 / n_colors, 18, 152, 77],
            [90 / n_colors, 19, 152, 76],
            [91 / n_colors, 19, 152, 75],
            [92 / n_colors, 19, 152, 75],
            [93 / n_colors, 20, 152, 74],
            [94 / n_colors, 20, 152, 73],
            [95 / n_colors, 20, 151, 72],
            [96 / n_colors, 21, 151, 71],
            [97 / n_colors, 21, 151, 70],
            [98 / n_colors, 26, 147, 69],
            [99 / n_colors, 31, 144, 68],
            [100 / n_colors, 36, 140, 67],
            [101 / n_colors, 41, 137, 66],
            [102 / n_colors, 46, 133, 65],
            [103 / n_colors, 52, 129, 64],
            [104 / n_colors, 57, 126, 63],
            [105 / n_colors, 62, 122, 62],
            [106 / n_colors, 67, 118, 60],
            [107 / n_colors, 72, 115, 59],
            [108 / n_colors, 77, 111, 58],
            [109 / n_colors, 82, 108, 57],
            [110 / n_colors, 87, 104, 56],
            [111 / n_colors, 92, 100, 55],
            [112 / n_colors, 97, 97, 54],
            [113 / n_colors, 103, 93, 53],
            [114 / n_colors, 108, 89, 52],
            [115 / n_colors, 113, 86, 51],
            [116 / n_colors, 118, 82, 50],
            [117 / n_colors, 123, 79, 49],
            [118 / n_colors, 128, 75, 48],
            [119 / n_colors, 133, 71, 47],
            [120 / n_colors, 138, 68, 46],
            [121 / n_colors, 143, 64, 45],
            [122 / n_colors, 148, 60, 43],
            [123 / n_colors, 153, 57, 42],
            [124 / n_colors, 159, 53, 41],
            [125 / n_colors, 164, 50, 40],
            [126 / n_colors, 169, 46, 39],
            [127 / n_colors, 174, 42, 38],
            [128 / n_colors, 179, 39, 37],
            [129 / n_colors, 184, 35, 36],
            [130 / n_colors, 185, 36, 36],
            [131 / n_colors, 186, 37, 36],
            [132 / n_colors, 187, 38, 36],
            [133 / n_colors, 188, 39, 36],
            [134 / n_colors, 189, 40, 36],
            [135 / n_colors, 189, 41, 36],
            [136 / n_colors, 190, 42, 36],
            [137 / n_colors, 191, 43, 36],
            [138 / n_colors, 192, 43, 37],
            [139 / n_colors, 193, 44, 37],
            [140 / n_colors, 194, 45, 37],
            [141 / n_colors, 195, 46, 37],
            [142 / n_colors, 196, 47, 37],
            [143 / n_colors, 197, 48, 37],
            [144 / n_colors, 198, 49, 37],
            [145 / n_colors, 199, 50, 37],
            [146 / n_colors, 199, 51, 37],
            [147 / n_colors, 200, 52, 37],
            [148 / n_colors, 201, 53, 37],
            [149 / n_colors, 202, 54, 37],
            [150 / n_colors, 203, 55, 37],
            [151 / n_colors, 204, 56, 37],
            [152 / n_colors, 205, 57, 37],
            [153 / n_colors, 206, 58, 37],
            [154 / n_colors, 207, 58, 38],
            [155 / n_colors, 208, 59, 38],
            [156 / n_colors, 209, 60, 38],
            [157 / n_colors, 209, 61, 38],
            [158 / n_colors, 210, 62, 38],
            [159 / n_colors, 211, 63, 38],
            [160 / n_colors, 212, 64, 38],
            [161 / n_colors, 213, 65, 38],
            [162 / n_colors, 214, 66, 38],
            [163 / n_colors, 215, 68, 38],
            [164 / n_colors, 215, 71, 38],
            [165 / n_colors, 216, 73, 38],
            [166 / n_colors, 216, 75, 38],
            [167 / n_colors, 217, 77, 38],
            [168 / n_colors, 217, 80, 37],
            [169 / n_colors, 218, 82, 37],
            [170 / n_colors, 218, 84, 37],
            [171 / n_colors, 219, 86, 37],
            [172 / n_colors, 219, 89, 37],
            [173 / n_colors, 220, 91, 37],
            [174 / n_colors, 220, 93, 37],
            [175 / n_colors, 221, 95, 37],
            [176 / n_colors, 221, 98, 37],
            [177 / n_colors, 222, 100, 37],
            [178 / n_colors, 222, 102, 36],
            [179 / n_colors, 223, 104, 36],
            [180 / n_colors, 223, 107, 36],
            [181 / n_colors, 224, 109, 36],
            [182 / n_colors, 224, 111, 36],
            [183 / n_colors, 225, 113, 36],
            [184 / n_colors, 225, 116, 36],
            [185 / n_colors, 226, 118, 36],
            [186 / n_colors, 226, 120, 36],
            [187 / n_colors, 227, 122, 36],
            [188 / n_colors, 227, 125, 35],
            [189 / n_colors, 228, 127, 35],
            [190 / n_colors, 228, 129, 35],
            [191 / n_colors, 229, 131, 35],
            [192 / n_colors, 229, 134, 35],
            [193 / n_colors, 230, 136, 35],
            [194 / n_colors, 230, 138, 34],
            [195 / n_colors, 231, 140, 34],
            [196 / n_colors, 231, 142, 33],
            [197 / n_colors, 231, 144, 33],
            [198 / n_colors, 232, 146, 32],
            [199 / n_colors, 232, 148, 32],
            [200 / n_colors, 233, 150, 31],
            [201 / n_colors, 233, 152, 30],
            [202 / n_colors, 233, 154, 30],
            [203 / n_colors, 234, 156, 29],
            [204 / n_colors, 234, 158, 29],
            [205 / n_colors, 234, 160, 28],
            [206 / n_colors, 235, 162, 28],
            [207 / n_colors, 235, 164, 27],
            [208 / n_colors, 235, 166, 26],
            [209 / n_colors, 236, 168, 26],
            [210 / n_colors, 236, 169, 25],
            [211 / n_colors, 237, 171, 25],
            [212 / n_colors, 237, 173, 24],
            [213 / n_colors, 237, 175, 23],
            [214 / n_colors, 238, 177, 23],
            [215 / n_colors, 238, 179, 22],
            [216 / n_colors, 238, 181, 22],
            [217 / n_colors, 239, 183, 21],
            [218 / n_colors, 239, 185, 21],
            [219 / n_colors, 239, 187, 20],
            [220 / n_colors, 240, 189, 19],
            [221 / n_colors, 240, 191, 19],
            [222 / n_colors, 241, 193, 18],
            [223 / n_colors, 241, 195, 18],
            [224 / n_colors, 241, 197, 17],
            [225 / n_colors, 242, 199, 17],
            [226 / n_colors, 242, 201, 16],
            [227 / n_colors, 242, 202, 16],
            [228 / n_colors, 242, 203, 16],
            [229 / n_colors, 242, 204, 17],
            [230 / n_colors, 242, 205, 17],
            [231 / n_colors, 241, 207, 17],
            [232 / n_colors, 241, 208, 17],
            [233 / n_colors, 241, 209, 18],
            [234 / n_colors, 241, 210, 18],
            [235 / n_colors, 241, 211, 18],
            [236 / n_colors, 241, 212, 18],
            [237 / n_colors, 241, 213, 19],
            [238 / n_colors, 241, 214, 19],
            [239 / n_colors, 241, 215, 19],
            [240 / n_colors, 241, 216, 19],
            [241 / n_colors, 240, 218, 20],
            [242 / n_colors, 240, 219, 20],
            [243 / n_colors, 240, 220, 20],
            [244 / n_colors, 240, 221, 20],
            [245 / n_colors, 240, 222, 21],
            [246 / n_colors, 240, 223, 21],
            [247 / n_colors, 240, 224, 21],
            [248 / n_colors, 240, 225, 21],
            [249 / n_colors, 240, 226, 22],
            [250 / n_colors, 240, 227, 22],
            [251 / n_colors, 239, 229, 22],
            [252 / n_colors, 239, 230, 22],
            [253 / n_colors, 239, 231, 23],
            [254 / n_colors, 239, 232, 23],
            [255 / n_colors, 239, 233, 23],
        ]
    )
    color_list[:, 1:] = color_list[:, 1:] / 255
    ii = 0
    cmap = []
    for ii in range(color_list.shape[0] - 1):
        cmap += list(
            sns.blend_palette(
                colors=[color_list[ii, 1:], color_list[ii + 1, 1:]],
                n_colors=np.rint(np.diff(color_list[ii : ii + 2, 0])[0] * n_colors),
            )
        )
    cmap = mpl.colors.ListedColormap(cmap)
    return cmap
