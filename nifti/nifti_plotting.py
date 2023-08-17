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
    cut_coords=[-50, -37, -24, -11, 2, 15, 28, 41],
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
    crop_cut=0.05,
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
    """Create a multi-slice plot of image and return the saved file.

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
    cut_coords : list, default : [-50, -37, -24, -11, 2, 15, 28, 41]
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
    crop_cut : float, default : 0.05
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
        dat = aop.crop_arr3d(dat, mask, crop_cut)
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


def nih_cmap():
    """Return the NIH colormap as a matplotlib ListedColormap."""
    n_colors = 256
    nih_colors = np.array(
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
    nih_colors[:, 1:] = nih_colors[:, 1:] / 255
    ii = 0
    nih = []
    for ii in range(nih_colors.shape[0] - 1):
        nih += list(
            sns.blend_palette(
                colors=[nih_colors[ii, 1:], nih_colors[ii + 1, 1:]],
                n_colors=np.rint(np.diff(nih_colors[ii : ii + 2, 0])[0] * n_colors),
            )
        )
    nih = mpl.colors.ListedColormap(nih)
    return nih
