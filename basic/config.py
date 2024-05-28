import matplotlib as mpl


def get_plot_defaults():
    """Return a dictionary of default plot settings."""
    from style.colors import get_colors

    colors, palettes = get_colors()
    params = {
        "colors": colors,
        "colws": {
            1: 2.05,
            2: 3.125,
            3: 7.28346,
        },
        "font": {
            "tick": 10,
            "label": 12,
            "title": 14,
            "annot": 12,
        },
        "lws": {
            "axis": 0.8,
            "line": 1.5,
            "marker": 0.8,
        },
        "pad": {
            "tick": 2,
            "label": 5,
            "title": 8,
        },
        "palettes": palettes,
    }
    return params


def set_rcparams(rcparams):
    params = get_plot_defaults()
    colors = [
        "2E45B8",
        "3EBCD2",
        "FF4983",
        "1DC9A4",
        "F9C31F",
        "B38FE7",
        "F97A1F",
        "E3120B",
    ]
    colws = params["colws"]
    font = params["font"]
    lws = params["lws"]
    pad = params["pad"]
    rcparams["axes.axisbelow"] = True
    rcparams["axes.formatter.offset_threshold"] = 2
    rcparams["axes.grid"] = False
    rcparams["axes.grid.which"] = "both"
    rcparams["axes.labelpad"] = pad["label"]
    rcparams["axes.labelsize"] = font["label"]
    rcparams["axes.linewidth"] = lws["axis"]
    rcparams["axes.prop_cycle"] = mpl.cycler("color", colors)
    rcparams["axes.spines.right"] = False
    rcparams["axes.spines.top"] = False
    rcparams["axes.titlepad"] = pad["title"]
    rcparams["axes.titlesize"] = font["title"]
    rcparams["figure.autolayout"] = True
    rcparams["figure.dpi"] = 300
    rcparams["figure.figsize"] = (colws[3], colws[3] * 0.618034)
    rcparams["figure.labelsize"] = font["label"]
    rcparams["figure.subplot.hspace"] = 0.2
    rcparams["figure.subplot.wspace"] = 0.2
    rcparams["figure.titlesize"] = font["title"]
    rcparams["font.sans-serif"] = "Arial"
    rcparams["font.serif"] = "Times New Roman"
    rcparams["font.family"] = "sans-serif"
    rcparams["grid.alpha"] = 0.5
    rcparams["grid.color"] = "#B7C6CF"
    rcparams["grid.linewidth"] = 0.2
    rcparams["hist.bins"] = 30
    rcparams["legend.borderaxespad"] = 0
    rcparams["legend.fontsize"] = font["tick"]
    rcparams["legend.frameon"] = False
    rcparams["legend.handletextpad"] = 0.4
    rcparams["legend.markerscale"] = 1
    rcparams["legend.title_fontsize"] = font["label"]

    rcparams["lines.color"] = colors[0]
    rcparams["lines.linewidth"] = lws["line"]
    rcparams["lines.markeredgewidth"]: lws["marker"]
    rcparams["lines.markersize"] = 8
    rcparams["patch.facecolor"] = colors[1]
    rcparams["patch.linewidth"] = lws["marker"]
    rcparams["pdf.fonttype"] = 42
    rcparams["savefig.bbox"] = "tight"
    rcparams["savefig.directory"] = "~/Downloads"
    rcparams["savefig.format"] = "pdf"
    rcparams["savefig.pad_inches"] = 0.05
    rcparams["xtick.labelsize"] = font["tick"]
    rcparams["xtick.major.size"] = 4
    rcparams["xtick.major.width"] = lws["axis"]
    rcparams["xtick.minor.ndivs"] = 2
    rcparams["ytick.labelsize"] = font["tick"]
    rcparams["ytick.major.size"] = 4
    rcparams["ytick.major.width"] = lws["axis"]
    rcparams["ytick.minor.ndivs"] = 2
    return rcparams
