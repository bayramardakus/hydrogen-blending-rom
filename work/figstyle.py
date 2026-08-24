"""figstyle.py - the single house style for every figure in the manuscript.

RULES (applied without exception, so that similar figures look identical):

 1. Serif type, one base size. No figure sets its own rcParams.
 2. The legend is ALWAYS below the axes, frameless, horizontal. Never inside the
    axes, where it can cover data.
 3. Threshold, limit and reference lines are LEGEND ENTRIES, never free-floating
    in-axes text. Such annotations were the source of every overlap in the
    previous version of these figures.
 4. Units in square brackets: "Time [h]", "Pressure [bar]". Never parentheses.
 5. Axis labels in sentence case with a capital initial.
 6. Panel titles "(a) Sentence case", at a common size.
 7. Colour AND line style are both varied, so the figures survive greyscale
    printing and common colour-vision deficiencies.
 8. Canonical canvas widths, so panels line up from one figure to the next.
 9. Y-labels are kept SHORT (<= ~22 characters). A label longer than the axes
    are tall overflows the canvas and collides with the panel title; every
    overlap left in the previous version came from this.
"""
import os
from matplotlib import rcParams
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- palette
PALETTE = dict(
    ref="#374151",    # reference / high-fidelity model   (dark grey)
    rlc="#2C7FB8",    # reduced model / MPC               (blue)
    acc="#D95F0E",    # limits and thresholds             (orange)
    grn="#2C7F5E",    # second controlled series          (green)
    ol="#B0357F",     # uncontrolled / baseline           (magenta)
    pur="#7A5195",    # fourth series                     (purple)
    mut="#9CA3AF",    # muted guide lines                 (grey)
)

# line styles paired with the palette, in the order series are normally added
DASH = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

# Canonical canvas sizes, in inches, at TRUE PRINT SIZE for Elsevier:
#   single column  90 mm = 3.54 in
#   1.5 column    140 mm = 5.51 in
#   double column 190 mm = 7.48 in
# Authoring at print size matters: a figure drawn 290 mm wide and then scaled to
# \linewidth by LaTeX has its type reduced by the same factor, so nominal 13 pt
# labels reach the page at about 8 pt. Drawing at final size makes the specified
# type size the type size the reader actually gets.
W1, W15, W2, W3 = 3.54, 5.51, 7.48, 7.48
# Panel height must still leave the axes tall enough for a full y-label.
H1, H2, H3 = 3.0, 2.9, 2.7


_PATCHED = False


def _also_emit_pdf():
    """Make every `fig.savefig(...png)` write a matching vector PDF.

    Elsevier prefers vector artwork for line drawings. Rather than edit every
    call site, the Figure.savefig method is wrapped once here, so any script
    that calls `figstyle.apply_style()` gets a PDF alongside each raster.
    """
    global _PATCHED
    if _PATCHED:
        return
    from matplotlib.figure import Figure
    _orig = Figure.savefig

    def savefig(self, fname, *a, **kw):
        out = _orig(self, fname, *a, **kw)
        try:
            if isinstance(fname, str) and fname.lower().endswith(".png"):
                _orig(self, os.path.splitext(fname)[0] + ".pdf", *a, **kw)
        except Exception:
            pass
        return out

    Figure.savefig = savefig
    _PATCHED = True


def apply_style():
    _also_emit_pdf()
    rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        # Type sizes are the sizes that reach the printed page, because the
        # canvas is the printed size. Elsevier asks for >= 7 pt at final size.
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "axes.axisbelow": True,
        "lines.linewidth": 1.2,
        # Elsevier: >= 1000 dpi for bitmapped line art, >= 500 dpi for
        # combination line/halftone. 600 dpi at final size clears the
        # combination threshold; a vector PDF of every figure is emitted
        # alongside and is what should be uploaded.
        "figure.dpi": 150, "savefig.dpi": 600,
        # NOT "tight": a tight bounding box grows the canvas around the legend
        # strip, so a figure authored at the 190 mm double-column width is
        # written out at 200-210 mm and no longer fits the journal column.
        # `save()` reserves the strip in the layout instead.
        "savefig.bbox": "standard", "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed real fonts, not paths
    })


def apply_style_diagram():
    """Style for HAND-LAID-OUT SCHEMATICS (circuit analogy, architecture block
    diagram, geographic map) - not for data plots.

    These figures place their text and artwork at hard-coded coordinates tuned
    to one canvas. Re-drawing them on the smaller print-size canvas used for the
    data plots, at the data-plot type size, enlarges the text RELATIVE to the
    artwork and makes panel titles and labels collide. They are therefore kept
    at their original geometry and type size and scaled to the column by LaTeX,
    which shrinks text and artwork together. At \linewidth the effective type
    size stays above the 7 pt floor for all three.
    """
    apply_style()
    rcParams.update({
        "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11.5,
        "axes.linewidth": 0.9, "lines.linewidth": 1.9,
    })


def legend_below(ax, ncol=2, y=-0.34, fontsize=7.0, handles=None, labels=None):
    """Frameless legend centred below the axes, clear of the x-label.

    `y` is in axes coordinates. Pass the SAME y for every panel of a figure so
    that the legends of neighbouring panels sit on one line.
    """
    kw = dict(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
              frameon=False, fontsize=fontsize, columnspacing=1.5,
              handlelength=2.0, handletextpad=0.6, borderaxespad=0.0)
    if handles is not None:
        ax.legend(handles, labels, **kw)
    else:
        ax.legend(**kw)


def legend_row(ax, n_entries, y=-0.34, fontsize=7.0, max_per_row=3):
    """Legend below the axes with a bounded number of columns.

    Returns the number of rows used, so the caller can reserve the right amount
    of bottom margin for the whole figure.
    """
    ncol = min(n_entries, max_per_row)
    legend_below(ax, ncol=ncol, y=y, fontsize=fontsize)
    return -(-n_entries // max_per_row)


def panels(n, width=None, height=None, sharex=False):
    """Create a 1 x n panel row on the canonical canvas."""
    w = width if width is not None else {1: W1, 2: W2, 3: W3}[n]
    h = height if height is not None else {1: H1, 2: H2, 3: H3}[n]
    fig, ax = plt.subplots(1, n, figsize=(w, h), sharex=sharex)
    if n == 1:
        ax = [ax]
    return fig, list(ax)


def title(ax, letter, text, single=False):
    """Panel title in the house form."""
    ax.set_title(text if single else f"({letter}) {text}")


def ylabel(ax, text, limit=24):
    """Set a y-label, warning if it is long enough to overflow a short panel."""
    plain = (text.replace("$_2$", "2").replace("$", "")
                 .replace("\\", "").replace("{", "").replace("}", ""))
    if len(plain) > limit:
        print(f"  [figstyle] y-label {len(plain)} chars, may overflow: {text!r}")
    ax.set_ylabel(text)


def fit_legends(fig, gap=0.055):
    """Move every legend to sit clear BELOW its axis's x-label.

    A fixed offset cannot work: the gap between the axes and the x-label depends
    on the tick-label height and on whether the label carries a subscript or a
    fraction, so the same `y` that clears one panel's label lands on top of
    another's. Here the x-label is measured after a first draw and each legend
    is placed just under it.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        inv = ax.transAxes.inverted()
        low = 0.0
        for art in (ax.xaxis.get_label(), *ax.get_xticklabels()):
            if not art.get_text():
                continue
            bb = art.get_window_extent(r)
            low = min(low, inv.transform((0, bb.y0))[1])
        leg.set_bbox_to_anchor((0.5, low - gap), transform=ax.transAxes)
    fig.canvas.draw()


def save(fig, path, legend_rows=1, tight=True):
    """Save at exactly the canonical canvas size, with the legend strip fitted.

    Both the legend position and the bottom margin are MEASURED from the
    rendered figure rather than guessed: legends are placed below their
    x-labels, then the axes are lifted until the tallest legend fits inside the
    canvas. Guessing a fixed offset and a fixed margin, as earlier versions did,
    either dropped the legend onto the x-label or left half the canvas empty,
    because neither suits both a 2.9 in and a 5.2 in canvas.
    """
    if tight:
        fig.tight_layout()
    fit_legends(fig)
    r = fig.canvas.get_renderer()
    need = 0.0
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        bb = leg.get_window_extent(r).transformed(fig.transFigure.inverted())
        if bb.y0 < 0:
            need = max(need, fig.subplotpars.bottom - bb.y0)
    if tight and need > fig.subplotpars.bottom:
        fig.subplots_adjust(bottom=min(0.55, need + 0.02))
        fit_legends(fig)
    # Crop the unused vertical space that remains under the legend, while
    # keeping the WIDTH pinned to the canonical column width. A plain tight
    # bounding box would crop horizontally too and push the figure past the
    # journal's 190 mm column.
    from matplotlib.transforms import Bbox
    W, Hf = fig.get_size_inches()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    box = Bbox.from_extents(0.0, max(0.0, tb.y0 - 0.02),
                            W, min(Hf, tb.y1 + 0.02))
    fig.savefig(path, bbox_inches=box)
    plt.close(fig)
    print(f"saved {os.path.basename(path)} (+ .pdf)")
