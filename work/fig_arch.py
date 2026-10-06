"""Three-layer observer-MPC control architecture -- clean feedback-loop layout.

The stack is spaced from the type size. The gaps between
the layer boxes are set from the height two lines of the label type need, and
the Layer-1 caption is set on two lines rather than one over-long line, which
is what used to run past the edge of the box.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figstyle as fs
from matplotlib import rcParams

import os as _os
FIGDIR = _os.environ.get("PAPER_FIGS", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "ms", "figs"))
_os.makedirs(FIGDIR, exist_ok=True)

fs.apply_style_diagram()

C_MPC = "#2C7FB8"; C_OBS = "#374151"; C_FB = "#D9822B"; C_PLANT = "#3A4A5A"
C_ACT = "#374151"; C_MEAS = "#2C7FB8"

FS_TITLE, FS_SUB, FS_LAB = 13.57, 11.83, 12.76

fig, ax = plt.subplots(figsize=(8.2, 4.8))

# ---- layer boxes (stacked, centred) ----
BX, BW = 1.4, 5.2                      # left x, width of the stacked layers
Y1, Y2, Y3 = 3.60, 1.95, 0.35          # box bottoms
H1, H23 = 1.15, 1.05                   # box heights


def box(x, y, w, h, title, sub, fc, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc=fc, ec="none", zorder=3))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", color=tc,
            fontsize=FS_TITLE, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", color=tc,
            fontsize=FS_SUB, zorder=4, linespacing=1.35)


box(BX, Y1, BW, H1, "Layer 1: Supervisory MPC",
    "$\\min\\ \\sum (x^\\top Q x + u^\\top R u)$\n"
    "s.t. pressure, blend and compressor limits", C_MPC)
box(BX, Y2, BW, H23, "Layer 2: Local feedback controllers",
    "realise compressor / injection set-points", C_FB)
box(BX, Y3, BW, H23, "Layer 3: Physical pipeline",
    r"RLC-reduced plant  $\dot{x}=Ax+Bu$", C_PLANT)

# observer box to the right, aligned with Layer 1
OX, OW = 7.00, 2.45
box(OX, Y1, OW, H1, "Kalman observer", r"$\hat{x}$ from pressures", C_OBS)

# ---- band labels on the far left ----
for y, txt in [(Y1 + H1 / 2, "supervisory"), (Y2 + H23 / 2, "regulatory"),
               (Y3 + H23 / 2, "physical")]:
    ax.text(BX - 0.28, y, txt, ha="right", va="center", fontsize=FS_LAB,
            color="0.5", rotation=90, style="italic")


def arrow(p0, p1, color, conn="arc3,rad=0", lw=1.8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                 lw=lw, color=color, connectionstyle=conn,
                 shrinkA=0, shrinkB=0, zorder=2))


# ---- actuation path (down the left of the stack) ----
xA = BX + BW * 0.32
arrow((xA, Y1), (xA, Y2 + H23), C_ACT)                 # MPC -> L2
arrow((xA, Y2), (xA, Y3 + H23), C_ACT)                 # L2 -> L3
ax.text(xA - 0.15, (Y1 + Y2 + H23) / 2, "compressor &\ninjection set-points",
        ha="right", va="center", fontsize=FS_LAB, color=C_ACT, linespacing=1.35)
ax.text(xA - 0.15, (Y2 + Y3 + H23) / 2, "actuation", ha="right", va="center",
        fontsize=FS_LAB, color=C_ACT)

# ---- measurement path (up the right, through the observer) ----
# The riser is placed at 0.8 of the observer box, not at its centre: at the
# centre it passed through the last letter of "measurements".
arrow((BX + BW, Y3 + H23 / 2), (OX + OW * 0.80, Y1),
      C_MEAS, conn="angle,angleA=0,angleB=90,rad=8")
ax.text(BX + BW + 0.15, (Y2 + Y3 + H23) / 2 + 0.55, "pressure\nmeasurements",
        ha="left", va="center", fontsize=FS_LAB, color=C_MEAS, linespacing=1.35)
# observer -> MPC (state estimate)
arrow((OX, Y1 + H1 / 2), (BX + BW, Y1 + H1 / 2), C_OBS)
ax.text((OX + BX + BW) / 2, Y1 + H1 / 2 + 0.12, r"$\hat{x}$", ha="center",
        va="bottom", fontsize=FS_TITLE, color=C_OBS)

ax.set_xlim(0.55, 9.5); ax.set_ylim(0.12, 4.92); ax.axis("off")
fig.tight_layout()
fs.save(fig, FIGDIR + "/fig_architecture.png")
print("saved fig_architecture.png")
