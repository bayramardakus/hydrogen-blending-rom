"""Three-layer observer-MPC control architecture -- clean feedback-loop layout."""
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

C_MPC="#2C7FB8"; C_OBS="#374151"; C_FB="#D9822B"; C_PLANT="#3A4A5A"
C_ACT="#374151"; C_MEAS="#2C7FB8"

fig, ax = plt.subplots(figsize=(8.2, 4.8))

# ---- layer boxes (stacked, centred) ----
BX, BW = 1.4, 5.2                      # left x, width of the stacked layers
def box(x, y, w, h, title, sub, fc, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc=fc, ec="none", zorder=3))
    ax.text(x+w/2, y+h*0.62, title, ha="center", va="center", color=tc,
            fontsize=11, fontweight="bold", zorder=4)
    ax.text(x+w/2, y+h*0.26, sub, ha="center", va="center", color=tc,
            fontsize=8.6, zorder=4)

box(BX, 3.55, BW, 1.05, "Layer 1: Supervisory MPC",
    r"$\min\ \sum (x^\top Q x + u^\top R u)$   s.t. pressure, blend & compressor limits", C_MPC)
box(BX, 2.05, BW, 1.05, "Layer 2: Local feedback controllers",
    "realise compressor / injection set-points", C_FB)
box(BX, 0.55, BW, 1.05, "Layer 3: Physical pipeline",
    r"RLC-reduced plant  $\dot{x}=Ax+Bu$", C_PLANT)

# observer box to the right, aligned with Layer 1
OX, OW = 7.3, 2.0
box(OX, 3.55, OW, 1.05, "Kalman observer",
    r"$\hat{x}$ from pressures", C_OBS)

# ---- band labels on the far left ----
for y, txt in [(4.07,"supervisory"), (2.57,"regulatory"), (1.07,"physical")]:
    ax.text(BX-0.28, y, txt, ha="right", va="center", fontsize=9.4,
            color="0.5", rotation=90, style="italic")

def arrow(p0, p1, color, conn="arc3,rad=0", lw=1.8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                 lw=lw, color=color, connectionstyle=conn,
                 shrinkA=0, shrinkB=0, zorder=2))

# ---- actuation path (down the left of the stack) ----
xA = BX + BW*0.32
arrow((xA, 3.55), (xA, 3.10), C_ACT)                 # MPC -> L2
arrow((xA, 2.05), (xA, 1.60), C_ACT)                 # L2 -> L3
# Two lines, not three: the gap between the Layer-1 and Layer-2 blocks is
# 0.45 axis units and three lines at this type size need 0.52, so the first
# line was always drawn on top of the blue block.
ax.text(xA-0.15, 3.325, "compressor &\ninjection set-points", ha="right",
        va="center", fontsize=9.4, color=C_ACT)
ax.text(xA-0.15, 1.82, "actuation", ha="right", va="center",
        fontsize=9.4, color=C_ACT)

# ---- measurement path (up the right, through the observer) ----
xM = BX + BW - 0.5
# plant right-edge -> up to observer bottom (elbow)
arrow((BX+BW, 1.075), (OX+OW*0.5, 3.55),
      C_MEAS, conn="angle,angleA=0,angleB=90,rad=8")
ax.text(BX+BW+0.15, 2.25, "pressure\nmeasurements", ha="left", va="center",
        fontsize=9.4, color=C_MEAS)
# observer -> MPC (state estimate)
arrow((OX, 4.075), (BX+BW, 4.075), C_OBS)
ax.text((OX+BX+BW)/2, 4.30, r"$\hat{x}$", ha="center", va="bottom",
        fontsize=11, color=C_OBS)

ax.set_xlim(0.55, 9.5); ax.set_ylim(0.28, 4.85); ax.axis("off")
fig.tight_layout()
fs.save(fig, FIGDIR + "/fig_architecture.png")
print("saved fig_architecture.png")
