"""
fig_rlc_circuit.py
------------------
The paper's central concept, as a figure: a pipe segment (with its pressure
drop, flow inertia, wall friction and gas compressibility) mapped onto an
equivalent resistor-inductor-capacitor branch, and the meshed network assembled
as an RLC circuit by nodal analysis.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
import figstyle as fs
from figstyle import apply_style, PALETTE

import os as _os
FIGDIR = _os.environ.get("PAPER_FIGS", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "ms", "figs"))
_os.makedirs(FIGDIR, exist_ok=True)

fs.apply_style_diagram()
C_PIPE = PALETTE["ref"]; C_EL = PALETTE["rlc"]; C_ACC = PALETTE["acc"]
C_CAP = PALETTE["grn"]


def wire(ax, pts, color=C_EL, lw=2.2, z=2):
    xs, ys = zip(*pts); ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=z)


def resistor(ax, x0, x1, y, color=C_ACC, lw=2.2, n=6, amp=0.12):
    xs = np.linspace(x0, x1, 2*n+2); ys = np.full_like(xs, y)
    for i in range(1, len(xs)-1):
        ys[i] = y + (amp if i % 2 else -amp)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=3)


def inductor(ax, x0, x1, y, color=C_EL, lw=2.2, n=4):
    r = (x1-x0)/(2*n)
    for i in range(n):
        cx = x0 + r*(2*i+1)
        ax.add_patch(Arc((cx, y), 2*r, 2*r, theta1=0, theta2=180, color=color, lw=lw, zorder=3))
    wire(ax, [(x0-0.02, y), (x0, y)], color, lw); wire(ax, [(x1, y), (x1+0.02, y)], color, lw)


def capacitor(ax, x, y, color=C_CAP, lw=2.4, plate=0.22, gap=0.12):
    ax.plot([x-plate, x+plate], [y+gap, y+gap], color=color, lw=lw, zorder=3)
    ax.plot([x-plate, x+plate], [y-gap, y-gap], color=color, lw=lw, zorder=3)


def ground(ax, x, y, color="0.35"):
    for i, w in enumerate([0.22, 0.14, 0.07]):
        ax.plot([x-w, x+w], [y-0.09*i, y-0.09*i], color=color, lw=2, zorder=3)


def node_dot(ax, x, y, color=C_EL):
    ax.scatter([x], [y], s=45, color=color, zorder=4)


fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.6), gridspec_kw={"width_ratios":[1, 1.15]})

# ================= (a) single pipe segment -> RLC branch =================
ax = axs[0]; ax.set_title("(a) Pipe segment $\\to$ RLC branch", fontsize=15)
# pipe as a tube
ax.add_patch(plt.Rectangle((0.4, 3.3), 3.6, 0.9, fc="#DCE6F0", ec=C_PIPE, lw=1.6, zorder=1))
for xx in np.linspace(0.7, 3.7, 6):
    ax.annotate("", (xx+0.28, 3.75), (xx, 3.75),
                arrowprops=dict(arrowstyle="-|>", color=C_PIPE, lw=1.4))
ax.text(2.2, 4.5, "gas flow  $\\phi$", ha="center", fontsize=13, color=C_PIPE)
ax.text(0.10, 3.02, "$P_a$", ha="left", fontsize=13.5, color=C_PIPE)
ax.text(4.35, 3.02, "$P_b$", ha="right", fontsize=13.5, color=C_PIPE)
ax.text(2.2, 3.05, "friction, inertia, linepack", ha="center",
        fontsize=13, color="0.4", style="italic")
# equivalence arrow
ax.annotate("", (2.2, 2.70), (2.2, 2.95), arrowprops=dict(arrowstyle="-|>", color="0.4", lw=2))
ax.text(2.5, 2.62, "$\\equiv$", fontsize=22, color="0.3", ha="left", va="center")
# RLC branch
yb = 1.7
node_dot(ax, 0.5, yb); ax.text(0.5, yb+0.28, "$P_a$", ha="center", fontsize=13.5, color=C_EL)
node_dot(ax, 3.9, yb); ax.text(3.9, yb+0.28, "$P_b$", ha="center", fontsize=13.5, color=C_EL)
wire(ax, [(0.5, yb), (1.05, yb)])
resistor(ax, 1.05, 1.95, yb); ax.text(1.5, yb+0.42, "$R$", ha="center", fontsize=13.5, color=C_ACC)
wire(ax, [(1.95, yb), (2.15, yb)])
inductor(ax, 2.15, 3.15, yb); ax.text(2.65, yb+0.45, "$L$", ha="center", fontsize=13.5, color=C_EL)
wire(ax, [(3.15, yb), (3.9, yb)])
ax.annotate("", (1.0, yb-0.33), (0.6, yb-0.33), arrowprops=dict(arrowstyle="-|>", color=C_EL, lw=1.6))
ax.text(1.15, yb-0.33, "$\\phi$", fontsize=13, color=C_EL, va="center")
# shunt capacitors (linepack) at each node
for xn in (0.5, 3.9):
    wire(ax, [(xn, yb), (xn, yb-0.6)]); capacitor(ax, xn, yb-0.75); wire(ax, [(xn, yb-0.9), (xn, yb-1.05)])
    ground(ax, xn, yb-1.05)
ax.text(0.5-0.42, yb-0.75, "$C$", ha="right", fontsize=13.5, color=C_CAP)
ax.text(3.9+0.42, yb-0.75, "$C$", ha="left", fontsize=13.5, color=C_CAP)
ax.set_xlim(-0.2, 4.5); ax.set_ylim(0.2, 4.9); ax.axis("off")

# ================= (b) network -> RLC circuit =================
ax = axs[1]; ax.set_title("(b) Meshed network $\\to$ RLC circuit (nodal analysis)", fontsize=15)
pos = {0:(0.4,2.5), 1:(1.8,2.5), 2:(3.3,3.7), 3:(4.8,2.5), 4:(3.3,1.3)}
edges = [(0,1),(1,2),(2,3),(1,4),(4,3)]
# draw RLC branches as R-L between nodes (simplified glyph: small zigzag+coil)
for (a,b) in edges:
    xa,ya=pos[a]; xb,yb=pos[b]
    ax.plot([xa,xb],[ya,yb], color=C_EL, lw=2.0, zorder=1)
    mx,my=0.5*(xa+xb),0.5*(ya+yb)
    ax.scatter([mx],[my], s=520, marker='s', facecolor="white", edgecolor=C_ACC, lw=1.6, zorder=2)
    ax.text(mx,my,"$R,L$",ha="center",va="center",fontsize=13,color=C_ACC,zorder=3)
# nodes with shunt C to ground
for n,(x,y) in pos.items():
    col = C_ACC if n==0 else C_EL
    ax.scatter([x],[y], s=620, color=col, edgecolor="white", lw=1.5, zorder=4)
    ax.text(x,y,f"$P_{n}$",ha="center",va="center",color="white",fontsize=13,fontweight="bold",zorder=5)
    # small capacitor stub downward
    ax.plot([x,x],[y-0.28,y-0.5],color=C_CAP,lw=1.6,zorder=1)
    ax.plot([x-0.13,x+0.13],[y-0.5,y-0.5],color=C_CAP,lw=2,zorder=1)
    ax.plot([x-0.13,x+0.13],[y-0.6,y-0.6],color=C_CAP,lw=2,zorder=1)
ax.text(0.4,1.55,"source\n(compressor)",ha="center",va="top",fontsize=13,color=C_ACC)
# legend-ish annotations
ax.text(2.55,0.42,"nodes = pressures $P_n$ with shunt capacitance $C_n$ (linepack)",
        ha="center",fontsize=13,color="0.35")
ax.text(2.55,0.05,"branches = series $R_k$, $L_k$ carrying flow $\\phi_k$",
        ha="center",fontsize=13,color="0.35")
ax.set_xlim(-0.35,5.35); ax.set_ylim(-0.25,4.5); ax.axis("off")

fig.tight_layout()
fs.save(fig, FIGDIR + "/fig_rlc_circuit.png")
