"""
make_figures.py
---------------
Publication-quality figures with a consistent house style: legends placed
below the axes, no text/graphic overlap.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle as fs
from figstyle import apply_style, legend_below, PALETTE
from gas_properties import mixture_properties
from network import Network, steady_state
from experiment_validation import run_case

import os as _os
FIGDIR = _os.environ.get("PAPER_FIGS", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "ms", "figs"))
_os.makedirs(FIGDIR, exist_ok=True)


apply_style()
C_REF = PALETTE["ref"]; C_RLC = PALETTE["rlc"]; C_ACC = PALETTE["acc"]
FIG = FIGDIR


# ---------------------------------------------------------------- Fig: props
def fig_properties():
    xs = np.linspace(0, 0.30, 61)
    a = []; hhv = []; rho = []; wob = []
    for x in xs:
        p = mixture_properties(x)
        a.append(p['a']); hhv.append(p['hhv_ratio']*100)
        rho.append(p['rho']); wob.append(p['wobbe_ratio']*100)
    fig, axs = plt.subplots(1, 2, figsize=(fs.W2, fs.H2))
    ax = axs[0]
    l1, = ax.plot(xs*100, a, color=C_RLC, lw=2, label="Wave speed $a$")
    ax.set_xlabel("Hydrogen fraction [vol%]")
    ax.set_ylabel("Wave speed $a$ [m/s]", color=C_RLC)
    ax.tick_params(axis='y', colors=C_RLC)
    ax.set_title("(a) Wave speed and density at 60 bar"); ax.grid(alpha=.3)
    ax2 = ax.twinx()
    l2, = ax2.plot(xs*100, rho, color=C_ACC, lw=2, ls="--", label="Density $\\rho$")
    ax2.set_ylabel("Density $\\rho$ [kg/m$^3$]", color=C_ACC)
    ax2.tick_params(axis='y', colors=C_ACC)
    legend_below(ax, ncol=2, handles=[l1, l2], labels=[h.get_label() for h in (l1, l2)])

    ax = axs[1]
    ax.plot(xs*100, hhv, color=C_RLC, lw=2, label="Volumetric HHV")
    ax.plot(xs*100, wob, color=C_ACC, lw=2, ls="--", label="Wobbe index")
    # The two blend guides are a LEGEND ENTRY, not a bare line: an unexplained
    # vertical rule in a figure is a defect, however obvious it seems to the
    # author.
    for i, xv in enumerate((5, 25)):
        ax.axvline(xv, color="0.7", lw=.8, ls=":",
                   label="5 and 25 vol% cases" if i == 0 else None)
    ax.set_xlabel("Hydrogen fraction [vol%]"); ax.set_ylabel("Relative to pure NG [%]")
    ax.set_title("(b) Energy content and interchangeability"); ax.grid(alpha=.3)
    legend_below(ax, ncol=3)
    fs.save(fig, f"{FIG}/fig_properties.png", legend_rows=1)


# ------------------------------------------------------ Fig: validation traj
def fig_validation(res5, res25):
    fig, axs = plt.subplots(1, 2, figsize=(fs.W2, fs.H2))
    for ax, r, lab in zip(axs, [res5, res25], ["5", "25"]):
        t = r['t'] / 3600.0
        for n in range(1, r['net'].n_nodes):
            ax.plot(t, r['P_ref'][n]/1e5, color=C_REF, lw=2.4, alpha=.55,
                    label="1-D reference" if n == 1 else None)
            ax.plot(t, r['P_rlc'][n]/1e5, color=C_RLC, lw=1.1, ls="--",
                    label="RLC reduced" if n == 1 else None)
        ax.set_xlabel("Time [h]"); ax.set_ylabel("Node pressure [bar]")
        ax.set_title(f"({'a' if lab=='5' else 'b'}) {lab}% H$_2$ blend  "
                     f"(MAPE = {r['mape']:.2f}%)")
        ax.grid(alpha=.3)
        legend_below(ax, ncol=2)
    fs.save(fig, f"{FIG}/fig_validation_pressure.png", legend_rows=1)


# ------------------------------------------------------ Fig: timing + error
def fig_timing(res5, res25):
    """Panel (a) is drawn from `sweep.npy` - the SAME timing run that populates
    Table 4 - not from the res5/res25 trajectories.

    Wall-clock timings vary by 10-20 % between runs on a shared machine. Drawing
    the figure from one run and the table from another made the two disagree by
    up to 22 % on the reported speed-up, which is precisely the kind of internal
    inconsistency a reader checks first. Panel (b) still uses the trajectories,
    which are deterministic.
    """
    fig, axs = plt.subplots(1, 2, figsize=(fs.W2, fs.H2))
    ax = axs[0]
    labels = ["5% H$_2$", "25% H$_2$"]
    try:
        sw = {int(round(r['h2'])): r
              for r in np.load("sweep.npy", allow_pickle=True)}
        src = [sw[5], sw[25]]
        tref = [r['t_ref']*1000 for r in src]
        trlc = [r['t_rlc']*1000 for r in src]
        n_ref, n_rlc = src[0]['n_ref'], src[0]['n_rlc']
    except (FileNotFoundError, KeyError):
        tref = [res5['t_ref']*1000, res25['t_ref']*1000]
        trlc = [res5['t_rlc']*1000, res25['t_rlc']*1000]
        n_ref, n_rlc = res5['n_ref'], res5['n_rlc']
    x = np.arange(2); w = 0.36
    ax.bar(x-w/2, tref, w, color=C_REF, label=f"1-D reference ({n_ref} states)")
    ax.bar(x+w/2, trlc, w, color=C_RLC, label=f"RLC reduced ({n_rlc} states)")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(top=max(tref)*6)
    ax.set_ylabel("Solve time [ms], log")
    ax.set_title("(a) Computation time")
    for i in range(2):
        sp = tref[i]/trlc[i]
        ax.text(i, tref[i]*1.9, f"{sp:.0f}$\\times$ faster", ha="center",
                fontsize=8, color=C_ACC, fontweight="bold")
    ax.grid(alpha=.3, axis="y")
    legend_below(ax, ncol=2)
    ax = axs[1]
    for r, c, lab in zip([res5, res25], [C_RLC, C_ACC], ["5% H$_2$", "25% H$_2$"]):
        t = r['t']/3600.0
        emax = np.max(np.abs(r['P_rlc'][1:]-r['P_ref'][1:])/1e5, axis=0)
        ax.plot(t, emax, color=c, lw=1.8, label=lab)
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Max. pressure error [bar]")
    ax.set_title("(b) Reduction error vs. time"); ax.grid(alpha=.3)
    legend_below(ax, ncol=2)
    fs.save(fig, f"{FIG}/fig_timing.png", legend_rows=1)


# ------------------------------------------------------ Fig: network schematic
def fig_network():
    """Didactic six-node schematic, redrawn for legibility.

    The previous version had three defects, all of the same kind: artwork drawn
    at one scale and type set at another.

     * Pipe diameter was encoded in the LINE WIDTH (lw = 1 + 3D, i.e. 3.4-4 pt),
       which produced thick grey bands that sat on top of the node markers.
       Diameter is now a light colour ramp on thin lines, as on the real-network
       map (Fig. 3), so the two figures read the same way.
     * The pipe labels were placed on the "upper" side of every edge by a rule
       that has no notion of what is already there, so the label of the vertical
       pipe N3-N5 landed on N3's withdrawal annotation. Each label side is now
       chosen explicitly, and each annotation is anchored on a side that no edge
       label occupies.
     * Title and labels were hard-coded at 13/10.5/10 pt on a 140 mm canvas, so
       the title overran the canvas on both sides and was clipped. The title is
       dropped - the caption already carries the counts - and the figure is set
       in the shared schematic style.
    """
    import networkx as nx
    from matplotlib.lines import Line2D
    fs.apply_style_diagram()
    net = Network()
    pos = {0: (0.0, 2.0), 1: (2.4, 2.0), 2: (5.0, 3.6), 3: (7.6, 2.0),
           4: (5.0, 0.4), 5: (7.6, -1.2)}

    # The two loop-closing chords: the remaining five pipes form a spanning tree.
    G = nx.Graph()
    G.add_nodes_from(pos)
    for k, (a, b, L, D) in enumerate(net.pipes):
        G.add_edge(a, b, k=k, length=L)
    T = nx.minimum_spanning_tree(G, weight="length")
    chords = [k for k, (a, b, L, D) in enumerate(net.pipes)
              if not T.has_edge(a, b)]

    # Side of each edge on which to hang its label: +1 = left of a->b, -1 = right.
    SIDE = {0: +1, 1: +1, 2: +1, 3: -1, 4: -1, 5: -1, 6: -1}
    # Where each node's withdrawal annotation goes, clear of every edge label.
    ANN = {1: (-0.05, -0.40, "center", "top"), 2: (0.0, 0.42, "center", "bottom"),
           3: (0.40, 0.10, "left", "bottom"), 4: (-0.42, -0.06, "right", "center"),
           5: (0.0, -0.40, "center", "top")}

    dias = np.array([D for (_, _, _, D) in net.pipes])
    dmin, dmax = dias.min(), dias.max()
    cmap = plt.get_cmap("Blues")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for k, (a, b, L, D) in enumerate(net.pipes):
        xa, ya = pos[a]; xb, yb = pos[b]
        if k in chords:
            ax.plot([xa, xb], [ya, yb], "--", color=C_ACC, lw=1.7, zorder=2,
                    dash_capstyle="round")
        else:
            frac = (D - dmin) / max(dmax - dmin, 1e-9)
            ax.plot([xa, xb], [ya, yb], "-", color=cmap(0.38 + 0.5 * frac),
                    lw=1.6, zorder=1, solid_capstyle="round")
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        dx, dy = xb - xa, yb - ya
        L2 = np.hypot(dx, dy)
        ang = np.degrees(np.arctan2(dy, dx))
        if ang > 90:
            ang -= 180
        elif ang < -90:
            ang += 180
        ox, oy = SIDE[k] * -dy / L2 * 0.30, SIDE[k] * dx / L2 * 0.30
        ax.text(mx + ox, my + oy, f"P{k}: {L/1000:.0f} km, {D:.1f} m",
                fontsize=9.5, ha="center", va="center", color="0.30",
                rotation=ang, rotation_mode="anchor", zorder=5)

    for n, (x, y) in pos.items():
        col = C_ACC if n == net.source else C_RLC
        ax.scatter([x], [y], s=360, color=col, zorder=3, edgecolor="white",
                   linewidth=1.2)
        ax.text(x, y, f"N{n}", ha="center", va="center", color="white",
                fontweight="bold", fontsize=10, zorder=4)
        if net.demand_nom[n] > 0:
            ox, oy, ha, va = ANN[n]
            ax.text(x + ox, y + oy, f"{net.demand_nom[n]:.0f} kg/s", ha=ha,
                    va=va, fontsize=9.5, color="0.30", zorder=4)
    sx, sy = pos[net.source]
    ax.text(sx, sy + 0.55, "Supply, 66 bar", ha="center", va="bottom",
            fontsize=9.5, color=C_ACC, fontweight="bold", zorder=4)

    hs = [Line2D([], [], marker="o", ls="none", ms=9, mfc=C_ACC, mec="white",
                 label="Supply node"),
          Line2D([], [], marker="o", ls="none", ms=9, mfc=C_RLC, mec="white",
                 label="Withdrawal node"),
          Line2D([], [], color=cmap(0.85), lw=2.0,
                 label="Pipe, shade = diameter [%.1f\u2013%.1f m]" % (dmin, dmax)),
          Line2D([], [], color=C_ACC, lw=1.8, ls="--",
                 label="Loop-closing chord")]
    ax.set_xlim(-1.0, 9.4)
    ax.set_ylim(-1.75, 4.10)
    ax.axis("off")
    fs.legend_below(ax, ncol=2, y=-0.02, fontsize=10.0, handles=hs,
                    labels=[h.get_label() for h in hs])
    fs.save(fig, f"{FIG}/fig_network.png", legend_rows=2)
    apply_style()


if __name__ == "__main__":
    print("running cases...")
    res5 = run_case(0.05, M_ref=20)
    res25 = run_case(0.25, M_ref=20)
    np.save("res5.npy", res5, allow_pickle=True)
    np.save("res25.npy", res25, allow_pickle=True)
    fig_properties()
    fig_validation(res5, res25)
    fig_timing(res5, res25)
    fig_network()
    print("done")
