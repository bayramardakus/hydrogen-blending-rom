"""Regenerate the two real-network figures for Case Study 3 on the reconstructed
SciGRID_gas sub-network: (1) geographic map, (2) closed-loop control panels."""
import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os as _os
import figstyle as fs
FIGDIR=_os.environ.get('PAPER_FIGS', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','ms','figs'))
_os.makedirs(FIGDIR, exist_ok=True)
from figstyle import apply_style, legend_below, PALETTE
import realnet, realnet_blend as RB
fs.apply_style_diagram()

def fig_map():
    """Geographic map of the extracted sub-network.

    Redrawn for legibility. The previous version encoded pipe diameter in the
    LINE WIDTH, which at this scale produced thick grey bands that dominated the
    plot, hid the node markers sitting on them, and made the two loop chords -
    the only topological feature worth seeing here - impossible to pick out.
    Diameter is now carried by a light colour ramp on thin lines, the chords are
    drawn distinctly, and every symbol is named in the legend. The title is
    dropped because the caption already carries the same counts.
    """
    import networkx as nx
    from matplotlib.lines import Line2D
    from matplotlib.patches import ConnectionPatch
    d = json.load(open("scigrid_subnet.json"))
    nodes = {n['i']: n for n in d['nodes']}
    N = d['n_nodes']
    lon = np.array([nodes[i]['lon'] for i in range(N)])
    lat = np.array([nodes[i]['lat'] for i in range(N)])
    sup, hub = d['supply'], d['western_hub']

    # the two loop-closing chords; the remaining edges form a spanning tree
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for k, e in enumerate(d['edges']):
        G.add_edge(e['a'], e['b'], k=k, length=e['length_km'])
    T = nx.minimum_spanning_tree(G, weight='length')
    chords = [k for k, e in enumerate(d['edges'])
              if not T.has_edge(e['a'], e['b'])]

    dias = np.array([e['dia_m'] for e in d['edges']])
    dmin, dmax = dias.min(), dias.max()
    cmap = plt.get_cmap('Blues')

    # ---- layout -------------------------------------------------------------
    # Two corners of this network are unreadable at column width:
    # the injection corner, where the supply, the first delivery node and both
    # ends of a loop-closing chord lie within a tenth of a degree, and the
    # southern corner, which carries a third of the delivery nodes in a
    # twentieth of the mapped area. Each is repeated at about four times the
    # scale in a panel OUTSIDE the map frame, with the window outlined on the
    # map and joined to its panel. The figure is set at the full text width, so
    # the axes are placed explicitly rather than by tight_layout, which cannot
    # know about the legend strip reserved at the foot.
    FS_TICK, FS_AX, FS_TTL, FS_NODE, FS_LEG = 13.0, 13.5, 13.5, 13.0, 13.0
    fig = plt.figure(figsize=(7.0, 5.65))
    ax = fig.add_axes([0.128, 0.295, 0.500, 0.680])
    axZ1 = fig.add_axes([0.735, 0.672, 0.223, 0.268])
    axZ2 = fig.add_axes([0.735, 0.288, 0.223, 0.268])

    def draw(tgt, k_node=1.0, lw=1.0):
        for k, e in enumerate(d['edges']):
            a, b = e['a'], e['b']
            if k in chords:
                tgt.plot([lon[a], lon[b]], [lat[a], lat[b]], '--',
                         color=PALETTE["acc"], lw=1.7 * lw, zorder=2,
                         dash_capstyle='round')
            else:
                frac = (e['dia_m'] - dmin) / max(dmax - dmin, 1e-9)
                tgt.plot([lon[a], lon[b]], [lat[a], lat[b]], '-',
                         color=cmap(0.38 + 0.5 * frac), lw=1.5 * lw, zorder=1,
                         solid_capstyle='round')
        tgt.scatter(lon, lat, s=30 * k_node, c='white', zorder=3,
                    edgecolor=PALETTE["ref"], linewidths=0.9)
        tgt.scatter(lon[sup], lat[sup], s=250 * k_node, marker='*',
                    c=PALETTE["acc"], zorder=5, edgecolor='white', lw=1.0)
        tgt.scatter(lon[hub], lat[hub], s=110 * k_node, marker='D',
                    c=PALETTE["grn"], zorder=5, edgecolor='white', lw=1.0)

    draw(ax)

    # Node names: the supply above and LEFT of its marker, the hub above its
    # own, both clear of every pipe and of the connectors to the zoom panels.
    _bb = dict(fc="white", ec="none", alpha=0.85, pad=1.2)
    ax.annotate(nodes[sup]['scigrid_id'], (lon[sup], lat[sup]),
                textcoords="offset points", xytext=(-11, -13), fontsize=FS_NODE,
                color="0.25", ha="right", va="top", bbox=_bb)
    ax.annotate(nodes[hub]['scigrid_id'], (lon[hub], lat[hub]),
                textcoords="offset points", xytext=(11, 13), fontsize=FS_NODE,
                color="0.25", ha="left", va="bottom", bbox=_bb)

    ax.set_xlabel("Longitude [\u00b0E]", fontsize=FS_AX)
    ax.set_ylabel("Latitude [\u00b0N]", fontsize=FS_AX)
    ax.tick_params(labelsize=FS_TICK)
    ax.margins(x=0.10, y=0.09)
    ax.grid(alpha=.3)

    # ---- the two magnified corners, outside the map frame -------------------
    zooms = [(axZ1, 10.94, 11.36, 52.62, 52.92, [11.0, 11.2], [52.7, 52.9],
              "Injection corner"),
             (axZ2, 11.42, 12.10, 51.71, 52.11, [11.5, 11.8, 12.1], [51.8, 52.0],
              "Southern corner")]
    for azi, x0, x1, y0, y1, xt, yt, ttl in zooms:
        draw(azi, k_node=1.45, lw=1.30)
        azi.set_xlim(x0, x1); azi.set_ylim(y0, y1)
        azi.set_xticks(xt); azi.set_yticks(yt)
        azi.tick_params(labelsize=FS_TICK, length=2.5, pad=2)
        azi.grid(alpha=0.25)
        azi.set_title(ttl, fontsize=FS_TTL, pad=5)
        for sp in azi.spines.values():
            sp.set_linewidth(0.8); sp.set_color('0.35')
        # One leader line per panel, not the pair matplotlib draws by default:
        # two lines from a small window cross the map and read as pipes.
        _rect, _conn = ax.indicate_inset(
            (x0, y0, x1 - x0, y1 - y0), inset_ax=azi,
            edgecolor='0.35', lw=0.8, alpha=0.95)
        for c in _conn:
            c.set_visible(False)
        fig.add_artist(ConnectionPatch(
            xyA=(x1, 0.5 * (y0 + y1)), coordsA=ax.transData,
            xyB=(0.0, 0.5), coordsB=azi.transAxes,
            color='0.35', lw=0.8, alpha=0.95, zorder=0))

    # ---- legend, in its own strip at the foot of the figure ------------------
    hs = [Line2D([], [], marker='*', ls='none', ms=13, mfc=PALETTE["acc"],
                 mec='white', label="Eastern supply, injection 1"),
          Line2D([], [], marker='D', ls='none', ms=7, mfc=PALETTE["grn"],
                 mec='white', label="Western hub, injection 2"),
          Line2D([], [], marker='o', ls='none', ms=6, mfc='white',
                 mec=PALETTE["ref"], label="Delivery node"),
          Line2D([], [], color=cmap(0.85), lw=2.0,
                 label="Pipe, diameter %.1f\u2013%.1f m" % (dmin, dmax)),
          Line2D([], [], color=PALETTE["acc"], lw=1.8, ls='--',
                 label="Loop-closing chord")]
    fig.legend(handles=hs, labels=[h.get_label() for h in hs],
               loc='lower center', bbox_to_anchor=(0.5, 0.008), ncol=2,
               frameon=False, fontsize=FS_LEG, handlelength=2.4,
               columnspacing=1.6, handletextpad=0.7)

    # Saved at exactly the canvas size: the layout above already reserves the
    # legend strip, and a tight box would crop the figure past the text width.
    fig.savefig(_os.path.join(FIGDIR, "fig_realnet_map.png"))
    plt.close(fig)
    print("saved fig_realnet_map.png (+ .pdf)")


def fig_mpc():
    bl=np.load("realnet_blend_results.npy",allow_pickle=True).item()
    pr=np.load("realnet_pressure_results.npy",allow_pickle=True).item()
    mpc,rea=bl['mpc'],bl['rea']
    fig,axs=plt.subplots(1,3,figsize=(fs.W3, fs.H3))
    # (a) two-point injection schedule
    ax=axs[0]
    ax.step(mpc['t'],mpc['aE']*100,where='post',color="#C7CED6",lw=1.4,label="available")
    ax.step(mpc['t'],mpc['uE']*100,where='post',color=PALETTE["acc"],lw=2.0,label="east injection")
    ax.step(mpc['t'],mpc['uW']*100,where='post',color=PALETTE["grn"],lw=2.0,label="west injection")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Injected H2 (vol%)")
    ax.set_title("(a) Two-point injection schedule"); ax.grid(alpha=.3); ax.set_xlim(0,42)
    legend_below(ax,ncol=3,y=-0.30)
    # (b) worst delivered blend
    ax=axs[1]
    ax.plot(mpc['t'],mpc['ydel'].max(axis=1)*100,color=PALETTE["rlc"],lw=2.2,label="MPC (two-point)")
    ax.plot(rea['t'],rea['ydel'].max(axis=1)*100,color=PALETTE["ol"],lw=2.0,ls='--',
            label="full-injection baseline")
    ax.axhline(20,color=PALETTE["acc"],lw=1.4,ls=':',label="20 vol% limit")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Worst delivered H2 (vol%)")
    ax.set_title("(b) Delivered blend at worst node"); ax.grid(alpha=.3); ax.set_xlim(0,42)
    legend_below(ax,ncol=2,y=-0.30)
    # (c) pressure regulation
    ax=axs[2]
    ax.plot(pr['ol']['t'],pr['ol']['pmin'],color=PALETTE["ol"],lw=2.0,ls='--',label="uncontrolled")
    ax.plot(pr['cl']['t'],pr['cl']['pmin'],color=PALETTE["rlc"],lw=2.2,label="pressure MPC")
    ax.axhline(60,color=PALETTE["acc"],lw=1.4,ls=':',label="60 bar floor")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Min. node pressure (bar)")
    ax.set_title("(c) Pressure regulation under surge"); ax.grid(alpha=.3)
    legend_below(ax,ncol=3,y=-0.30)
    fig.savefig(_os.path.join(FIGDIR,"fig_realnet_mpc.png")); plt.close(fig)
    print("wrote fig_realnet_mpc.png")

if __name__=="__main__":
    import os; os.makedirs("figs",exist_ok=True)
    fig_map()
