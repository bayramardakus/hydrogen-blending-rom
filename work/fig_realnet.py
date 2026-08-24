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

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    dias = np.array([e['dia_m'] for e in d['edges']])
    dmin, dmax = dias.min(), dias.max()
    cmap = plt.get_cmap('Blues')

    for k, e in enumerate(d['edges']):
        a, b = e['a'], e['b']
        if k in chords:
            ax.plot([lon[a], lon[b]], [lat[a], lat[b]], '--',
                    color=PALETTE["acc"], lw=1.7, zorder=2,
                    dash_capstyle='round')
        else:
            frac = (e['dia_m'] - dmin) / max(dmax - dmin, 1e-9)
            ax.plot([lon[a], lon[b]], [lat[a], lat[b]], '-',
                    color=cmap(0.38 + 0.5 * frac), lw=1.5, zorder=1,
                    solid_capstyle='round')

    ax.scatter(lon, lat, s=30, c='white', zorder=3,
               edgecolor=PALETTE["ref"], linewidths=0.9)
    ax.scatter(lon[sup], lat[sup], s=250, marker='*', c=PALETTE["acc"],
               zorder=5, edgecolor='white', lw=1.0)
    ax.scatter(lon[hub], lat[hub], s=110, marker='D', c=PALETTE["grn"],
               zorder=5, edgecolor='white', lw=1.0)
    ax.annotate(nodes[sup]['scigrid_id'], (lon[sup], lat[sup]),
                textcoords="offset points", xytext=(12, -18), fontsize=11.5,
                color="0.25")
    ax.annotate(nodes[hub]['scigrid_id'], (lon[hub], lat[hub]),
                textcoords="offset points", xytext=(-14, 22), fontsize=11.5,
                color="0.25", ha="center")

    hs = [Line2D([], [], marker='*', ls='none', ms=14, mfc=PALETTE["acc"],
                 mec='white', label="Eastern supply, injection 1"),
          Line2D([], [], marker='D', ls='none', ms=7, mfc=PALETTE["grn"],
                 mec='white', label="Western hub, injection 2"),
          Line2D([], [], marker='o', ls='none', ms=6, mfc='white',
                 mec=PALETTE["ref"], label="Delivery node"),
          Line2D([], [], color=cmap(0.85), lw=2.0,
                 label="Pipe, shade = diameter [%.1f\u2013%.1f m]" % (dmin, dmax)),
          Line2D([], [], color=PALETTE["acc"], lw=1.8, ls='--',
                 label="Loop-closing chord")]
    ax.set_xlabel("Longitude [\u00b0E]")
    ax.set_ylabel("Latitude [\u00b0N]")
    ax.margins(0.10)
    ax.grid(alpha=.3)
    legend_below(ax, ncol=2, y=-0.14, fontsize=11.5, handles=hs,
                 labels=[h.get_label() for h in hs])
    fs.save(fig, _os.path.join(FIGDIR, "fig_realnet_map.png"), legend_rows=3)


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
