"""Composition-front propagation: step injection reaches nodes with different delays."""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle as fs
from figstyle import apply_style, legend_below, PALETTE
from gas_properties import mixture_properties
from network import Network, steady_state
from composition import build_composition_model, node_fraction

import os as _os
FIGDIR = _os.environ.get("PAPER_FIGS", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "ms", "figs"))
_os.makedirs(FIGDIR, exist_ok=True)


apply_style()
C=[PALETTE["rlc"],PALETTE["acc"],PALETTE["grn"],PALETTE["ol"],PALETTE["pur"],PALETTE["ref"]]
FIG=FIGDIR

net=Network(); props=mixture_properties(0.05); P0,Q0=steady_state(net,props)
cm=build_composition_model(net,props,P0,Q0)
Ay,By,Wnode=cm['Ay'],cm['By'],cm['Wnode']
def rhs(t,y): return Ay@y+By[:,0]*0.20
y0=np.full(net.n_pipes,0.05)
tt=np.linspace(0,30*3600,400)
sol=solve_ivp(rhs,[0,30*3600],y0,t_eval=tt,method="RK45",rtol=1e-7,atol=1e-9)

fig,ax=fs.panels(1, width=fs.W15, height=2.9)
ax=ax[0]
th=tt/3600
DASH=["-","--","-.",":",(0,(3,1,1,1))]
for j,n in enumerate([1,2,4,3,5]):
    yn=np.array([node_fraction(Wnode,sol.y[:,i],n,0.20,net.source) for i in range(len(tt))])
    ax.plot(th,yn*100,ls=DASH[j],color=C[j],label=f"Node N{n}")
# injection and baseline levels as LEGEND entries, not as text inside the axes,
# where at final print size they collided with the title.
ax.axhline(20,color=PALETTE["mut"],ls=":",
           label="Injected and baseline [5, 20 vol%]")
ax.axhline(5,color=PALETTE["mut"],ls=":")
ax.set_xlabel("Time [h]")
fs.ylabel(ax,"Delivered H$_2$ [vol%]")
ax.set_xlim(0,30)
ax.set_title("Composition-front propagation after a step in injection")
# headroom above the 20 vol% guide, which the curves were running into
ax.set_ylim(top=23.0)
fs.legend_below(ax,ncol=3,y=-0.30)
fs.save(fig, f"{FIG}/fig_composition.png", legend_rows=2)
# report arrival delays (time to reach 90% of step at each node)
for n in [1,2,3,4,5]:
    yn=np.array([node_fraction(Wnode,sol.y[:,i],n,0.20,net.source) for i in range(len(tt))])
    hit=np.flatnonzero(yn>=0.05+0.9*0.15)
    if hit.size:
        print(f"  N{n}: 90% arrival at {th[hit[0]]:.1f} h")
    else:
        # argmax on an all-False array returns 0, which printed "0.0 h" for the
        # two nodes that never reach 90% inside the window: the opposite of the
        # truth. Report the fact instead.
        print(f"  N{n}: 90% not reached within {th[-1]:.0f} h")
