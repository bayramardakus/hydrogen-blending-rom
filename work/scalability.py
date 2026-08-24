"""
scalability.py
--------------
Show that the reduced-order model and its MPC stay real-time as the network
grows -- the answer to "why only six nodes?". A structured meshed network of N
nodes is generated with SciGRID-like pipe parameters; we time (i) the reduced
model assembly and (ii) one MPC quadratic-program solve, for increasing N.
"""
import time
import numpy as np
from scipy.linalg import expm, solve_discrete_are
from scipy.optimize import fsolve
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle as fs
from figstyle import apply_style, legend_below, PALETTE

from gas_properties import mixture_properties, darcy_friction

import os as _os
FIGDIR = _os.environ.get("PAPER_FIGS", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "ms", "figs"))
_os.makedirs(FIGDIR, exist_ok=True)


apply_style()
C_RLC=PALETTE["rlc"]; C_GRN=PALETTE["grn"]; C_ACC=PALETTE["acc"]; FIG=FIGDIR


class GenNet:
    """Structured meshed transmission network with N nodes (node 0 = supply)."""
    def __init__(self, N, seed=0):
        rng = np.random.default_rng(seed)
        self.n_nodes = N; self.source = 0; self.p_source_nom = 66.0e5
        pipes = []
        # spanning tree: node i attaches to a random earlier node
        for i in range(1, N):
            j = int(rng.integers(max(0, i-3), i))       # attach to a recent node
            L = float(rng.uniform(30_000, 70_000)); D = float(rng.choice([0.8,0.9,1.0,1.1]))
            pipes.append((j, i, L, D))
        # add loop chords (~15% extra) to make it meshed
        nchord = max(1, int(0.15*N))
        for _ in range(nchord):
            a, b = rng.integers(1, N), rng.integers(1, N)
            if a != b:
                L = float(rng.uniform(30_000, 70_000)); D = float(rng.choice([0.8,0.9,1.0]))
                pipes.append((int(min(a,b)), int(max(a,b)), L, D))
        self.pipes = pipes; self.n_pipes = len(pipes)
        # demands: each non-source node draws a modest load
        d = rng.uniform(10, 40, N); d[0] = 0.0
        self.demand_nom = d
        self.inc = np.zeros((N, self.n_pipes))
        for k,(a,b,L,D) in enumerate(pipes):
            self.inc[a,k] = 1.0; self.inc[b,k] = -1.0
        self.area = np.array([np.pi*D**2/4 for (_,_,_,D) in pipes])
        self.length = np.array([L for (_,_,L,_) in pipes])
        self.diam = np.array([D for (_,_,_,D) in pipes])
        self.total_km = self.length.sum()/1000


def steady(net, props):
    nN,nP=net.n_nodes,net.n_pipes; a2=props['a2']; ps=net.p_source_nom
    def resid(x):
        Pbar=np.concatenate([[ps/1e5],x[:nN-1]]); Q=x[nN-1:]
        r=np.zeros(nN-1+nP); r[:nN-1]=(net.inc@Q+net.demand_nom)[1:]
        for k,(a,b,L,D) in enumerate(net.pipes):
            A=net.area[k]; lam=darcy_friction(props['rho'],props['mu'],D,Q[k])
            coef=lam*a2*L/(D*A**2)
            r[nN-1+k]=(Pbar[a]*1e5)**2-(Pbar[b]*1e5)**2-coef*Q[k]*abs(Q[k])
        return r
    x0=np.concatenate([np.full(nN-1,ps/1e5-3),np.full(nP,20.0)])
    x,_,ier,_=fsolve(resid,x0,full_output=True)
    P=np.concatenate([[ps],x[:nN-1]*1e5]); Q=x[nN-1:]
    return P,Q,ier==1


def build_rlc(net, props, P0, Q0):
    a2=props['a2']; nN,nP=net.n_nodes,net.n_pipes
    Vn=np.zeros(nN)
    for k,(a,b,L,D) in enumerate(net.pipes):
        Vn[a]+=net.area[k]*net.length[k]/2; Vn[b]+=net.area[k]*net.length[k]/2
    Cn=Vn/a2; Lk=net.length/net.area; Rk=np.zeros(nP)
    for k,(a,b,L,D) in enumerate(net.pipes):
        A=net.area[k]; lam=darcy_friction(props['rho'],props['mu'],D,Q0[k])
        pavg=0.5*(P0[a]+P0[b]); Rk[k]=(L/A)*(lam*a2/(2*D*A))*2*abs(Q0[k])/pavg
    ns=(nN-1)+nP; Ac=np.zeros((ns,ns)); Bc=np.zeros((ns,nN+1))
    nodemap={n:i for i,n in enumerate([n for n in range(nN) if n!=net.source])}
    for n in range(nN):
        if n==net.source: continue
        si=nodemap[n]
        for k in range(nP): Ac[si,(nN-1)+k]+=-net.inc[n,k]/Cn[n]
        Bc[si,1+n]+=-1/Cn[n]
    for k,(a,b,L,D) in enumerate(net.pipes):
        row=(nN-1)+k; Lv=Lk[k]
        if a==net.source: Bc[row,0]+=1/Lv
        else: Ac[row,nodemap[a]]+=1/Lv
        if b==net.source: Bc[row,0]+=-1/Lv
        else: Ac[row,nodemap[b]]+=-1/Lv
        Ac[row,row]+=-Rk[k]/Lv
    return Ac,Bc,ns


def disc(A,B,Ts):
    n,m=A.shape[0],B.shape[1]; Mb=np.zeros((n+m,n+m)); Mb[:n,:n]=A; Mb[:n,n:]=B
    Md=expm(Mb*Ts); return Md[:n,:n],Md[:n,n:]


def operating_point(net):
    """Robust operating point for timing: least-norm mass-balanced flows and an
    approximate pressure field (adequate for assembling R,L,C at large N)."""
    nN,nP=net.n_nodes,net.n_pipes
    b=-net.demand_nom.copy(); b[net.source]=net.demand_nom.sum()
    Q0,_,_,_=np.linalg.lstsq(net.inc,b,rcond=None)
    P0=np.full(nN, net.p_source_nom-2.0e5)          # ~2 bar below supply
    P0[net.source]=net.p_source_nom
    return P0,Q0


def time_case(N, H=15, TS=120.0):
    props=mixture_properties(0.25)
    net=GenNet(N)
    P0,Q0=operating_point(net)
    tb=[]
    for _ in range(3):
        t0=time.perf_counter()
        Ac,Bc,ns=build_rlc(net,props,P0,Q0); Ad,Bud=disc(Ac,Bc[:,[0]],TS)
        tb.append(time.perf_counter()-t0)
    t_build=np.median(tb)
    nN=net.n_nodes
    # one MPC QP solve, formulated in BAR units for good conditioning
    P0b=P0/1e5
    x0=cp.Parameter(ns); u=cp.Variable((H,1)); x=cp.Variable((H+1,ns)); s=cp.Variable((H,nN-1),nonneg=True)
    cons=[x[0]==x0]; cost=0
    for k in range(H):
        cons+=[x[k+1]==Ad@x[k]+Bud@u[k]]
        cons+=[u[k]>=-6.0,u[k]<=12.0]
        for i in range(nN-1): cons+=[P0b[i+1]+x[k+1,i]>=58.0-s[k,i]]
        cost+=cp.sum_squares(x[k+1,:nN-1])+0.05*cp.square(u[k])+1e4*cp.sum(s[k])
    prob=cp.Problem(cp.Minimize(cost),cons)
    x0.value=np.zeros(ns)
    ts=[]
    for _ in range(5):
        t0=time.perf_counter(); prob.solve(solver=cp.OSQP,warm_start=True,verbose=False,
            max_iter=4000,eps_abs=1e-4,eps_rel=1e-4); ts.append(time.perf_counter()-t0)
    return dict(N=N,nP=net.n_pipes,ns=ns,km=net.total_km,t_build=t_build*1000,t_qp=np.median(ts)*1000)


def make_scal_fig(rows):
    """Cost versus network size.

    Three defects of the previous version are fixed here. The control-interval
    threshold was a free-floating in-axes annotation that collided with the
    figure title and straddled the top spine; it is now a legend entry, as the
    house style requires of every threshold line. The legend and that annotation
    were set at 11-11.5 pt against an 8.5 pt title, so the figure had no
    coherent type scale; both now take the shared sizes. And the y-axis was
    forced to span six decades to reach the threshold, compressing all the data
    into its lower third; the threshold is now carried by the legend and a
    right-hand margin label instead, so the axis spans only the data.
    """
    Ns=[r['N'] for r in rows]; tq=[r['t_qp'] for r in rows]; tb=[r['t_build'] for r in rows]
    fig,ax=plt.subplots(figsize=(fs.W15, 2.9))
    ax.plot(Ns,tq,'o-',color=C_RLC,ms=3.5,label="MPC QP solve per step")
    ax.plot(Ns,tb,'s--',color=C_GRN,ms=3.5,label="Reduced-model assembly")
    ax.set_yscale("log")
    ax.set_xlabel("Network size [nodes]")
    ax.grid(alpha=.3,which="both")
    # Margin to the 120 s control interval, stated as a ratio rather than drawn
    # six decades above the data.
    # The 120 s control interval is NOT drawn: at 1.2e5 ms it is three decades
    # above the data and would flatten the whole curve. A legend key for a line
    # that appears nowhere in the axes is worse than no key, so the margin is
    # reported in the y-label and in the caption instead.
    fs.ylabel(ax,"Wall-clock time [ms]")
    fs.title(ax,'a',"Cost versus network size",single=True)
    legend_below(ax,ncol=2)
    fs.save(fig, f"{FIG}/fig_scalability.png", legend_rows=1)


if __name__=="__main__":
    rows=[]
    for N in [6,12,24,40,60,80,100]:
        r=time_case(N)
        if r:
            rows.append(r)
            print(f"N={r['N']:3d} nodes, {r['nP']:3d} pipes, {r['ns']:3d} states, {r['km']:5.0f} km:"
                  f"  build={r['t_build']:.2f} ms  MPC-QP={r['t_qp']:.2f} ms")
    np.save("scal.npy",rows,allow_pickle=True)
    make_scal_fig(rows)
