"""
mpc_analysis.py
---------------
Depth analyses of the pressure-regulation MPC (Section: control design):
  (1) prediction-horizon sweep  -> justifies the chosen H
  (2) control-weight (R) sensitivity -> pressure-tracking vs compressor-effort trade-off
  (3) Kalman observer convergence from a wrong initial estimate
  (4) MPC vs PID vs uncontrolled -> constraint-aware disturbance rejection

For these controller studies the fast reduced model is used as the plant so that
hundreds of closed-loop runs are affordable; fidelity of that model vs. the 1-D
reference is established separately.
"""
import time
import numpy as np
from scipy.linalg import expm, solve_discrete_are
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle as fs
from figstyle import apply_style, legend_below, PALETTE

from gas_properties import mixture_properties
from network import Network, steady_state, build_rlc

import os as _os
FIGDIR = _os.environ.get("PAPER_FIGS", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "ms", "figs"))
_os.makedirs(FIGDIR, exist_ok=True)


apply_style()
C_REF=PALETTE["ref"]; C_RLC=PALETTE["rlc"]; C_ACC=PALETTE["acc"]; C_OL=PALETTE["ol"]; C_GRN=PALETTE["grn"]
FIG=FIGDIR

TS=120.0; T_END=7200.0
P_MIN=58.0e5; P_SRC_MIN=60.0e5; P_SRC_MAX=78.0e5

def demand(t, net):
    d=net.demand_nom.astype(float).copy()
    d[3]+=55.0*np.clip((t-1800)/300,0,1)
    d[5]+=30.0*np.clip((t-4200)/300,0,1)
    return d

def disc(A,B,Ts):
    n,m=A.shape[0],B.shape[1]; Mb=np.zeros((n+m,n+m)); Mb[:n,:n]=A; Mb[:n,n:]=B
    Md=expm(Mb*Ts); return Md[:n,:n],Md[:n,n:]

def setup():
    net=Network(); props=mixture_properties(0.25); P0,Q0=steady_state(net,props)
    A_c,B_c,C_out,info=build_rlc(net,props,P0,Q0); ns=info['ns']; nN=net.n_nodes
    Bu=B_c[:,[0]]; Bd=B_c[:,1:]
    Ad,Bud=disc(A_c,Bu,TS); _,Bdd=disc(A_c,Bd,TS); Cd=C_out
    Qk=np.eye(ns)*1e-2; Rk=np.eye(nN-1)*1e-3
    Pk=solve_discrete_are(Ad.T,Cd.T,Qk,Rk); Lobs=Pk@Cd.T@np.linalg.inv(Cd@Pk@Cd.T+Rk)
    return net,props,P0,Q0,Ad,Bud,Bdd,Cd,Lobs,ns,nN,info

def build_mpc(Ad,Bud,Bdd,P0,ns,nN,H,Rw):
    x0=cp.Parameter(ns); dseq=cp.Parameter((H,nN))
    u=cp.Variable((H,1)); x=cp.Variable((H+1,ns)); s=cp.Variable((H,nN-1),nonneg=True)
    Qx=np.zeros((ns,ns))
    for i in range(nN-1): Qx[i,i]=1.0
    Qf=solve_discrete_are(Ad,Bud,Qx,np.array([[Rw]])); SC=1e5; cons=[x[0]==x0]; cost=0
    for k in range(H):
        cons+=[x[k+1]==Ad@x[k]+Bud@u[k]+Bdd@dseq[k]]
        cons+=[u[k]>=P_SRC_MIN-P0[0], u[k]<=P_SRC_MAX-P0[0]]
        for i in range(nN-1): cons+=[P0[i+1]+x[k+1,i]>=P_MIN-s[k,i]]
        cost+=cp.quad_form(x[k+1,:nN-1]/SC,np.eye(nN-1))+Rw*cp.square(u[k]/SC)+1e4*cp.sum(s[k]/SC)
    cost+=cp.quad_form(x[H,:nN-1]/SC, Qf[:nN-1,:nN-1]/1e10)
    return cp.Problem(cp.Minimize(cost),cons),x0,dseq,u

def run_closed(ctrl="mpc",H=15,Rw=0.05,x0est_err=0.0):
    net,props,P0,Q0,Ad,Bud,Bdd,Cd,Lobs,ns,nN,info=setup()
    d0=net.demand_nom.copy()
    if ctrl=="mpc":
        prob,px0,pd,vu=build_mpc(Ad,Bud,Bdd,P0,ns,nN,H,Rw)
    x=np.zeros(ns)                        # true plant deviation state
    xhat_pred=np.zeros(ns); xhat_pred[:nN-1]+=x0est_err   # a priori estimate
    nsteps=int(T_END/TS)
    tl,ul,pminl,eobsl=[],[],[],[]
    ei=0.0; uabs=P0[0]
    for step in range(nsteps):
        t=step*TS
        ymeas=Cd@x
        eobsl.append(np.linalg.norm(x-xhat_pred))         # a priori estimation error
        xhat=xhat_pred+Lobs@(ymeas-Cd@xhat_pred)          # measurement correction
        dnow=demand(t+TS/2,net); ddev=dnow-d0
        if ctrl=="mpc":
            px0.value=xhat
            pd.value=np.array([demand(t+j*TS,net)-d0 for j in range(H)])
            prob.solve(solver=cp.OSQP,warm_start=True,verbose=False,max_iter=8000,
                       eps_abs=1e-5,eps_rel=1e-5)
            du=float(vu.value[0,0])
        elif ctrl=="pid":
            # reactive one-sided pressure-support PI (boost only, no feedforward):
            # holds nominal compressor until the minimum pressure threatens the
            # limit, then raises it. Cannot anticipate the surge.
            pmin=(P0[1:]+x[:nN-1]).min()
            e=(P_MIN+0.7e5)-pmin           # Pa; positive when pressure too low
            ei=np.clip(ei+e*TS, 0, 6e8)    # integral, anti-windup
            du=np.clip(8.0*e+1.0e-3*ei, 0.0, P_SRC_MAX-P0[0])
        else:  # uncontrolled
            du=0.0
        uabs=P0[0]+du
        x=Ad@x+Bud[:,0]*du+Bdd@ddev
        xhat_pred=Ad@xhat+Bud[:,0]*du+Bdd@ddev            # observer time update
        tl.append((t+TS)/3600); ul.append(uabs/1e5)
        pminl.append((P0[1:]+x[:nN-1]).min()/1e5)
    tl=np.array(tl); ul=np.array(ul); pminl=np.array(pminl); eobsl=np.array(eobsl)
    # control energy = sum (du)^2 dt in bar^2 h
    energy=np.sum((ul-P0[0]/1e5)**2)*TS/3600
    return dict(t=tl,u=ul,pmin=pminl,eobs=eobsl,minP=pminl.min(),
                energy=energy,viol=(pminl<P_MIN/1e5-1e-3).mean()*100)

def make_figures(Hs,hres,Rs,rres,obs,comp):
    fig,axs=plt.subplots(1,2,figsize=(fs.W2, fs.H2))
    # (a) horizon sweep
    ax=axs[0]
    mn=[r['minP'] for r in hres]; en=[r['energy'] for r in hres]
    ax.plot(Hs,mn,'o-',color=C_RLC,lw=2,label="Min. delivery pressure")
    ax.axhline(58,color=C_ACC,ls="--",lw=1.2,label="Delivery limit")
    ax.set_xlabel("Prediction horizon $H$ [steps]"); ax.set_ylabel("Min. pressure [bar]",color=C_RLC)
    ax.tick_params(axis='y',colors=C_RLC); ax.set_title("(a) Horizon sweep"); ax.grid(alpha=.3)
    ax2=ax.twinx(); le,=ax2.plot(Hs,en,'s--',color=C_GRN,lw=1.6,label="Control effort")
    ax2.set_ylabel("Control effort [bar$^2$ h]",color=C_GRN); ax2.tick_params(axis='y',colors=C_GRN)
    h,l=ax.get_legend_handles_labels()
    fs.legend_below(ax,ncol=3,y=-0.30,handles=h+[le],labels=l+["Control effort"])
    # (b) R Pareto
    ax=axs[1]
    en=[r['energy'] for r in rres]; mn=[r['minP'] for r in rres]
    ax.plot(en,mn,'o-',color=C_RLC,lw=2,ms=3.5,label="Pareto front")
    # Label the Pareto points. Offsets are chosen from the LOCAL SPACING, not by
    # a fixed alternating rule: the two high-R points sit almost on top of each
    # other at the left end of the front, and a fixed +-9 pt vertical offset put
    # both labels underneath the markers, where they were unreadable. Crowded
    # points are labelled sideways with a leader line instead.
    # The per-point R labels are dropped. At the stiff end the
    # points nearly coincide, so the labels overlapped each other and the
    # front; the caption states that R falls from 1 at the lower-left to 0.005
    # at the upper-right along the front.
    ax.margins(x=0.10, y=0.18)
    ax.axhline(58,color=C_ACC,ls="--",lw=1.2,label="Delivery limit")
    ax.set_xlabel("Control effort [bar$^2$ h]")
    fs.ylabel(ax,"Min. pressure [bar]")
    ax.set_title("(b) Effort weight $R$ trade-off"); ax.grid(alpha=.3)
    fs.legend_below(ax,ncol=1,y=-0.30)
    # NOTE: the observer-convergence panel that used to sit here has been
    # removed. It showed the estimation error decaying to 1e-20 bar, which is
    # what a perfect model on a noiseless plant gives and says nothing about the
    # filter. The observer is now characterised in fig_estimator.png, under
    # transducer noise and an unknown demand disturbance.
    fs.save(fig, f"{FIG}/fig_mpc_analysis.png", legend_rows=2)
    print("saved fig_mpc_analysis.png")
    # controller comparison
    fig,ax=plt.subplots(figsize=(fs.W15, 3.2))
    ax.plot(comp['uncontrolled']['t'],comp['uncontrolled']['pmin'],color=C_OL,lw=2,label="Uncontrolled")
    ax.plot(comp['pid']['t'],comp['pid']['pmin'],color=C_GRN,lw=2,ls="-.",label="Reactive PID")
    ax.plot(comp['mpc']['t'],comp['mpc']['pmin'],color=C_RLC,lw=2,label="MPC (feed-forward)")
    ax.axhline(58,color=C_ACC,ls="--",lw=1.4,label="Delivery limit")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Min. node pressure [bar]")
    ax.set_title("Disturbance rejection: MPC vs PID"); ax.grid(alpha=.3)
    fs.legend_below(ax,ncol=2,y=-0.30)
    fs.save(fig, f"{FIG}/fig_controllers.png", legend_rows=2)
    print("saved fig_controllers.png")


if __name__=="__main__":
    # (1) horizon sweep
    print("Horizon sweep:")
    Hs=[3,5,10,15,20,30]; hres=[]
    for H in Hs:
        r=run_closed("mpc",H=H); hres.append(r)
        print(f"  H={H:2d}: minP={r['minP']:.2f} bar  energy={r['energy']:.2f}  viol={r['viol']:.1f}%")
    # (2) R sensitivity
    print("R (effort weight) sensitivity:")
    Rs=[0.005,0.02,0.05,0.2,1.0]; rres=[]
    for Rw in Rs:
        r=run_closed("mpc",H=15,Rw=Rw); rres.append(r)
        print(f"  R={Rw:5.3f}: minP={r['minP']:.2f} bar  energy={r['energy']:.2f}")
    # (3) observer convergence
    obs=run_closed("mpc",H=15,x0est_err=3.0e5)
    # (4) controller comparison
    print("Controller comparison:")
    comp={}
    for c in ["uncontrolled","pid","mpc"]:
        comp[c]=run_closed(c,H=15)
        print(f"  {c:12s}: minP={comp[c]['minP']:.2f} bar  viol={comp[c]['viol']:.1f}%  energy={comp[c]['energy']:.2f}")
    np.save("mpc_ana.npy",
            dict(Hs=Hs,hres=hres,Rs=Rs,rres=rres,obs=obs,comp=comp),allow_pickle=True)
    print("saved mpc_ana.npy")
    make_figures(Hs,hres,Rs,rres,obs,comp)
