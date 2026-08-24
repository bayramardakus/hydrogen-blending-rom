"""
blend_mpc.py
------------
Composition-aware predictive control: the genuinely hydrogen-specific problem.

An intermittent (renewable-driven) supply of green hydrogen is available for
injection at the source. We wish to absorb as much of it as possible, but the
hydrogen fraction delivered to every customer must stay within the interchangeability
window (<= 20 vol%). Because gas moves slowly, an injection change reaches distant
customers only after hours, so the delivered blend is a delayed, mixed image of the
injection history. A composition-aware MPC (using the transport model of
composition.py) anticipates this delay and injects the maximum safe amount; a
composition-blind reactive controller that simply injects whatever hydrogen is
available overshoots the limit once the front arrives.

We compare:
  * MPC        : composition-aware, maximises uptake subject to <=20% at every node
  * Reactive   : inject all available H2 (no anticipation)
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figstyle import apply_style, legend_below, PALETTE

from gas_properties import mixture_properties
from network import Network, steady_state
from composition import build_composition_model

import os as _os
FIGDIR = _os.environ.get("PAPER_FIGS", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "ms", "figs"))
_os.makedirs(FIGDIR, exist_ok=True)


apply_style()
C_REF=PALETTE["ref"]; C_RLC=PALETTE["rlc"]; C_ACC=PALETTE["acc"]; C_OL=PALETTE["ol"]; C_GRN=PALETTE["grn"]
FIG=FIGDIR

TS = 900.0             # 15 min control step (composition is slow)
T_END = 24*3600.0      # 24 h
H = 16                 # 4 h prediction horizon
Y_LIMIT = 0.20         # max delivered hydrogen fraction (interchangeability)
Y_BASE = 0.05          # baseline blend always injectable
DELIVERY_NODES = [2, 3, 4, 5]


def availability(t):
    """Intermittent green-H2 availability (fraction that COULD be injected)."""
    h = t/3600.0
    # midday renewable surplus 8h-17h pushes available injection up to 30%
    surplus = 0.30 if 8 <= h < 17 else Y_BASE
    # a short extra spike in the evening
    if 19 <= h < 21:
        surplus = 0.26
    return surplus


def discretize(A, B, Ts):
    n, m = A.shape[0], B.shape[1]
    Mb = np.zeros((n+m, n+m)); Mb[:n,:n]=A; Mb[:n,n:]=B
    Md = expm(Mb*Ts); return Md[:n,:n], Md[:n,n:]


def run_blend(controller="mpc"):
    net = Network(); props = mixture_properties(Y_BASE)
    P0, Q0 = steady_state(net, props)
    cm = build_composition_model(net, props, P0, Q0, y_nom=Y_BASE)
    Ay, By, Wnode = cm['Ay'], cm['By'], cm['Wnode']
    nP = net.n_pipes
    Ad, Bd = discretize(Ay, By, TS)          # composition discrete model
    # delivery-node blend selector rows
    Wdel = Wnode[DELIVERY_NODES, :]          # (nDel x nP)

    # ---- MPC problem (composition only; pressure handled separately) ----
    if controller == "mpc":
        y0p = cp.Parameter(nP)
        avail = cp.Parameter(H, nonneg=True)
        u = cp.Variable(H)                    # injection fraction each step
        y = cp.Variable((H+1, nP))
        s = cp.Variable((H, len(DELIVERY_NODES)), nonneg=True)
        uprev = cp.Parameter()
        cons = [y[0] == y0p]
        cost = 0
        for k in range(H):
            cons += [y[k+1] == Ad @ y[k] + (Bd[:,0]) * u[k]]
            cons += [u[k] >= Y_BASE, u[k] <= avail[k]]
            cons += [Wdel @ y[k+1] <= Y_LIMIT + s[k]]
            # maximise uptake (-u) + heavy penalty on exceeding limit
            cost += -1.0*u[k] + 1e3*cp.sum(s[k])
        # modest injection-rate penalty to avoid chattering
        cost += 8.0*cp.square(u[0]-uprev) + 8.0*cp.sum_squares(cp.diff(u))
        prob = cp.Problem(cp.Minimize(cost), cons)

    # ---- closed-loop over the plant (linear composition transport) ----
    y = np.full(nP, Y_BASE)
    u_last = Y_BASE
    nsteps = int(T_END/TS)
    tlog, ulog, ydel_log, avail_log = [], [], [], []
    for step in range(nsteps):
        t = step*TS
        av_now = availability(t)
        if controller == "mpc":
            y0p.value = y
            uprev.value = u_last
            avail.value = np.array([availability(t+j*TS) for j in range(H)])
            prob.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
            u_now = float(u.value[0]); u_last = u_now
        else:  # reactive: inject everything available (composition-blind)
            u_now = av_now
        # advance plant one step
        y = Ad @ y + Bd[:,0]*u_now
        ydel = Wdel @ y
        tlog.append((t+TS)/3600.0); ulog.append(u_now)
        ydel_log.append(ydel); avail_log.append(av_now)
    return (np.array(tlog), np.array(ulog), np.array(ydel_log),
            np.array(avail_log), DELIVERY_NODES)


def metrics(t_h, res):
    t_s = t_h*3600.0
    u = res[1]; ydel = res[2]; avail = res[3]
    absorbed = np.trapezoid(np.maximum(u-Y_BASE,0), t_s)
    offered = np.trapezoid(np.maximum(avail-Y_BASE,0), t_s)
    return dict(peak=ydel.max()*100, uptake_pct=100*absorbed/offered,
                viol_pct=(ydel.max(axis=1) > Y_LIMIT+1e-4).mean()*100)


def fig_blend(mpc, rea):
    t = mpc[0]
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.8))
    ax = axs[0]
    ax.fill_between(t, Y_BASE*100, mpc[3]*100, color="0.85",
                    label="Available green H$_2$", step="mid")
    ax.step(t, mpc[1]*100, where="mid", color=C_GRN, lw=2, label="MPC injection")
    ax.step(t, rea[1]*100, where="mid", color=C_OL, lw=1.4, ls="--", label="Reactive injection")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Injected H$_2$ fraction [vol%]")
    ax.set_title("(a) Green-H$_2$ injection scheduling"); ax.grid(alpha=.3)
    legend_below(ax, ncol=1, fontsize=11)
    ax = axs[1]
    ax.plot(t, rea[2].max(axis=1)*100, color=C_OL, lw=2, label="Reactive (worst node)")
    ax.plot(t, mpc[2].max(axis=1)*100, color=C_GRN, lw=2, label="MPC (worst node)")
    ax.axhline(Y_LIMIT*100, color=C_ACC, lw=1.5, ls="--", label="Interchangeability limit (20%)")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Delivered H$_2$ fraction [vol%]")
    ax.set_title("(b) Worst-case delivered blend"); ax.grid(alpha=.3)
    legend_below(ax, ncol=1, fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_blend_control.png"); plt.close(fig)
    print("saved fig_blend_control.png")


if __name__ == "__main__":
    mpc = run_blend("mpc")
    rea = run_blend("reactive")
    for name, res in [("MPC", mpc), ("Reactive", rea)]:
        m = metrics(res[0], res)
        print(f"{name:9s}: peak delivered blend={m['peak']:.1f}%  "
              f"green-H2 uptake={m['uptake_pct']:.1f}%  "
              f"time-in-violation={m['viol_pct']:.1f}%")
    fig_blend(mpc, rea)
